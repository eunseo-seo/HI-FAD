"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchcontrib.optim import SWA
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2021_eval, genSpoof_list)
from evaluation import calculate_tDCF_EER, compute_eer
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_attention(att_map, name='attn', epoch=0, writer=None):
    """
    Plot a heatmap of an attention map (2D numpy array) using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(att_map, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'{name} @ epoch {epoch}')
    fig.colorbar(im, ax=ax)

    if writer:
        writer.add_figure(name, fig, global_step=epoch)

    plt.close(fig)

def main_worker(rank, world_size, args):
    # DDP 초기화
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            raise ValueError('GPU not detected!')

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible (rank별 seed)
    set_seed(args.seed + rank, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2021))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    if rank == 0:
        writer = SummaryWriter(model_tag)
        os.makedirs(model_save_path, exist_ok=True)
        copy(args.config, model_tag / "config.conf")
    else:
        writer = None
    dist.barrier() if world_size > 1 else None

    # define model architecture
    model = get_model(model_config, device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)

    # define dataloaders (train에만 DistributedSampler 적용)
    trn_loader, dev_loader, eval_loader, train_set = get_loader_ddp(
        database_path, args.seed, config, world_size, rank)

    best_save_path = os.path.join(model_save_path, 'best')
    if rank == 0 and not os.path.exists(best_save_path):
        os.makedirs(best_save_path, exist_ok=True)
    dist.barrier() if world_size > 1 else None

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    not_improving = 0
    n_mejores = args.n_mejores_loss
    bests = np.ones(n_mejores, dtype=float) * float('inf')
    bests_eer = np.ones(n_mejores, dtype=float) * float('inf')
    best_loss = float('inf')
    best_eer = float('inf')

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    if rank == 0:
        os.makedirs(metric_path, exist_ok=True)
    dist.barrier() if world_size > 1 else None

    # Training
    if args.train:
        if rank == 0:
            for i in range(n_mejores):
                np.savetxt(os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0, 0)))
                np.savetxt( os.path.join(best_save_path, 'best_eer_{}.pth'.format(i)), np.array((0,0)))
        dist.barrier() if world_size > 1 else None

        for epoch in range(config["num_epochs"]):
            if world_size > 1:
                trn_loader.sampler.set_epoch(epoch)
            print(f"[Rank {rank}] Start training epoch{epoch:03d}")
            loss_dict = train_epoch(trn_loader, model, optimizer, device, scheduler, config, rank=rank)
            running_loss = loss_dict['total']
            val_loss, eer = evaluate_accuracy(dev_loader, model, device, world_size=world_size)
            if rank == 0 and writer:
                # Log total loss
                writer.add_scalar("train/loss_total", running_loss, epoch)
                # Log individual losses
                writer.add_scalar("train/loss_em", loss_dict['loss_em'], epoch)
                writer.add_scalar("train/loss_org", loss_dict['loss_org'], epoch)
                writer.add_scalar("train/loss_kd", loss_dict['loss_kd'], epoch)
                # Log validation metrics
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/eer", eer * 100, epoch)

            if rank == 0:
                # Get student-only state dict for saving
                student_state_dict = get_student_state_dict(model)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(student_state_dict, os.path.join(model_save_path, 'best.pth'))
                    print('New best epoch (saved student model only)')
                    not_improving = 0
                else:
                    not_improving += 1
                if eer <best_eer:
                    best_eer = eer
                    torch.save(student_state_dict, os.path.join(model_save_path, 'best_eer.pth'))
                    print('New best_eer epoch (saved student model only)')
                    
                for i in range(n_mejores):
                    if bests[i] > val_loss:
                        for t in range(n_mejores - 1, i, -1):
                            bests[t] = bests[t - 1]
                            os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t - 1, best_save_path, t))
                        bests[i] = val_loss
                        torch.save(student_state_dict, os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                        break
                
                for i in range(n_mejores):
                    if bests_eer[i]> eer:
                        for t in range(n_mejores-1,i,-1):
                            bests_eer[t]=bests_eer[t-1]
                            os.system('mv {}/best_eer_{}.pth {}/best_eer_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                        bests_eer[i]=eer
                        torch.save(student_state_dict, os.path.join(best_save_path, 'best_eer_{}.pth'.format(i)))
                        break
                
                print(f'\n epoch {epoch} - total_loss {running_loss:.4f} (em: {loss_dict["loss_em"]:.4f}, org: {loss_dict["loss_org"]:.4f}, kd: {loss_dict["loss_kd"]:.4f}) - val_loss {val_loss:.4f} - EER {eer * 100:.2f}')
                print('n-best loss:', bests)
                print('n-best eer loss:', bests_eer)
            dist.barrier() if world_size > 1 else None
        if rank == 0:
            print('Total epochs: ' + str(epoch) + '\n')

    if rank == 0:
        print('######## Eval ########')
    if args.average_model:
        if rank == 0:
            # Load student-only weights (strict=False to ignore missing teacher components)
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0)), map_location=device), strict=False)
            print('Student model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
            sd = torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0)), map_location=device)
            for i in range(1, args.n_average_model):
                sd2 = torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i)), map_location=device)
                print('Student model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
                for key in sd:
                    if key in sd2:
                        sd[key] = (sd[key] + sd2[key])
            for key in sd:
                sd[key] = (sd[key]) / args.n_average_model
            model.load_state_dict(sd, strict=False)
            print('Student model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
        dist.barrier() if world_size > 1 else None
    else:
        if rank == 0:
            model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth'), map_location=device), strict=False)
            print('Student model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))
        dist.barrier() if world_size > 1 else None

    if args.comment_eval and rank == 0:
        eval_score_path = eval_score_path.name + '_{}'.format(args.comment_eval)

    if rank == 0:
        print("Start evaluation...")
        if not os.path.exists('Scores/{}.txt'.format(eval_score_path)):
            produce_evaluation_file(eval_loader, model, device, eval_score_path)
            print("DONE.")
        else:
            print('Score file already exists')
    dist.barrier() if world_size > 1 else None

    if world_size > 1:
        dist.destroy_process_group()

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    
    # Total model parameters (including teacher components)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    
    # Student-only parameters (excluding teacher components)
    teacher_only_prefixes = ['oracle_encoder', 'fusion_module', 'wav2vec2', 'ssl_proj']
    nb_student_params = sum([
        param.view(-1).size()[0] for name, param in model.named_parameters()
        if not any(name.startswith(p) for p in teacher_only_prefixes)
    ])
    
    # Teacher-only parameters
    nb_teacher_params = nb_params - nb_student_params
    
    print("=" * 50)
    print(f"Total model params: {nb_params:,}")
    print(f"  - Student (AASIST) params: {nb_student_params:,}")
    print(f"  - Teacher-only params: {nb_teacher_params:,}")
    print("=" * 50)

    return model


def get_student_state_dict(model):
    """
    Extract only the student model parameters (excluding teacher-only components).
    Teacher-only components: oracle_encoder, fusion_module, wav2vec2, ssl_proj
    """
    # Handle DDP wrapped model
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
        prefix = ''
    else:
        state_dict = model.state_dict()
        prefix = ''
    
    # Components to exclude (teacher-only)
    teacher_only_prefixes = ['oracle_encoder', 'fusion_module', 'wav2vec2', 'ssl_proj']
    
    student_state_dict = {}
    for key, value in state_dict.items():
        # Check if this key belongs to teacher-only components
        is_teacher_only = any(key.startswith(p) or f'.{p}' in key for p in teacher_only_prefixes)
        if not is_teacher_only:
            student_state_dict[key] = value
    
    return student_state_dict

def get_loader_ddp(database_path, seed, config, world_size, rank):
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2021_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2021))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print(f"[Rank {rank}] no. training files: {len(file_train)}")

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed + rank)
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)
    else:
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)

    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print(f"[Rank {rank}] no. validation files: {len(file_dev)}")

    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev, labels=d_label_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader, train_set

def evaluate_accuracy(
    dev_loader: DataLoader, 
    model, 
    device : torch.device,
    world_size=1
) -> tuple:
    import torch.distributed as dist
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    bona_scores = []
    spoof_scores = []

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Forward pass - EM-Network returns tuple during eval (student_hidden, student_output)
            outputs = model(batch_x, mode='eval')
            
            # Check if model returns tuple (EM-Network) or single output (regular AASIST)
            if isinstance(outputs, tuple):
                # EM-Network: (student_hidden, student_output) during eval
                batch_out = outputs[1]  # Use student_output
            else:
                # Regular AASIST: single output
                batch_out = outputs
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
            for score, label in zip(batch_score, batch_y.cpu().numpy()):
                if label == 0:  # spoof
                    spoof_scores.append(score)
                else:  # bonafide
                    bona_scores.append(score)

    # --- DDP: all_gather ---
    val_loss_tensor = torch.tensor([val_loss], device=device)
    num_total_tensor = torch.tensor([num_total], device=device)
    if world_size > 1:
        gathered_loss = [torch.zeros_like(val_loss_tensor) for _ in range(world_size)]
        gathered_total = [torch.zeros_like(num_total_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_loss, val_loss_tensor)
        dist.all_gather(gathered_total, num_total_tensor)
        val_loss = sum([t.item() for t in gathered_loss])
        num_total = sum([t.item() for t in gathered_total])
    val_loss /= num_total if num_total > 0 else 1

    # bona_scores, spoof_scores도 all_gather 필요
    bona_scores_tensor = torch.tensor(bona_scores, device=device)
    spoof_scores_tensor = torch.tensor(spoof_scores, device=device)
    if world_size > 1:
        # gather all scores (variable length, so gather_object)
        bona_gathered = [None for _ in range(world_size)]
        spoof_gathered = [None for _ in range(world_size)]
        dist.all_gather_object(bona_gathered, bona_scores)
        dist.all_gather_object(spoof_gathered, spoof_scores)
        bona_scores = np.concatenate(bona_gathered)
        spoof_scores = np.concatenate(spoof_gathered)
    else:
        bona_scores = np.array(bona_scores)
        spoof_scores = np.array(spoof_scores)

    eer = compute_eer(bona_scores, spoof_scores)[0]
    return val_loss, eer
                    
def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            # Forward pass - EM-Network returns tuple during eval (student_hidden, student_output)
            outputs = model(batch_x, mode='eval')
            
            # Check if model returns tuple (EM-Network) or single output (regular AASIST)
            if isinstance(outputs, tuple):
                # EM-Network: (student_hidden, student_output) during eval
                batch_out = outputs[1]  # Use student_output
            else:
                # Regular AASIST: single output
                batch_out = outputs
            
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list): 
            fh.write("{} {}\n".format(fn, sco))
        fh.close()
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    rank=None
):
    """Train the model for one epoch"""
    running_loss = 0
    running_loss_em = 0
    running_loss_org = 0
    running_loss_kd = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion_cls = nn.CrossEntropyLoss(weight=weight)
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    # Get KD hyperparameters from config (support both dict and Namespace)
    if isinstance(config, dict):
        temperature = config.get("kd_temperature", 4.0)
        alpha = config.get("kd_alpha", 0.5)
    else:
        temperature = getattr(config, "kd_temperature", 4.0)
        alpha = getattr(config, "kd_alpha", 0.5)  # Weight for KD loss
    
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # Forward pass - EM-Network MUST return tuple of 4 values during training
        outputs = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]), mode='train')
        
        # EM-Network MUST return (student_hidden, student_output, teacher_hidden, teacher_output)
        if not isinstance(outputs, tuple) or len(outputs) != 4:
            raise RuntimeError(
                f"EM-Network model must return tuple of 4 values during training. "
                f"Got: {type(outputs)}, length: {len(outputs) if isinstance(outputs, tuple) else 'N/A'}. "
                f"This indicates Oracle Guidance is missing or model is not in training mode."
            )
        
        student_hidden, student_output, teacher_hidden, teacher_output = outputs
        
        # Objective 1: L_em(φ) - Teacher (EM-Network) classification loss
        loss_em = criterion_cls(teacher_output, batch_y)
        
        # Objective 2: L_org(θ) - Student (original sequence model) classification loss
        loss_org = criterion_cls(student_output, batch_y)
        
        # L_kd(φ, θ) - Knowledge distillation loss (Student learns from Teacher)
        teacher_soft = F.softmax(teacher_output.detach() / temperature, dim=1)
        student_log_soft = F.log_softmax(student_output / temperature, dim=1)
        loss_kd = criterion_kd(student_log_soft, teacher_soft) * (temperature ** 2)
        
        # Total loss: L_em + L_org + α * L_kd
        batch_loss = loss_em + loss_org + alpha * loss_kd
        
        running_loss += batch_loss.item() * batch_size
        running_loss_em += loss_em.item() * batch_size
        running_loss_org += loss_org.item() * batch_size
        running_loss_kd += loss_kd.item() * batch_size

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    running_loss_em /= num_total
    running_loss_org /= num_total
    running_loss_kd /= num_total
    
    return {
        'total': running_loss,
        'loss_em': running_loss_em,
        'loss_org': running_loss_org,
        'loss_kd': running_loss_kd
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default=None,
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    
    #Eval
    parser.add_argument('--n_mejores_loss', type=int, default=5, help='save the n-best models')
    parser.add_argument('--average_model', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether average the weight of the n_best epochs')
    parser.add_argument('--n_average_model', default=5, type=int)

    parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
    parser.add_argument('--comment_eval',default=None, type= str)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12223'
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
    else:
        main_worker(0, 1, args)
