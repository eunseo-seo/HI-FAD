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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchcontrib.optim import SWA

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

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
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

    # make experiment reproducible
    set_seed(args.seed, config)

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
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    best_save_path = os.path.join(model_save_path, 'best')      # exp_retuls/<model_tag>/weights/best
    if not os.path.exists(best_save_path):
        os.makedirs(best_save_path,exist_ok=True)


    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    not_improving=0
    n_mejores= args.n_mejores_loss

    bests=np.ones(n_mejores,dtype=float)*float('inf')
    bests_eer=np.ones(n_mejores,dtype=float)*float('inf')
    best_loss=float('inf')
    best_eer=float('inf')


    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    if args.train:
        for i in range(n_mejores):
            np.savetxt( os.path.join(best_save_path, 'best_{}.pth'.format(i)), np.array((0,0)))
            np.savetxt( os.path.join(best_save_path, 'best_eer_{}.pth'.format(i)), np.array((0,0)))
        for epoch in range(config["num_epochs"]):
            print("Start training epoch{:03d}".format(epoch))
            running_loss = train_epoch(trn_loader, model, optimizer, device,
                                    scheduler, config)
            val_loss, eer = evaluate_accuracy(dev_loader, model, device)
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("val_loss", val_loss, epoch)
            writer.add_scalar("eer", eer*100, epoch)
            
            if val_loss<best_loss:
                best_loss=val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving=0
            else:
                not_improving+=1

            if eer <best_eer:
                best_eer = eer
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_eer.pth'))
                print('New best_eer epoch')

            for i in range(n_mejores):
                if bests[i]>val_loss:
                    for t in range(n_mejores-1,i,-1):
                        bests[t]=bests[t-1]
                        os.system('mv {}/best_{}.pth {}/best_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests[i]=val_loss
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_{}.pth'.format(i)))
                    break
            
            for i in range(n_mejores):
                if bests_eer[i]> eer:
                    for t in range(n_mejores-1,i,-1):
                        bests_eer[t]=bests_eer[t-1]
                        os.system('mv {}/best_eer_{}.pth {}/best_eer_{}.pth'.format(best_save_path, t-1, best_save_path, t))
                    bests_eer[i]=eer
                    torch.save(model.state_dict(), os.path.join(best_save_path, 'best_eer_{}.pth'.format(i)))
                    break
                
            print('\n epoch {} - running_loss {} - val_loss {} - EER {}'.format(epoch, running_loss, val_loss, eer*100))
            print('n-best loss:', bests)
            print('n-best eer loss:', bests_eer)
        print('Total epochs: ' + str(epoch) +'\n')

    print('######## Eval ########')
    if args.average_model:
        sdl=[]
        model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(0)), map_location=device))
        print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(0))))
        sd = model.state_dict()
        for i in range(1,args.n_average_model):
            model.load_state_dict(torch.load(os.path.join(best_save_path, 'best_{}.pth'.format(i)), map_location=device))
            print('Model loaded : {}'.format(os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key]=(sd[key]+sd2[key])
        for key in sd:
            sd[key]=(sd[key])/args.n_average_model
        model.load_state_dict(sd)
        print('Model loaded average of {} best models in {}'.format(args.n_average_model, best_save_path))
    else:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best.pth'), map_location=device))
        print('Model loaded : {}'.format(os.path.join(model_save_path, 'best.pth')))

    if args.comment_eval:
        eval_score_path = eval_score_path.name + '_{}'.format(args.comment_eval)

    print("Start evaluation...")
    if not os.path.exists('Scores/{}.txt'.format(eval_score_path)):
        produce_evaluation_file(eval_loader, model, device, eval_score_path)
        print("DONE.")
    else:
        print('Score file already exists')


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
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
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
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
    print("no. validation files:", len(file_dev))

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

    return trn_loader, dev_loader, eval_loader

def evaluate_accuracy(
    dev_loader: DataLoader, 
    model, 
    device : torch.device) -> None:

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

            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            for score, label in zip(batch_score, batch_y.cpu().numpy()):
                if label == 0:  # spoof
                    spoof_scores.append(score)
                else:  # bonafide
                    bona_scores.append(score)


    val_loss /= num_total
    eer = compute_eer(np.array(bona_scores), np.array(spoof_scores))[0]     # eer_cm = compute_eer(bona_cm, spoof_cm)[0]
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
            batch_out = model(batch_x)
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
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))

        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

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
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
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

    main(parser.parse_args())
