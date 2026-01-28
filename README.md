# HI-FAD
본 폴더는 HI-FAD에 대한 연구 코드를 저장합니다.
폴더의 구조는 다음과 같습니다.

```bash
DB/
├── LA/
│   ├── ASVspoof_LA_cm_protocols/
│   │    ├── ASVspoof2019.LA.cm.train.trn.txt
│   │    ├── ASVspoof2019.LA.cm.dev.trl.txt
│   │    ├── ASVspoof2021.LA.cm.eval.trl.txt
│   ├── ASVspoof2019_LA_train/
│   ├── ASVspoof2019_LA_dev/
│   ├── ASVspoof2019_LA_eval/
│   ├── ASVspoof2019_LA_asv_protocols/
│   ├── ASVspoof2019_LA_asv_scores/
│   ├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2021_LA_eval/
config/
├── AASIST_waveletencoder_frequency.conf
models/
├── AASIST_waveletencoder_frequency.py
```
---
## Dataset
본 실험은 ASVspoof 2021 dataset의 logical access (LA) 파티션에서 수행됩니다(2019 LA train database을 훈련하고 2021 LA eval database에서 평가).

### Download Dataset
The ASVspoof 2019 dataset는 본 링크에서 다운받을 수 있습니다 [here](https://datashare.ed.ac.uk/handle/10283/3336).

[ASVspoof 2021 LA database](https://zenodo.org/record/4837263#.YnDIinYzZhE)는 zenodo site에 release 되어 있습니다.

ASVspoof 2021 dataset keys (labels)과 metadata는 본 링크에서 다운받을 수 있습니다 [here](https://www.asvspoof.org/index2021.html).

혹은 아래의 코드를 통해 다운받을 수 있습니다.
```
cd DB/

# ASVspoof 2019 dataset
wget  https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y

# ASVspoof 2021 eval dataset
wget https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1

# ASVspoof 2021 EVALUATION KEYS AND METADATA
wget https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz

# Make LA_cm_protocols folder
mkdir ASVspoof_LA_cm_protocols
```

이후 ASVspoof2019.LA.cm.train.trn.txt, ASVspoof2019.LA.cm.dev.trl.txt, ASVspoof2021.LA.cm.eval.trl.txt 파일을 ASVspoof_LA_cm_protocols 폴더로 이동시킵니다.


## Train
```
# Train Our Model
python AASIST_waveletencoder_frequency.py --config config/AASIST_waveletencoder_frequency.conf

# Train Custom Model
python [YOUT MAIN FILE] --config config/[YOUT CONFIG FILE].conf
```

## Evaluation
evaluate.sh를 실행하기 전 Scores 파일 경로를 설정해야합니다.
ex) `python ./la_evaluate.py exp_result/[Scores 파일 경로] [LA keys 폴더 경로] eval`

```
bash evaluate.sh
```
---
