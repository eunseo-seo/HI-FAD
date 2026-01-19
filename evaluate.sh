#!/usr/bin/env bash

set -eu
# python ./la_evaluate.py ./exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt  /data3/DB/FakeAudio/LA/keys/LA eval
python ./la_evaluate.py ./exp_result/waveletencoder_frequency/LA_AASIST_waveletencoder_frequency_ep100_bs8/eval_scores_using_best_dev_model_waveletencoder_frequency.txt /data3/DB/FakeAudio/LA/keys/LA eval
