#!/bin/bash

# run training on Librispeech, Tedlium and VCTK

# data directory
data=/data/speech_recognition_data

# general config
stage=0

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh 

# exit once error occurs
set -e

if [ $stage -le 0 ]; then

        # download the language model trained on librispeech
        # we will not train our own system but use the one from librispeech
#        local/libri_download_lm.sh "www.openslr.org/resources/11" data/local/lm

        # begin data preparation (assumed that everything is downloaded)

        # libri data preparation
        echo "$0: Prepare librispeech data."

        for part in dev-clean test-clean dev-other test-other train-clean-100 \
                train-clean-360 train-other-500; do
                local/libri_data_prep.sh $data/LibriSpeech/$part \
                        data/libri_"$(echo $part | sed s/-/_/g)"
        done

        echo "$0: Prepare tedlium data."
        # tedlium data preparation
        local/ted_data_prep.sh

        # split speakers up into 3-minute chunks. This doesn't hurt adaptation,
        # and lets us use more jobs for decoding etc.
        for part in dev test train; do
                utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 \
                data/$part.orig data/ted_$part
        done

        echo "$0: Prepare VCTK data."
        local/vctk_data_prep.sh $data/VCTK-Corpus data/vctk
        
fi
