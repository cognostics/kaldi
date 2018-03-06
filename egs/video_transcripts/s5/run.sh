#!/bin/bash

# run training on Librispeech, Tedlium and VCTK

# data directory
data=/data/speech_recognition_data
nj=32  # number of jobs
mfccdir=mfcc # directory of where to store the mfcc files

# general config
stage=0

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh 

# exit once error occurs
set -e

if [ $stage -le 0 ]; then

        # begin data preparation (assumed that everything is downloaded)

        # libri data preparation
        echo "$0: Prepare librispeech data."

        for part in dev-clean test-clean dev-other test-other train-clean-100 \
                train-clean-360 train-other-500; do
                local/libri_data_prep.sh $data/LibriSpeech/$part \
                        data/libri_"${part//-/_}"
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


if [ $stage -le 1 ]; then
        # use the librispeech language model and add the dictionary entries
        # from ted which are contained in the lm_build github repository
        # download the language model trained on librispeech
        # we will not train our own system but use the one from librispeech
        local/libri_download_lm.sh "www.openslr.org/resources/11" data/local/libri_lang
        
        # when "--stage 3" option is used below we skip the G2P steps, and use the
        # lexicon we have already downloaded from openslr.org/11/
        local/libri_prepare_dict.sh --stage 3 --nj $nj --cmd "$train_cmd" \
           data/local/libri_lang data/local/libri_lang data/local/libri_dict

        # at a first instance just append the dictionaries to each other
        # without a new g2p calculation, as it takes time and is not sure to
        # give good results. The phoneme sets are not exactly the same, but
        # thankfully librispeech is a superset of the tedlium one and thus
        # appending and using the librispeech lm will do the trick.

        rm -rf data/local/dict_nosp data/local/lm
        mkdir -p data/local/dict_nosp

        echo "$0: Create new dictionary with TEDlium words and Librispeech lm."
        cp data/local/libri_dict/* data/local/dict_nosp/
        rm data/local/dict_nosp/lexicon_raw_nosil.txt
        (tr '(a-z)' '(A-Z)' < lm_build/data/local/dict_phn/lexicon.txt) >>\
                data/local/dict_nosp/lexicon.txt
        sort -u -o data/local/dict_nosp/lexicon.txt data/local/dict_nosp/lexicon.txt

        # lm_build uses NSN for spoken noise to map <UNK> to, we will delete
        # that line as to not mess anything up
        sed -i '/<UNK> NSN/d' data/local/dict_nosp/lexicon.txt

        # lm_build has tags for BREATH COUGH NOISE SMACK UH and UM in speech.
        # While this information could be exploited we will ignore this for now
        # and hope it just falls into the category spoken noise.
        for remove in "\[BREATH\] BRH" "\[COUGH\] CGH" "\[NOISE\] NSN"\
                "\[SMACK\] SMK" "\[UH\] UHH" "\[UM\] UM"; do
                sed -i "/$remove/d" data/local/dict_nosp/lexicon.txt
        done

        utils/prepare_lang.sh data/local/dict_nosp \
          "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

        cp -r data/local/libri_lang data/local/lm
        rm data/local/lm/librispeech-lexicon.txt
        rm data/local/lm/librispeech-vocab.txt

        local/format_lms.sh --src-dir data/lang_nosp data/local/lm

        # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
        utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
          data/lang_nosp data/lang_nosp_test_tglarge
        utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
          data/lang_nosp data/lang_nosp_test_fglarge
fi


if [ $stage -le 2 ]; then

        echo "$0: Get all the mfcc files."

        mkdir -p $mfccdir

        for part in libri_dev_clean libri_test_clean libri_dev_other\
                libri_test_other libri_train_clean_100 libri_train_clean_360\
                libri_train_other_500 ted_dev ted_test ted_train vctk; do
          steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$part exp/make_mfcc/$part $mfccdir
          steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
        done

fi


if [ $stage -le 3 ]; then

        # In this stage we will follow the librispeech run script in subsetting
        # the training data, so as to speed up the whole process. But it is
        # important to subset from all of the data (i.e. not just clean_100 as
        # in the script but on clean_100 plus vctk and ted_train

        echo "$0: Combine data sets and then subset them for initial training."

        utils/data/combine_data.sh data/train_clean data/libri_train_clean_100\
                data/vctk data/ted_train

        # For the monophone stages we select the shortest utterances, which should make it
        # easier to align the data from a flat start.

        utils/subset_data_dir.sh --shortest data/train_clean 2000 data/train_2kshort
        utils/subset_data_dir.sh data/train_clean 5000 data/train_5k
        utils/subset_data_dir.sh data/train_clean 10000 data/train_10k
        
fi

# Note the code for all the training is completely taken over from librispeech
# Remove all the decoding up until nnet3 training
# One stage comprises training + alignment of the data

if [ $stage -le 4 ]; then
        echo "$0: Train monophone model (tri1b)."

        # train a monophone system
        steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
          data/train_2kshort data/lang_nosp exp/mono         

        steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
          data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

        # train a first delta + delta-delta triphone system on a subset of 5000 utterances
        steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
            2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1 
         
        steps/align_si.sh --nj $nj --cmd "$train_cmd" \
          data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k

fi


if [ $stage -le 5 ]; then

        echo "$0: Train an LDA+MLLT system (tri2b)."

        # train an LDA+MLLT system.
        steps/train_lda_mllt.sh --cmd "$train_cmd" \
           --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
           data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b
         
        # Align a 10k utts subset using the tri2b model
        steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
          data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k

fi


if [ $stage -le 6 ]; then
        
        echo "$0: Train an LDA+MLLT+SAT system (tri3b)."

        # Train tri3b, which is LDA+MLLT+SAT on 10k utts
        steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
          data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b

        # align the entire train_clean_100 subset using the tri3b model
        steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
          data/train_clean data/lang_nosp \
          exp/tri3b exp/tri3b_ali_clean

fi


if [ $stage -le 7 ]; then

        echo "$0: Train another LDA+MLLT+SAT system with more data (tri4b)."

        # train another LDA+MLLT+SAT system on the entire 100 hour subset
        steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
          data/train_clean data/lang_nosp \
          exp/tri3b_ali_clean exp/tri4b

        # Now we compute the pronunciation and silence probabilities from training data,
        # and re-create the lang directory.
        steps/get_prons.sh --cmd "$train_cmd" \
          data/train_clean data/lang_nosp exp/tri4b
        utils/dict_dir_add_pronprobs.sh --max-normalize true \
          data/local/dict_nosp \
          exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
          exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

        utils/prepare_lang.sh data/local/dict \
          "<UNK>" data/local/lang_tmp data/lang
        local/format_lms.sh --src-dir data/lang data/local/lm

        utils/build_const_arpa_lm.sh \
          data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
        utils/build_const_arpa_lm.sh \
          data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

        # ... and then combine the two sets into a 460 hour one
        # Note that while it is called 460 the tedlium and vctk data is of
        # course contained, but for lack of a better name this will do
        utils/combine_data.sh \
          data/train_clean_460 data/train_clean data/libri_train_clean_360

        # align the new, combined set, using the tri4b model
        steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
          data/train_clean_460 data/lang exp/tri4b exp/tri4b_ali_clean_460

fi


if [ $stage -le 8 ]; then

        echo "$0: Train a larger SAT model trained on 460 hours + ted + vctk (tri5b)."

        # create a larger SAT model, trained on the 460 hours of data.
        steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
          data/train_clean_460 data/lang exp/tri4b_ali_clean_460 exp/tri5b

        # combine all the data
        utils/combine_data.sh \
          data/train_960 data/train_clean_460 data/libri_train_other_500

        steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
          data/train_960 data/lang exp/tri5b exp/tri5b_ali_960

fi


if [ $stage -le 9 ]; then

        echo "$0: Train a model on all the available data (tri6b)."

        # train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
        # as it is faster.
        steps/train_quick.sh --cmd "$train_cmd" \
          7000 150000 data/train_960 data/lang exp/tri5b_ali_960 exp/tri6b

        # decode using the tri6b model
        (
          utils/mkgraph.sh data/lang_test_tgsmall \
            exp/tri6b exp/tri6b/graph_tgsmall
          for test in test_clean test_other dev_clean dev_other; do
            steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
              exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
            steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
              data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test
            steps/lmrescore_const_arpa.sh \
              --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
              data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test
            steps/lmrescore_const_arpa.sh \
              --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
              data/$test exp/tri6b/decode_{tgsmall,fglarge}_$test
          done
        )&

        # this does some data-cleaning. The cleaned data should be useful when we add
        # the neural net and chain systems.
        local/run_cleanup_segmentation.sh

        steps/cleanup/debug_lexicon.sh --remove-stress true  --nj $nj --cmd "$train_cmd" data/train_clean \
           data/lang exp/tri6b data/local/dict/lexicon.txt exp/debug_lexicon
fi

# skip this for now as it has not been rewritten for all the data sets and is
# of questionable use anyway

# if [ $stage -le 10 ]; then
# 
#         echo "$0: Perform rescoring of tri6b by means of faster-rnnlm."
# 
#         #Perform rescoring of tri6b be means of faster-rnnlm using Noise contrastive estimation
#         #Note, that could be extremely slow without CUDA
#         #We use smaller direct layer size so that it could be stored in GPU memory (~2Gb)
#         #Suprisingly, bottleneck here is validation rather then learning
#         #Therefore you can use smaller validation dataset to speed up training
#         wait && local/run_rnnlm.sh \
#             --rnnlm-ver "faster-rnnlm" \
#             --rnnlm-options "-hidden 150 -direct 400 -direct-order 3 --nce 20" \
#             --rnnlm-tag "h150-me3-400-nce20" $data/LibriSpeech/lm data/local/lm
# 
# fi

if [ $stage -le 11 ]; then
       
        echo "$0: Train nnet3 model on entire data with cleaning."

        # train nnet3 tdnn models on the entire data with data-cleaning (xent and chain)
        local/chain/run_tdnn.sh --stage 11 # set "--stage 11" if you have already run local/nnet3/run_tdnn.sh

fi
