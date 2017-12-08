#!/bin/bash

# This script demonstrates kaldi using pretrained models. It will decode videos
# in a supplied directory. This script is adapted from the apiai_decode
# example.

# This script tries to follow kaldis general example file structure, so please
# be mindful of that when making changes.

. ./path.sh
. ./cmd.sh

shopt -s globstar
export LC_ALL=C

# begin general config
MODEL_DIR="exp/chain_cleaned/tdnn_1b_sp"
DATA_MAIN_DIR="/data/speech_recognition_data"
DATA_SUB_DIR="lectures"
DATA_DIR="$DATA_MAIN_DIR/$DATA_SUB_DIR"
VIDEO_FILES="mp4"  # directory such that $DATA_DIR/$VIDEO_DIR is the video dir
VIDEO_SUFFIX=".$VIDEO_FILES"  # file ending for video files
TEXT_SUFFIX=".txt"
META_DATA="data/$DATA_SUB_DIR"
SPLIT_SUFFIX="_split"

LANG="data/lang_chain"

GRAPH_DIR="exp/chain_cleaned/tdnn_1b_sp/graph_tgsmall"
HCLG="$GRAPH_DIR/HCLG.fst"
WORDS="$GRAPH_DIR/words.txt"
SUBSAMPLING="$MODEL_DIR/frame_subsampling_factor"

# pass this as true to search for txt files and compare decoding results
txt_compare=false

iter="final"  # use the final iteration that was trained
job_number=60
stage=0  # for in between starting
# stage 0 is beginning, converting wavs and (re)writing meta data
# stage 1 is extracting mfcc and cmvn features
# stage 2 is the fmllr transform

# options for splitting
overlap=5
window=30

. utils/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
        # begin sanity checks
        for app in nnet3-latgen-faster apply-cmvn lattice-scale; do
            command -v $app >/dev/null 2>&1 || { echo >&2 "$app not found, is kaldi compiled?"; exit 1; }
        done;

        command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "ffmpeg not found, please install"; exit 1; }

        # check that all of the following files exist (they are all we need for 
        # decoding)
        [ ! -d "$DATA_DIR/$VIDEO_FILES" ] && echo \ "$0: expected $DATA_DIR/$VIDEO_FILES to exist" && exit 1;

        for file in "$MODEL_DIR/$iter.mdl" "$HCLG" "$WORDS" "$SUBSAMPLING"; do
            [ ! -f "$file" ] && echo \ "$0: expected $file to exist" && exit 1;
        done;

        # maybe mkdir the necessary dirs to which we will extract all our intermediate results
        for dir in "$DATA_DIR/wav" "exp/make_mfcc/$DATA_SUB_DIR" "mfcc" "data" \
                "$META_DATA"; do
             if [ ! -d "$dir" ]; then
                mkdir "$dir"
             fi
        done;

        # rewrite any of the meta files from scratch
        for meta_file in "wav.scp" "spk2utt" "utt2spk"; do
                if [ -f "$META_DATA/$meta_file" ]; then
                        rm "$META_DATA/$meta_file"
                fi
                touch $META_DATA/$meta_file
        done;

        # remove the previously created split data directory
        [ -d "$META_DATA$SPLIT_SUFFIX" ] && rm -r "$META_DATA$SPLIT_SUFFIX"
        [ -d "$META_DATA${SPLIT_SUFFIX}_max2" ] && rm -r "$META_DATA${SPLIT_SUFFIX}_max2"

        # finally replace any spaces in video files names with hyphens
        find $DATA_DIR/$VIDEO_FILES -name "* *$VIDEO_SUFFIX" -execdir rename 's/ /-/g' "{}" \;
        # and remove any versioning addition to the names
        for file in $DATA_DIR/$VIDEO_FILES/**/*-v[1-9].[1-9]$VIDEO_SUFFIX; do
                mv "$file" "${file%-v[1-9].[1-9]$VIDEO_SUFFIX}$VIDEO_SUFFIX"
        done

        # end sanity checks

        # maybe convert the video files to 16kHz 16-bit wav files in little-endian format
        for file in $DATA_DIR/$VIDEO_FILES/**/*$VIDEO_SUFFIX; do
                basename=${file##*/}  # remove any path apart from basename
                basename="${basename%$VIDEO_SUFFIX}"  # convert foo.txt to foo
                audiofile="$DATA_DIR/wav/$basename.wav"
                # if you cannot find the audio file
                if [ ! -f "$audiofile" ]; then
                        # ffmpeg call to convert to audio
                        echo "$0: convert $file to $audiofile"
                        ffmpeg -i "$file" -vn -acodec pcm_s16le -ar 16000 -ac 1 \
                        "$audiofile" || { echo "$0: Unable to convert video to audio.";
                        exit 1; }
                fi
                # populate the meta data files with the files handled here.
                speaker_id=${basename%-*}
                speaker_id=${speaker_id%-*}
                # Think of the example file with the name
                # 17S1_VL_MS714M_Prof_Thirumany_Module_1_Lecture_5.mp4
                # basename removes the .mp4 and $speaker_id is the unique identifier for
                # the whole module, i.e. everything up until including Module_1.
                # The utterance ID then includes the lecture name again, i.e. is just
                # the $basename.
                if $txt_compare; then
                        echo "$0: Split the .wav files based on the found text files."               
                        text_file="${file%$VIDEO_SUFFIX}$TEXT_SUFFIX"
                        # only handle the file if it has a transcript
                        if [ -f "$text_file" ]; then
                               # the following variable keeps track of the
                               # offset due to errors in the transcript
                               declare -i offset=0
                               touch "$META_DATA/segments" "$META_DATA/text"
                               # the following code goes through the transcript
                               # line by line and extracts the timings. The
                               # timings are then appended to $basename in
                               # order to form the new utterance id and
                               # appended to the segments and text file.
                               while IFS= read -r line; do
                                       if [ ! -z "$line" ]; then
                                                text=${line% *}  # this will just be the text
                                                timestamp=${line##* }  # (min:sec-min:sec)
                                                begin=${timestamp%-*}
                                                begin=${begin##(}
                                                # convert begin time to seconds
                                                begin=$(echo "$begin" | awk -F. '{ print ($1 * 60) + $2 }')
                                                end=${timestamp##*-}
                                                end=${end%%)}
                                                # convert end time to seconds
                                                end=$(echo "$end" | awk -F. '{ print ($1 * 60) + $2 }')
                                                # sometimes the minutes are not
                                                # incremented, do this makeshift
                                                # fix for this
                                                if [ "$end" -le  "$begin" ]; then
                                                        offset=$offset+1
                                                fi
                                                end=$((end + 60*offset))
                                                # save the utt id with 6 digits
                                                uttid="${basename}-$(printf "%06d" "$begin")-$(printf "%06d" "$end")"
                                                echo "$uttid $text" >> $META_DATA/text
                                                echo "$uttid $basename $(printf "%06d" "$begin") $(printf "%06d" "$end")" >> $META_DATA/segments
                                                echo "$uttid $speaker_id" >> $META_DATA/utt2spk
                                               
                                        fi
                               done < "$text_file"
                               echo "$basename $audiofile" >> $META_DATA/wav.scp
                        fi
                else
                        echo "$basename $speaker_id" >> $META_DATA/utt2spk
                        echo "$basename $audiofile" >> $META_DATA/wav.scp
                fi
                utils/utt2spk_to_spk2utt.pl $META_DATA/utt2spk > $META_DATA/spk2utt || exit 1
        done
        utils/data/fix_data_dir.sh $META_DATA

        if ! $txt_compare; then
                echo "$0: Split the .wav files into manageable portions."

                steps/cleanup/split_long_utterance_wo_text.sh --seg-length $window --overlap-length $overlap \
                        $META_DATA $META_DATA$SPLIT_SUFFIX || exit 1;
        else
                utils/data/copy_data_dir.sh $META_DATA $META_DATA$SPLIT_SUFFIX || exit 1;
        fi
        # split the data based on utterances, to allow for more efficient computation
        utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
          $META_DATA$SPLIT_SUFFIX "$META_DATA${SPLIT_SUFFIX}_max2"
fi

if [ $stage -le 1 ]; then
        echo "$0: Computing mfcc and cmvn"

        steps/make_mfcc.sh --nj $job_number --mfcc-config conf/mfcc_hires.conf \
              --cmd "$decode_cmd" "$META_DATA${SPLIT_SUFFIX}_max2" || { 
                echo "$0: Unable to calculate mfcc, ensure 16kHz, 16 bit little-endian wav format or see log";
                exit 1; };
            steps/compute_cmvn_stats.sh $META_DATA${SPLIT_SUFFIX}_max2 || exit 1;
fi


# if [ $stage -le 2 ]; then
#         echo "$0: Compute fmllr transforms."
#         # decode using the tri6b model from the librispeech training run, i.e. perform
#         # fmllr transform
# 
#         steps/decode_fmllr.sh --nj "$job_number" --cmd "$decode_cmd" --skip_scoring "true" \
#                 exp/nnet3_cleaned/tri7b \
#                 $META_DATA${SPLIT_SUFFIX}_max2 \
#                 exp/tri6b_cleaned/decode_tri7b_$DATA_SUB_DIR || exit 1;
# fi

if [ $stage -le 2 ]; then
        echo "$0: Extract ivectors."
        steps/online/nnet3/prepare_online_decoding.sh  --cmd "$train_cmd" --nj "$job_number" --mfcc_config "conf/mfcc_hires.conf" \
         $LANG exp/nnet3_cleaned/extractor exp/nnet3_cleaned/tri7b "output1" 
#         steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj "$job_number" \
#           $META_DATA${SPLIT_SUFFIX}_max2 $LANG exp/nnet3_cleaned/extractor \
#           exp/nnet3_cleaned/tri7b \
#           exp/nnet3_cleaned/ivectors_${DATA_SUB_DIR} || exit 1;
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $job_number \
        --config exp/nnet3_cleaned/tri7b/conf/ivector_extractor.conf \
        $META_DATA${SPLIT_SUFFIX}_max2 exp/nnet3_cleaned/extractor \
        exp/nnet3_cleaned/ivectors_${DATA_SUB_DIR} || exit 1;
fi

if [ $stage -le 3 ]; then
      echo "$0: Decode the audio."
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $job_number --cmd "$decode_cmd" --iter $iter \
          --online-ivector-dir exp/nnet3_cleaned/ivectors_${DATA_SUB_DIR} \
          $GRAPH_DIR $META_DATA${SPLIT_SUFFIX}_max2 \
          $MODEL_DIR/decode_${DATA_SUB_DIR} || exit 1
fi
