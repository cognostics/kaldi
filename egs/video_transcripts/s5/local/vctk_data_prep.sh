#!/bin/bash

# Prepare data for the VCTK corpus.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

spk_file=$src/speaker-info.txt

mkdir -p "$dst" || exit 1;
mkdir -p "$src/wav16" || exit 1;

[ ! -d "$src" ] && echo "$0: no such directory $src" && exit 1;
[ ! -f "$spk_file" ] && echo "$0: expected file $spk_file to exist" && exit 1;

if ! which ffmpeg >&/dev/null; then
   echo "Please install 'ffmpeg' on ALL worker nodes!"
   exit 1
fi


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm "$wav_scp"
trans=$dst/text; [[ -f "$trans" ]] && rm "$trans"
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm "$utt2spk"
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm "$spk2gender"
utt2dur=$dst/utt2dur; [[ -f "$utt2dur" ]] && rm "$utt2dur"


while IFS=$'\n' read -r line; do
        echo "$line"
        reader=$(awk <<< "$line" '{ print $1 }')
        [ "$reader" = "ID" ] && continue  # skip the first line
        [ "$reader" -eq 315 ] && continue  # speaker 315 has no transcripts..
        # change the gender to lowercase
        gender=$(awk <<< "$line" '{ print $3 }' | tr '(A-Z)' '(a-z)')
        echo "$reader $gender" >> "$spk2gender"

        mkdir -p "$src/wav16/p$reader"
        
        for wav in $src/wav48/p$reader/*.wav; do
                # get the basename without file extension which is a unique
                # identifier for the utterance
                utterance=$(basename "$wav")
                utterance="${utterance%.wav}"

                # the uttid uses hyphens not underscores
                uttid=${utterance//_/-}

                # get the fitting transcript
                text=$(<"$src/txt/p$reader/$utterance.txt") || exit 1;
                # remove any punctuation from the text
                text=${text//[,.]/}
                # capitalize all letters, since it seems to be standard
                text=$( echo "$text" | tr '(a-z)' '(A-Z)' )

                wav_o="$src/wav16/p$reader/$utterance.wav"

                # conversion command for the wav to pipe the correct bitrate of
                # 16kHz
                if [ ! -f "$wav_o" ]; then
                        sox "$wav" -r 16000 "$wav_o"
                fi

                # create all the remaining data files
                echo "$uttid $text" >> "$trans"
                echo "$uttid $wav_o" >> "$wav_scp"
                echo "$uttid $reader" >> "$utt2spk"
        done
done < "$spk_file"

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <"$utt2spk" >"$spk2utt" || exit 1

ntrans=$(wc -l <"$trans")
nutt2spk=$(wc -l <"$utt2spk")
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

utils/data/get_utt2dur.sh "$dst" 1>&2 || exit 1

utils/validate_data_dir.sh --no-feats "$dst" || exit 1;

echo "$0: successfully prepared data in $dst"

exit 0
