#!/bin/bash


# some data-prep flags..
AUGMENT=false # indicates, whether to peform augmentation. Kaldi has to be set up first.
# repo_root=$PWD #NOTE: you can change this to match your system...
#repo_root="/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model"
repo_root="/dataHDD1/timkleinlein/among-us-analysis/Speech-Extraction/Personal_VAD_Model"
KALDI=$repo_root/kaldi
nj_features=4 # number of CPU workers used for feature extraction
utt_count=10000 # the number of generated utterances
kaldi_root=$repo_root/kaldi/egs/pvad # only valid if Kaldi is set up..
feature_dir_name=features_test # change this to whatever you want your feature dir to be named...

# Here, you can specify the LibriSpeech folders used by the data preparation scripts.
# Make sure that you have the subsets downloaded and unzipped before using them..
libri_folders=()
libri_folders+="dev-clean"
#libri_folders+=" dev-other"
#libri_folders+=" test-clean"
#libri_folders+=" test-other"
#libri_folders+=" train-clean-100"
#libri_folders+=" train-clean-360"
#libri_folders+=" train-other-500"

# set the NAME env variable prior to running this script if you wish to use a different
# name for the concatenated dataset folder than 'clean'
if [ -z ${NAME+x} ]; then
  export NAME=test # name is empty, default to 'clean'
fi

#==================================================

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# set the concatenated utterance dataset destination directory
if [ "$AUGMENT" = true ]; then
  mkdir -p $repo_root/kaldi/egs/pvad/data
  concat_dir=$repo_root/kaldi/egs/pvad/data/$NAME
else
  concat_dir=$repo_root/data/$NAME
fi

#cd "/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/Personal_VAD_Model"

if [ -e $1 ]; then
  echo "Please specifiy the data preparation stage."
  exit 0
else
  stage=$1
fi

# Generate the concatenations
if [ $stage -le 0 ]; then

  # first, check if the librispeech directory exists
  if [[ ! -d "data/LibriSpeech" ]]; then
    # if not, try to extract the alignments and libri-folders
    cd data
    for subset in $libri_folders
    do
      tar -xf $subset.tar.gz || { echo "The specified librispeech subset is no available, please download it first at: https://www.openslr.org/12"; exit 1; }
    done

    unzip LibriSpeech-Alignments.zip || { echo "The librispeech alignments are not available, please download them first at: https://zenodo.org/record/2619474"; exit 1; }
    cd ../
  fi


  echo "${green}Generating concatenated utterances...${reset}"
  echo "${green}Source datasets:" "$libri_folders" "${reset}"
  echo src/concatenate_utterances.py --libri_root data/LibriSpeech --concat_dir $concat_dir\
    --count $utt_count $libri_folders
  python src/concatenate_utterances.py --libri_root data/LibriSpeech-Alignments/LibriSpeech --concat_dir "$concat_dir"\
    --count $utt_count || { echo "Utterance concatenation failed. Exiting.."; exit 1; }
  echo "${green}Concatenated utterances were saved to" $concat_dir "${reset}"

fi

# if AUGMENT is specified, perform reverberation and augmentation in the kaldi folder..
if [ "$AUGMENT" = true ]; then
  if [ $stage -le 1 ]; then

    # move to the kaldi directory
    echo "${green}Moving to kaldi directory...${reset}"
    cd $kaldi_root

    # make sure there are no duplicate entries in the wav.scp utt2spk and text files
    #sort $concat_dir/utt2spk | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/utt2spk
    #sort $concat_dir/wav.scp | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/wav.scp
    #sort $concat_dir/text | uniq -u >$concat_dir/tmp && mv $concat_dir/tmp $concat_dir/text

    utils/fix_data_dir.sh data/$NAME

    # create the spk2utt for the clean data
    utils/utt2spk_to_spk2utt.pl data/$NAME/utt2spk > data/$NAME/spk2utt

  fi

  if [ $stage -le 2 ]; then
    cd $kaldi_root

    # reverberate the data, augment the data
    ./reverberate_augment.sh 0
    cd $repo_root
  fi
fi


# Feature extraction...
if [ $stage -le 3 ]; then
  if [ "$AUGMENT" = true ]; then

    # copy the embedding files..
    cp -r data/embeddings $kaldi_root/data/

    cd $kaldi_root

    # append the path to kaldi binaries to PATH, needed for using wav-reverberate
    export PATH="$KALDI/src/featbin:$PATH"
    cd data/augmented
    feature_dir=$kaldi_root/data/$feature_dir_name
  else
    cd data
    feature_dir=$feature_dir_name
  fi

  mkdir -p $feature_dir
  echo "${green}Splitting the wav.scp file to" $nj_features "parts...${reset}"

  # split the wav.scp into multiple files to allow for multiprocessing
  cd test
  echo "${green}Splitting the wav.scp file to" $nj_features "parts...${reset}"

  # Count the total number of lines in wav.scp
  total_lines=$(wc -l < wav.scp)

  # Calculate the number of lines per split file
  lines_per_file=$(( (total_lines + nj_features - 1) / nj_features ))

  # Split the wav.scp into multiple files to allow for multiprocessing
  split -l $lines_per_file -a 2 wav.scp split_

  # Rename files to have .scp extension
  count=0
  for f in split_*; do
      mv "$f" "split_$(printf "%02d" $count).scp"
      count=$((count + 1))
  done


  cd ../..
  echo "${green}Running feature extraction..${reset}"

  # extract features, dest: $kaldi_root/data/features
  if [ "$AUGMENT" = true ]; then
    python3 src/extract_features.py --data_root data/augmented --dest_path data/$feature_dir_name \
      --embed_path data/embeddings --use_kaldi --kaldi_root $kaldi_root || { echo "Feature extraction failed. Exiting.."; exit 1; }
  else
    python3 src/extract_features.py --data_root data/$NAME --dest_path data/$feature_dir_name \
      --embed_path data/embeddings || { echo "Feature extraction failed. Exiting.."; exit 1; }
  fi

  # combine back the feature scps
  pwd
  cd data/features_test
  cat fbanks_*.scp > fbanks.scp
  cat scores_*.scp > scores.scp
  cat labels_*.scp > labels.scp
  cat targets_*.scp > targets.scp

  # remove the old *.scp files..
  for name in fbanks_ targets_ scores_ labels_
  do
    rm $name*.scp
  done

  # move from kaldi_root to data/....
  if [ "$AUGMENT" = true ]; then
    echo "${green}Moving the extracted features to the repo root -> data/ directory.${reset}"
    if [[ -d $repo_root/data/$feature_dir_name ]]; then
      echo "${green}The feature directory already exists.. not moving anything.${reset}"
      echo "${green}Features remain in $kaldi_root/data/$feature_dir_name.${reset}"
    else
      cd ..
      mv $feature_dir_name $repo_root/data/
      echo "${green}Features saved to $repo_root/data/$feature_dir_name.${reset}"
    fi
  else
      echo "${green}Features saved to $repo_root/data/$feature_dir_name.${reset}"
  fi

  cd $repo_root
  echo "${green}Feature extraction done.${reset}"
fi
