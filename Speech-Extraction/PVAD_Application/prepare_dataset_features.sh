#!/bin/bash

repo_root="/dataHDD1/timkleinlein/among-us-analysis/Speech-Extraction/PVAD_Application"
nj_features=4 # number of CPU workers used for feature extraction
feature_dir_name=features_application
stage=$1 # Get the stage from the command line argument

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Generate the audio files in correct format and the alignments
if [ $stage -le 0 ]; then
  python src/create_alignments_application.py || { echo "Alignment creation failed. Exiting.."; exit 1; }
  echo "${green}Audios and alignments were successfully created.${reset}"
fi

# Feature extraction...
if [ $stage -le 1 ]; then

  mkdir -p data/$feature_dir_name
  cd data

  # split the wav.scp into multiple files to allow for multiprocessing
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

  cd ..
  echo "${green}Running feature extraction..${reset}"

  # extract features
  python src/extract_features_application.py --data_root data --dest_path data/$feature_dir_name --embed_path data/embeddings || { echo "Feature extraction failed. Exiting.."; exit 1; }

  # combine back the feature scps
  cd data/$feature_dir_name
  cat fbanks_*.scp > fbanks.scp
  cat scores_*.scp > scores.scp
  cat targets_*.scp > targets.scp

  # remove the old *.scp files..
  for name in fbanks_ scores_ targets_
  do
    rm $name*.scp
  done

  echo "${green}Feature extraction done.${reset}"
fi
