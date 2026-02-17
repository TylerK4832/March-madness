#!/usr/bin/env bash
# Download real NCAA March Madness data from public GitHub mirrors.
# These files match the Kaggle March Machine Learning Mania schema.
# Data covers 2003-2018/2019 (detailed box scores).
#
# For the latest data, download directly from Kaggle:
#   kaggle competitions download -c march-machine-learning-mania-2025

set -e
DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DIR"

echo "Downloading NCAA March Madness data to $DIR ..."

# Core files from kjaisingh/Forecasting-March-Madness (2003-2019)
BASE="https://raw.githubusercontent.com/kjaisingh/Forecasting-March-Madness/master/data"
for f in NCAATourneyDetailedResults.csv RegularSeasonDetailedResults.csv NCAATourneySeeds.csv Teams.csv; do
  target="M${f}"
  echo "  $target ..."
  curl -sL "$BASE/$f" -o "$DIR/$target"
done

# Massey Ordinals from fallonfarmer/march-madness-2018 (2003-2018)
echo "  MMasseyOrdinals.csv (large file, may take a moment) ..."
curl -sL "https://raw.githubusercontent.com/fallonfarmer/march-madness-2018/master/data/MasseyOrdinals.csv" \
  -o "$DIR/MMasseyOrdinals.csv"

echo ""
echo "Done! Files in $DIR:"
ls -lh "$DIR"/*.csv
