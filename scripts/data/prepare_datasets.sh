#!/bin/bash

# Sprawdzanie, czy został podany argument
#if [ $# -eq 0 ]; then
#    echo "Brak argumentu. Użycie: ./prepare_datasets.sh <argument>"
#    exit 1
#fi

# Przechwycenie argumentu
#ARG=$1

# Uruchamianie pliku Pythonowego z argumentem
python3 "download_data.py"
python3 "filter_annotations.py"
python3 "prepare_training_subsets.py"
python3 "prepare_yolo_dataset.py" #"$ARG"
