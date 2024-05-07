#!/bin/bash

# Check for AICHOR_INPUT_PATH and set FOLDER accordingly
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/chess/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/chess/"
fi

# Check if the dataset already exists
if [ -e "${FOLDER}task.json" ]; then
    echo "CHESS dataset already exists in $FOLDER"
else
    echo "Downloading CHESS dataset to $FOLDER"
    
    # Ensure the target directory exists
    if [ ! -d "${FOLDER}" ]; then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi

    # Download the specific part of the dataset
    wget https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/chess_state_tracking/real_short/task.json -P $FOLDER
    echo "Dataset downloaded to $FOLDER"
fi
