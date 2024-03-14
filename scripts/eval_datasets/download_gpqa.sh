# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/gpqa/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/gpqa/"
fi

if [ -e "${FOLDER}/gpqa_main.csv" ]
then
    echo "GPQA data already exists in $FOLDER"
else
    echo "Downloading GPQA dataset to $FOLDER"
    # check if $FOLDER exists
    if [ ! -e "${FOLDER}" ]; then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    # Download the dataset using wget
    wget -O "${FOLDER}dataset.zip" https://github.com/idavidrein/gpqa/blob/main/dataset.zip?raw=true

    # Unzip with password
unzip -P deserted-untie-orchid -o "${FOLDER}dataset.zip" -d "${FOLDER}"

    # Move the CSV files from the 'dataset' subdirectory up one level to $FOLDER
    mv "${FOLDER}dataset/"*.csv "${FOLDER}"

    # Check if other files need to be kept, if not remove the entire 'dataset' directory
    # If you need to keep files like 'license.txt', do not remove the directory.
    # If you decide to remove the directory, use 'rm -r' to remove non-empty directory
    # rm -r "${FOLDER}dataset/"

    # remove zip file
    rm "${FOLDER}dataset.zip"
fi
