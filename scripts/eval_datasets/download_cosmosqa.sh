# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/cosmosqa/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/cosmosqa/"
fi

if [ -e "${FOLDER}cosmos-qa-dev" ]
then
    echo "Cosmos QA data already exists in $FOLDER"
else
    # check $FOLDER exists
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    echo "downloading Cosmos QA dataset to $FOLDER"

    # Use Kaggle API to download the dataset
    kaggle datasets download thedevastator/cosmos-qa-a-large-scale-commonsense-based-readin -p $FOLDER --unzip

    # Note: The Kaggle API automatically unzips the downloaded file
fi
