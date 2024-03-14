# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/medmcqa/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/medmcqa/"
fi

if [ -e "${FOLDER}train.json" ]
then
    echo "data already exists in $FOLDER"
else
    echo "downloading medmcqa dataset to $FOLDER"
    # zip
    gdown --id 15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky
    # check $FOLDER exists
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    # unzip
    unzip data.zip -d ${FOLDER}
    # remove zip
    rm data.zip
fi
