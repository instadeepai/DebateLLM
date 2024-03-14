# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/medqa/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/medqa/"
fi

if [ -e "${FOLDER}/questions" ]
then
    echo "data already exists in $FOLDER"
else
    echo "downloading medqa dataset to $FOLDER"
    # zip
    gdown --id 1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw
    # check $FOLDER exists
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    # unzip
    unzip data_clean.zip -d ${FOLDER}

    # move files up one level for consistency
    mv ${FOLDER}data_clean/questions/US/* ${FOLDER}/.

    # remove zip
    rm -r ${FOLDER}data_clean/

    # remove zip
    rm data_clean.zip
fi
