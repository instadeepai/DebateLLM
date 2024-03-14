# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/mmlu/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/mmlu/"
fi

if [ -e "${FOLDER}data/dev" ]
then
    echo "data already exists in $FOLDER"
else
    # check $FOLDER exists
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    echo "downloading mmlu dataset to $FOLDER"
    # zip
    wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P $FOLDER

    # unzip
    tar -xvf ${FOLDER}data.tar -C $FOLDER

    # move files up one level for consistency
    mv ${FOLDER}data/* ${FOLDER}/.

    # remove redundant subdirectory
    rm -rf ${FOLDER}/data

    # remove zip
    rm ${FOLDER}data.tar
fi
