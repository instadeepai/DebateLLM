# `load_dataset` doesn't [work](https://github.com/kbressem/medAlpaca/issues/29) so we manually use wget.
# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/pubmedqa/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/pubmedqa/"
fi

if [ -e "${FOLDER}test.json" ]
then
    echo "data already exists in $FOLDER"
else
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    echo "downloading usmle dataset to $FOLDER"

    # download the labeled part of the dataset and the dev/test split
    wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json -P $FOLDER
    wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json -P $FOLDER
fi

# Execute Python script
python3 << EOF

import json

# Load data from the json files
with open("${FOLDER}ori_pqal.json", 'r') as f:
    data1 = json.load(f)

with open("${FOLDER}test_ground_truth.json", 'r') as f:
    data2 = json.load(f)

# Initialize dictionaries for train and test
train = {}
test = {}

# Populate the train and test sets
for key, value in data1.items():
    if key in data2:
        test[key] = value
    else:
        train[key] = value

# Write the train and test sets to json files
with open('${FOLDER}dev.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('${FOLDER}test.json', 'w') as f:
    json.dump(test, f, indent=4)

EOF
