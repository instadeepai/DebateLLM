# `load_dataset` doesn't [work](https://github.com/kbressem/medAlpaca/issues/29) so we manually use wget.
# AICHOR_OUTPUT_PATH includes a trailing /
if [[ -z "${AICHOR_INPUT_PATH}" ]]; then
    FOLDER=data/datasets/usmle/
else
    FOLDER="${AICHOR_INPUT_PATH}data/datasets/usmle/"
fi

if [ -e "${FOLDER}step1.json" ]
then
    echo "data already exists in $FOLDER"
else
    if [ ! -e "${FOLDER}" ]
    then
        echo "Creating folder ${FOLDER}"
        mkdir -p "${FOLDER}"
    fi
    echo "downloading usmle dataset to $FOLDER"

    # question with images (to be filtered out)
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/question_with_images.json -P $FOLDER

    # part 1
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step1.json -P $FOLDER
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step1_solutions.json -P $FOLDER

    # part 2
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step2.json -P $FOLDER
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step2_solutions.json -P $FOLDER

    # part 3
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step3.json -P $FOLDER
    wget https://huggingface.co/datasets/medalpaca/medical_meadow_usmle_self_assessment/raw/main/step3_solutions.json -P $FOLDER
fi
