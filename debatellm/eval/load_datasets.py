# Copyright Anon 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from debatellm.utils.eval import (
    answer_to_letter_pubmedqa,
    mmlu_categories,
    mmlu_medical_subcategories,
    mmlu_subcategories,
)
from debatellm.utils.s3_io import FileHandler

MEDMCQA_SOLUTIONS = {1: "A", 2: "B", 3: "C", 4: "D"}


def load_previous_incorrect_questions(
    file_handler: FileHandler,
    dataset_name: str,
) -> Tuple[list, Callable]:
    """Load MMLU dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_name : the name of the datasets to load the Q&As from.

    Returns:
        questions : list of questions.
        format_question : dummy function to format question.
    """

    questions = []
    for q in file_handler.read_json(dataset_name + ".json"):
        q["previous_answer"] = q.pop("answer")
        q["previous_answer_letter"] = q.pop("answer_letter")
        q["previous_correct"] = q.pop("correct")
        questions.append(q)
    return questions, lambda x: x["formatted_question"]


# flake8: noqa: C901
def mmlu_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load MMLU dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the
        directory name of exam to evaluate on.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]

    # Use only medical subcategories if the dataset settings specify to do so.
    if dataset_settings["medical_tasks_only"]:
        task_filenames = [
            task + f"_{exam_type}.csv" for task in mmlu_medical_subcategories
        ]
    else:
        task_filenames = file_handler.listdir(exam_type)
    task_names = [
        t.replace(".csv", "").replace(f"_{exam_type}", "") for t in task_filenames
    ]

    task_filenames = [task_filenames[-1]]
    task_names = [task_names[-1]]

    for task_file, task_name in zip(task_filenames, task_names):
        subcategory = mmlu_subcategories[task_name][0]

        col_names = ["question", "opa", "opb", "opc", "opd", "solution"]
        df = file_handler.read_csv(
            path=os.path.join(exam_type, task_file), col_names=col_names
        )
        list_of_dicts = df.to_dict("records")

        for i, q in enumerate(list_of_dicts):
            q["no"] = int(i)
            q["options"] = {"A": q["opa"], "B": q["opb"], "C": q["opc"], "D": q["opd"]}
            q["task"] = task_name
            q["subcategory"] = subcategory
            q["category"] = [k for k, v in mmlu_categories.items() if subcategory in v][
                0
            ]
            questions.append(q)

    def format_question(d: dict) -> str:

        # Format the subject str to remove "dev", "test", "train" and underscores
        question = d["category"] + ": " + d["subcategory"] + ": " + d["question"]
        options = d["options"]
        for k, v in options.items():
            question += f"\n{k}: {v}"
        return question

    return questions, format_question


# flake8: noqa: C901
def medmcqa_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load MedMCQA dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the file
        name of the exam to evaluate on.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]

    data_path = f"{exam_type}.json"
    question_file = file_handler.read_json_lines(path=data_path)
    for i, q in enumerate(question_file):
        question = {
            "no": int(i),
            "question": q["question"],
            "subcategory": q["topic_name"],
            "category": q["subject_name"],
            "options": {"A": q["opa"], "B": q["opb"], "C": q["opc"], "D": q["opd"]},
            "solution": MEDMCQA_SOLUTIONS[q["cop"]],
        }

        questions.append(question)

    def format_question(d: dict) -> str:
        question = ""
        options = d["options"]
        for key in ["category", "subcategory", "question"]:
            if d[key]:
                question += f"\n{key}: {d[key]}"
        for k, v in options.items():
            question += f"\n{k}: {v}"
        return question

    return questions, format_question


# flake8: noqa: C901
def usmle_questions(
    file_handler: FileHandler, dataset_settings: Dict
) -> Tuple[list, Callable]:
    """Load usemle questions.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the include_image_based_questions
        setting. This determines whether image based questions should be included
        or not.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    #######################
    # Load the dataset.   #
    #######################
    questions = []

    # Load questions with images.
    questions_with_images = file_handler.read_json(path="question_with_images.json")

    for step_idx in [1, 2, 3]:
        step = file_handler.read_json(path=f"step{step_idx}.json")
        solutions = file_handler.read_json(path=f"step{step_idx}_solutions.json")

        # Optionally skip image-based questions.
        include_ibq = dataset_settings["include_image_based_questions"]
        skip_questions = [] if include_ibq else questions_with_images[f"step{step_idx}"]

        # Load the questions.
        for i, question in enumerate(step):
            if i not in skip_questions:
                question["category"] = "Step%s" % int(step_idx)
                question["no"] = int(question["no"])
                question["solution"] = solutions[str(question["no"])].upper()

                if len(question["options"]) > 0:
                    questions.append(question)

    def format_question(d: dict) -> str:
        question = d["question"]
        options = d["options"]

        for k, v in options.items():
            question += f"\n{k}: {v}"

        return question

    return questions, format_question


# flake8: noqa: C901
def pubmedqa_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load PubMedQA dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the
        filename of the exam to evaluate on.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]
    data_path = f"{exam_type}.json"
    dict_loaded = file_handler.read_json(path=data_path)

    for i, q in enumerate(dict_loaded.values()):
        question = {
            "question": q["QUESTION"],
            "no": int(i),
            "contexts": q["CONTEXTS"],
            "labels": q["LABELS"],
            "category": "All",
            "solution": answer_to_letter_pubmedqa(q["final_decision"]),
        }
        questions.append(question)

    def format_question(d: dict) -> str:
        question = d["question"]
        contexts = d["contexts"]
        labels = [label.capitalize() for label in d["labels"]]
        full_question = "Context: "
        for label, context in zip(labels, contexts):
            full_question += f"\n{label}: {context}"
        full_question += f"\nQuestion: {question}"
        full_question += "\nOptions: \nA: Yes \nB: No \nC: Maybe"
        return full_question

    return questions, format_question


# flake8: noqa: C901
def medqa_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load MedQA questions.

    Args:
        file_handler : file handler for s3 or local files.
        dataset_settings : dictionary containing the filename of the
        exam to evaluate on.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    #######################
    # Load the dataset.   #
    #######################
    questions = []
    exam_type = dataset_settings["exam_type"]
    data_path = f"{exam_type}.jsonl"
    question_file = file_handler.read_json_lines(path=data_path)

    for i, q in enumerate(question_file):
        question = {
            "question": q["question"],
            "no": int(i),
            "category": q["meta_info"].capitalize(),
            "options": q["options"],
            "solution": q["answer_idx"],
        }
        questions.append(question)

    def format_question(d: dict) -> str:
        question = d["question"]
        options = d["options"]
        for k, v in options.items():
            question += f"\n{k}: {v}"
        return question

    return questions, format_question


def ciar_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load counterIntuitiveQA dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the
        filename of the exam to evaluate on.

    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]
    data_path = f"{exam_type}.json"
    dict_loaded = file_handler.read_json(path=data_path)

    for i, q in enumerate(dict_loaded):
        indices = [0, 1]
        random.shuffle(indices)
        options = [
            q["answer"][-1],
            q["incorrect answer"][-1],
        ]  # -1 because it's generally best looking answer
        options = [options[i] for i in indices]
        solution = ["A", "B"][indices.index(0)]

        question = {
            "question": q["question"],
            "no": int(i),
            "explanation": q["explanation"],
            "incorrect_explanation": q["incorrect explanation"],
            "options": options,
            "category": "All",
            "solution": solution,
        }
        questions.append(question)

    def format_question(d: dict) -> str:
        question = d["question"]
        options = d["options"]
        labels = ["A", "B"]
        for label, option in zip(labels, options):
            question += f"\n{label}: {option}"
        return question

    return questions, format_question


# flake8: noqa: C901
def cosmosqa_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load Cosmos QA dataset and question format function.
    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the
        directory name of exam to evaluate on.
    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]

    # Define the column names you are interested in
    col_names = [
        "context",
        "answer0",
        "answer1",
        "answer2",
        "answer3",
        "label",
    ]

    # Ignore mypy issues
    df = file_handler.read_csv(
        path=f"{exam_type}.csv",
        header="infer",  # type: ignore
        delimiter=None,  # type: ignore
        quotechar='"',
        usecols=col_names,
    )

    # Reset the index to ensure the dictionary keys start from 0
    df.reset_index(drop=True, inplace=True)

    # Convert the DataFrame to a dictionary with integer keys starting from 0
    list_of_dicts = df.to_dict("records")

    for i, q in enumerate(list_of_dicts):
        answers = [
            q["answer0"],
            q["answer1"],
            q["answer2"],
            q["answer3"],
        ]
        correct_index = int(q["label"])
        solution_letter = ["A", "B", "C", "D"][correct_index]

        question = {
            "question": q["context"],
            "no": int(i),
            "category": "All",
            "options": {
                "A": answers[0],
                "B": answers[1],
                "C": answers[2],
                "D": answers[3],
            },
            "solution": solution_letter,
        }
        questions.append(question)

    def format_question(d: dict) -> str:
        # Format the subject str to remove "dev", "test", "train" and underscores
        question = d["category"] + ": " + d["question"]
        options = d["options"]
        for k, v in options.items():
            question += f"\n{k}: {v}"
        return question

    return questions, format_question


# flake8: noqa: C901
def gpqa_questions(
    file_handler: FileHandler,
    dataset_settings: Dict,
) -> Tuple[list, Callable]:
    """Load GPQA dataset and question format function.

    Args:
        file_handler: file handler for s3 or local files.
        dataset_settings : dictionary containing the
        directory name of exam to evaluate on.
    Returns:
        questions : list of questions.
        format_question : function to format question correctly.
    """

    questions = []
    exam_type = dataset_settings["exam_type"]

    # Define the column names you are interested in
    col_names = [
        "Question",
        "Subdomain",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]

    # filename = f"./data/datasets/gpqa/gpqa_{exam_type}.csv"
    # df = pd.read_csv(filename, usecols=col_names)

    # Ignore mypy issues
    df = file_handler.read_csv(
        path=f"gpqa_{exam_type}.csv",
        header="infer",  # type: ignore
        delimiter=None,  # type: ignore
        quotechar='"',
        usecols=col_names,
    )

    # Reset the index to ensure the dictionary keys start from 0
    df.reset_index(drop=True, inplace=True)

    # Convert the DataFrame to a dictionary with integer keys starting from 0
    list_of_dicts = df.to_dict("records")

    for i, q in enumerate(list_of_dicts):
        answers = [
            q["Correct Answer"],
            q["Incorrect Answer 1"],
            q["Incorrect Answer 2"],
            q["Incorrect Answer 3"],
        ]

        # Shuffle the answers
        random.shuffle(answers)

        # Find the index of the correct answer and map it to the multiple-choice letter
        correct_index = answers.index(q["Correct Answer"])
        solution_letter = ["A", "B", "C", "D"][correct_index]

        question = {
            "question": q["Question"],
            "no": int(i),
            "category": q["Subdomain"],
            "options": {
                "A": answers[0],
                "B": answers[1],
                "C": answers[2],
                "D": answers[3],
            },
            "solution": solution_letter,
        }
        questions.append(question)

    def format_question(d: dict) -> str:
        # Format the subject str to remove "dev", "test", "train" and underscores
        question = d["category"] + ": " + d["question"]
        options = d["options"]
        for k, v in options.items():
            question += f"\n{k}: {v}"
        return question

    return questions, format_question
