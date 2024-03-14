# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

import copy
import datetime
import math
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import neptune
import numpy as np
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

import debatellm.eval.load_datasets as load_datasets
from debatellm.utils.eval import batch_generator
from debatellm.utils.s3_io import FileHandler

# https://github.com/pytorch/pytorch/issues/101152#issuecomment-1572203723
# import tensorflow as tf  # isort: skip

# For s3 tb logging
# https://github.com/tensorflow/tensorboard/issues/5480#issuecomment-1016612630
# import tensorflow_io  # noqa: F401 # isort: skip


def eval_on_batch(
    logger: Any,
    questions: list,
    system_fn: Callable,
    format_question_fn: Callable,
    format_solution_fn: Callable,
) -> list:
    system = system_fn()
    questions_and_answers = []
    for _, question in enumerate(questions):
        if "answer" not in question:
            formatted_question = format_question_fn(question)

            count = 0
            while True:
                count += 1

                if count == 10:
                    print("Error: Failed to evaluate 10 times. Exiting.")
                    logger["error"].append("Failed to evaluate 10 times. Exiting.")
                    break

                try:
                    answer, info = system.answer(
                        question=formatted_question,
                    )
                    break
                except Exception as e:
                    # TODO: This only works for single processes.
                    # Get this to work for multiple processes.
                    tb = traceback.format_exc()
                    print("An error occurred:", e)
                    print("Traceback:", tb)
                    logger["error"].append(f"Error {count}: {e}. Traceback: {tb}")
                    time.sleep(10)

            # formatted question
            question["formatted_question"] = formatted_question
            # full response
            question["answer"] = info["response"]
            # only the letter e.g. A, B, C, D
            question["answer_letter"] = answer
            question["correct"] = format_solution_fn(
                question["answer_letter"], question["solution"]
            )
            info = system.metrics(info, format_solution_fn, question["solution"])

            # Three levels of metrics: agent/question, debate, and round.
            if "agent_metrics" in info.keys():
                question["agent_metrics"] = info["agent_metrics"]
            if "debate_metrics" in info.keys():
                question["debate_metrics"] = info["debate_metrics"]
            if "round_metrics" in info.keys():
                question["round_metrics"] = info["round_metrics"]
            questions_and_answers.append(question)
    return questions_and_answers


def evaluate(
    system_fn: Callable,
    system_name: str,
    dataset: str,
    questions: list,
    format_question_fn: Callable,
    format_solution_fn: Callable,
    max_eval_count: Optional[int],
    num_eval_workers: int,
    batch_size: int,
    file_handler: FileHandler,
    logger: Any = None,
    seed: int = 42,
) -> None:
    # Shuffle and select the questions.
    # TODO: Turn this into a dataset object
    # and add flexibility to allow for
    # e.g. 10 questions of each step.
    np.random.seed(seed)
    shuffle_questions = copy.deepcopy(questions)
    np.random.shuffle(shuffle_questions)
    if max_eval_count:
        shuffle_questions = shuffle_questions[:max_eval_count]

    outputs = {}
    eval_categories = list({q["category"] for q in shuffle_questions})
    cat_correct = {cat: 0.0 for cat in eval_categories}
    cat_counts = {cat: 0 for cat in eval_categories}

    running_batch_count = 0
    running_agent_count: Dict[Any, Any] = defaultdict(lambda: defaultdict(int))
    cumulative_agent_stats: Dict[Any, Any] = {}
    cumulative_debate_stats: Dict[Any, Any] = {}
    cumulative_round_stats: Dict[Any, Any] = defaultdict(lambda: defaultdict(float))

    # Take a batch of questions of: batch_size * num_eval_workers
    batch_dataset = batch_generator(shuffle_questions, batch_size, num_eval_workers)
    pbar = tqdm(
        batch_dataset,
        total=int(np.ceil(len(shuffle_questions) / (batch_size * num_eval_workers))),
    )
    pbar.set_description_str(f"\nEvaluating {dataset.upper()} {system_name}")

    for _, batch_of_questions in enumerate(pbar):
        (
            cumulative_cost,
            cumulative_answer_time,
            cumulative_prompt_tokens,
            cumulative_response_tokens,
        ) = (0, 0, 0, 0)
        running_batch_count += len(batch_of_questions)

        #######################
        # Run the system.     #
        #######################

        # Split the batch of questions into num_eval_workers
        split_shuffle_questions = np.array_split(batch_of_questions, num_eval_workers)

        count = 0
        # TODO: Is this correct? Does this not mess up cumulative stats?
        while True:
            count += 1

            if count == 10:
                print("Error: Failed to evaluate 10 times. Exiting.")
                logger["error"].append("Failed to evaluate 10 times. Exiting.")
                break

            try:
                # Run the eval on each batch in parallel
                results_list = Parallel(n_jobs=num_eval_workers)(
                    delayed(eval_on_batch)(
                        logger,
                        questions=shuffle_questions_batch,
                        system_fn=system_fn,
                        format_question_fn=format_question_fn,
                        format_solution_fn=format_solution_fn,
                    )
                    for shuffle_questions_batch in split_shuffle_questions
                )

                # flatten results list
                results_list = [item for sublist in results_list for item in sublist]
                dummy_question = results_list[0]

                if logger:
                    print("\nLogging the following statistics to neptune:")
                    print("Logging: eval/score/total_count")
                    print("Logging: eval/score/total_acc")

                # What metrics to log
                log_agent_metrics: bool = "agent_metrics" in dummy_question.keys()
                log_debate_metrics: bool = "debate_metrics" in dummy_question.keys()
                log_round_metrics: bool = "round_metrics" in dummy_question.keys()

                if log_agent_metrics and logger:
                    if not bool(cumulative_agent_stats):
                        print("\nLogging Agent Metrics")
                        for agent, metric_names in dummy_question[
                            "agent_metrics"
                        ].items():
                            cumulative_agent_stats[agent] = {
                                metric_name: 0 for metric_name in metric_names.keys()
                            }

                            for metric_name in metric_names.keys():
                                print(f"Logging: eval/{agent}/{metric_name}")

                            # Don't need to log or maintain consistent metrics
                            for metric_name in ["agent_name", "agent_engine"]:
                                logger[f"eval/{agent}/{metric_name}"] = dummy_question[
                                    "agent_metrics"
                                ][agent][metric_name]
                                cumulative_agent_stats[agent].pop(metric_name)

                    # Log the agent level metrics
                    for agent, metric_names in cumulative_agent_stats.items():
                        for metric_name in metric_names.keys():

                            # Question level metrics: calculate the running average
                            metric_arr = [
                                x["agent_metrics"][agent][metric_name]
                                for x in results_list
                            ]
                            cumulative_agent_stats[agent][metric_name] += np.nansum(
                                metric_arr
                            )

                            # Add total number of answers that are not NaN
                            running_agent_count[agent][metric_name] += np.count_nonzero(
                                ~np.isnan(metric_arr)
                            )

                            avg_value = (
                                cumulative_agent_stats[agent][metric_name]
                                / running_agent_count[agent][metric_name]
                            )

                            metric_loc = f"eval/{agent}/{metric_name}"
                            if not math.isnan(avg_value):
                                logger[metric_loc] = avg_value
                            else:
                                print(
                                    f"Warning: Skipping logging of {metric_loc}"
                                    " as its output is NaN."
                                )

                        # This maintains the cumulative cost: sum of all agents' question costs
                        cumulative_cost += cumulative_agent_stats[agent][
                            "cost_per_question"
                        ]
                        cumulative_answer_time += cumulative_agent_stats[agent][
                            "time_per_question"
                        ]
                        cumulative_prompt_tokens += cumulative_agent_stats[agent][
                            "total_prompt_tokens"
                        ]
                        cumulative_response_tokens += cumulative_agent_stats[agent][
                            "total_response_tokens"
                        ]

                if log_round_metrics and logger:
                    if not bool(cumulative_round_stats):
                        # Initialize the round level metrics
                        print("\nLogging Round Metrics")
                        for agent, metric_names in dummy_question[
                            "round_metrics"
                        ].items():
                            for metric_name in metric_names.keys():
                                print(f"Logging: eval/{agent}/round_X/{metric_name}")

                    # Log round level metrics
                    for agent, metrics in cumulative_round_stats.items():
                        for metric_name in metrics.keys():
                            loc = f"eval/{agent}/{metric_name}"
                            metric_rounds = list(
                                dummy_question["round_metrics"][agent][
                                    metric_name
                                ].keys()
                            )
                            for debate_round in metric_rounds:
                                cumulative_round_stats[agent][metric_name][
                                    debate_round
                                ] += np.sum(
                                    [
                                        x["round_metrics"][agent][metric_name][
                                            debate_round
                                        ]
                                        for x in results_list
                                        if debate_round
                                        in x["round_metrics"][agent][metric_name]
                                    ]
                                )
                                logger[f"{loc}/{debate_round}"] = (
                                    cumulative_round_stats[agent][metric_name][
                                        debate_round
                                    ]
                                    / running_batch_count
                                )

                if log_debate_metrics and logger:
                    if not bool(cumulative_debate_stats):
                        # Initialize the debate level metrics
                        print("\nLogging Debate Metrics")
                        metric_names = list(dummy_question["debate_metrics"].keys())
                        cumulative_debate_stats = {
                            metric_name: 0 for metric_name in metric_names
                        }

                        for metric in metric_names:
                            print(f"Logging: eval/debate/{metric}")

                    # Log the debate level metrics
                    for metric_name in cumulative_debate_stats.keys():
                        cumulative_debate_stats[metric_name] += np.sum(
                            [x["debate_metrics"][metric_name] for x in results_list]
                        )
                        avg_value = (
                            cumulative_debate_stats[metric_name] / running_batch_count
                        )
                        logger[f"eval/debate/{metric_name}"] = avg_value

                # Calculate scores over the batch
                for cat in list(eval_categories):
                    # results_list
                    filtered_q_and_a = [x for x in results_list if x["category"] == cat]
                    filtered_q_and_a_correct = [x["correct"] for x in filtered_q_and_a]

                    cat_correct[cat] += np.sum(filtered_q_and_a_correct)
                    cat_counts[cat] += len(filtered_q_and_a_correct)

                    acc = (
                        cat_correct[cat] / cat_counts[cat] if cat_counts[cat] > 0 else 0
                    )
                    outputs[f"acc_{cat}"] = acc
                    print(
                        f"Score - {cat}, {cat_counts[cat]} evals, % correct (out of 1): {acc:.2f} "  # noqa: E501
                    )

                    # Log category scores to neptune.
                    if logger:
                        logger[f"eval/score/{cat}_count"] = cat_counts[cat]
                        logger[f"eval/score/{cat}_acc"] = acc

                    # Save the q&a's and incorrect q&a's to a file location.
                    outname = f"{system_name.split('/')[-1]}_{cat}.json"
                    file_handler.dump_batch_of_question_and_answers(
                        outname, filtered_q_and_a
                    )

                incorrect_q_and_a = [x for x in results_list if not x["correct"]]
                outname = f"{system_name.split('/')[-1]}_incorrect.json"
                file_handler.dump_batch_of_question_and_answers(
                    outname, incorrect_q_and_a
                )

                outname = f"{system_name.split('/')[-1]}.json"
                file_handler.dump_batch_of_question_and_answers(outname, results_list)

                #######################
                # Calculate results.  #
                #######################

                # Write final results to file.
                final_results = {
                    i: cat_correct[i] / cat_counts[i] if cat_counts[i] > 0 else None
                    for i in eval_categories
                }

                totals = int(sum([cat_correct[i] for i in eval_categories]))
                counts = sum([cat_counts[i] for i in eval_categories])

                acc = totals / counts
                final_results["acc_total"] = acc
                print(
                    f"\nRunning Total, {counts} evals, % correct (out of 1): {acc:.2f} "
                )  # noqa: E501
                pbar.set_postfix(final_results)

                # Log total scores to neptune.
                if logger:
                    logger["eval/score/total_count"] = counts
                    logger["eval/score/total_acc"] = acc
                    logger["eval/percent_complete"] = (
                        100 * running_batch_count / len(shuffle_questions)
                    )
                    logger["eval/total_cost"] = cumulative_cost
                    logger["eval/avg_sec_per_question"] = (
                        cumulative_answer_time / running_batch_count
                    )

                    avg_prompt = cumulative_prompt_tokens / running_batch_count
                    logger["eval/avg_prompt_tokens_per_question"] = avg_prompt
                    avg_response = cumulative_response_tokens / running_batch_count
                    logger["eval/avg_response_tokens_per_question"] = avg_response
                    logger["eval/avg_tokens_per_question"] = avg_prompt + avg_response
                break
            except Exception as e:
                # TODO: This only works for single processes.
                # Get this to work for multiple processes.
                tb = traceback.format_exc()
                print("An error occurred:", e)
                print("Traceback:", tb)
                logger["error"].append(f"Error {count}: {e}. Traceback: {tb}")
                time.sleep(10)

    if logger:
        logger.stop()

    outpath = os.path.join(file_handler.results_path, "final_results.json")
    file_handler.save_json(outpath, final_results)


def eval_system(
    system_fn: Callable,
    system_name: str,
    dataset: str,
    path_to_exams: str,
    dataset_settings: Dict,
    eval_batch_size: int,
    max_eval_count: Optional[int] = None,
    num_eval_workers: int = 4,
    logger: Optional[neptune.Run] = None,
    seed: int = 42,
    save_question_answer_mode: bool = False,
) -> None:
    # Create file handler that set up the access to the s3 bucket if needed.
    file_handler = FileHandler(
        s3_endpoint=os.environ.get("S3_ENDPOINT"),
        bucket="input",
        path_to_exams=path_to_exams,
        save_question_mode=save_question_answer_mode,
    )

    # Create the results folder.
    now = datetime.datetime.now()
    exp_id = now.strftime("%Y-%m-%d_%H:%M:%S")
    folder_name = f"{exp_id}_{system_name.replace(r'/', '-')}"
    file_handler.make_results_folder(folder_name)

    # Functionality to allow this function to vary per dataset
    format_solution_fn = lambda answer, solution: answer.upper() == solution.upper()

    # Filtered set of Q&As that were previously answered incorrectly.
    if "incorrect" in dataset_settings["exam_type"]:
        questions, format_question_fn = load_datasets.load_previous_incorrect_questions(
            file_handler, dataset_name=dataset_settings["exam_type"]
        )

    elif dataset == "usmle":
        questions, format_question_fn = load_datasets.usmle_questions(
            file_handler, dataset_settings
        )

    elif dataset == "medmcqa":
        questions, format_question_fn = load_datasets.medmcqa_questions(
            file_handler, dataset_settings
        )

    elif dataset == "mmlu":
        questions, format_question_fn = load_datasets.mmlu_questions(
            file_handler,
            dataset_settings,
        )

    elif dataset == "pubmedqa":
        questions, format_question_fn = load_datasets.pubmedqa_questions(
            file_handler, dataset_settings
        )

    elif dataset == "medqa":
        questions, format_question_fn = load_datasets.medqa_questions(
            file_handler, dataset_settings
        )
    elif dataset == "ciar":
        questions, format_question_fn = load_datasets.ciar_questions(
            file_handler, dataset_settings
        )
    elif dataset == "cosmosqa":
        questions, format_question_fn = load_datasets.cosmosqa_questions(
            file_handler,
            dataset_settings,
        )
    elif dataset == "gpqa":
        questions, format_question_fn = load_datasets.gpqa_questions(
            file_handler,
            dataset_settings,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    evaluate(
        system_fn,
        system_name,
        dataset,
        questions,
        format_question_fn,
        format_solution_fn,
        max_eval_count,
        num_eval_workers,
        eval_batch_size,
        file_handler,
        logger,
        seed,
    )
