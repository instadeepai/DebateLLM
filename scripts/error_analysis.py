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

# Import necessary libraries
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from debatellm.utils.eval import extract_last_floating_capital_letter_as_answer


def extract_answers(df):
    # Iterate through the DataFrame rows to update each 'answer' field with the extracted answers
    for index, row in df.iterrows():
        agent_answers = row["answer"].copy()

        if agent_answers:  # Check if the 'answer' field is not empty or None
            for agent, rounds in agent_answers.items():
                for round_num, response in rounds.items():
                    # Use the function to extract the last floating capital letter as answer
                    extracted_answer = extract_last_floating_capital_letter_as_answer(
                        response
                    )

                    # Create a copy of the existing response dictionary, update it, and then
                    # replace the original
                    updated_round = (
                        rounds[round_num].copy()
                        if isinstance(rounds[round_num], dict)
                        else {}
                    )
                    updated_round["response"] = response
                    updated_round["extracted_answer"] = extracted_answer

                    # Update the 'answer' field dictionary with the new round dictionary
                    agent_answers[agent][round_num] = updated_round

        # Update the 'answer' column with the modified dictionaries
        df.at[index, "answer"] = (
            agent_answers
            if agent_answers
            else {"Agent_0": {"Round_0": {"response": None, "extracted_answer": "-1"}}}
        )
    return df


def create_tree_structure(df):
    # Function to generate a tree-like structure to show DataFrame's column and
    # nested field relationships
    def generate_tree_structure(d, indent=0, parent=""):
        output = []
        if isinstance(d, dict):
            for k, v in d.items():
                output.append("  " * indent + f"{parent}{k}")
                output.extend(generate_tree_structure(v, indent + 1, f"{k} -> "))
        elif isinstance(d, list):
            output.append("  " * indent + f"{parent}list")
            if len(d) > 0:
                output.extend(generate_tree_structure(d[0], indent + 1, "list -> "))
        return output

    # Sample row to represent the DataFrame structure
    sample_row = df.iloc[0].to_dict()

    # Generate the tree structure starting from the DataFrame level
    tree_structure = ["DataFrame"]
    for column, value in sample_row.items():
        tree_structure.append(f"|-- {column}")
        tree_structure.extend(generate_tree_structure(value, 2, "|   |-- "))

    # Join the list into a string to display the tree structure
    return "\n".join(tree_structure)


def plot_answers_diversity(df):
    # Initialize dictionaries to hold the counts of unique answers per round for correct
    # and incorrect answers
    unique_answers_per_round_correct = {}
    unique_answers_per_round_incorrect = {}

    # Iterate through the DataFrame to collect data
    for _, row in df.iterrows():
        agent_answers = row["answer"]
        is_correct = row["correct"]

        if agent_answers:  # Check if the 'answer' field is not empty or None
            if "Reasoning_0" in agent_answers["Agent_0"].keys():
                agent_answers = agent_answers["Agent_0"]
                unique_answers_this_round = {"Round_0": set()}
            else:
                unique_answers_this_round = {
                    round_num: set()
                    for round_num in agent_answers[next(iter(agent_answers))]
                }

            for _, rounds in agent_answers.items():

                if "response" in rounds.keys():
                    rounds = {"Round_0": rounds}

                for round_num, round_data in rounds.items():
                    unique_answers_this_round[round_num].add(
                        round_data.get("extracted_answer", "-1")
                    )

            # Count the number of unique answers for this question and accumulate it to the
            # correct or incorrect dictionary
            for round_num, unique_answers in unique_answers_this_round.items():
                target_dict = (
                    unique_answers_per_round_correct
                    if is_correct
                    else unique_answers_per_round_incorrect
                )

                if round_num not in target_dict:
                    target_dict[round_num] = []

                target_dict[round_num].append(len(unique_answers))

    # Calculate the average number of unique answers for each round
    avg_unique_answers_per_round_correct = {
        k: np.mean(v) for k, v in unique_answers_per_round_correct.items()
    }
    avg_unique_answers_per_round_incorrect = {
        k: np.mean(v) for k, v in unique_answers_per_round_incorrect.items()
    }

    # Data for plotting
    round_nums_correct = sorted(avg_unique_answers_per_round_correct.keys())
    avg_unique_correct = [
        avg_unique_answers_per_round_correct[r] for r in round_nums_correct
    ]
    round_nums_incorrect = sorted(avg_unique_answers_per_round_incorrect.keys())
    avg_unique_incorrect = [
        avg_unique_answers_per_round_incorrect[r] for r in round_nums_incorrect
    ]

    # Create the bar plot
    bar_width = 0.35
    indices_correct = np.arange(len(round_nums_correct))
    indices_incorrect = np.arange(len(round_nums_incorrect))

    # Further adjust plot for even larger text
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 18})

    plt.bar(
        indices_correct,
        avg_unique_correct,
        bar_width,
        label="Correct Answers",
        color="green",
    )
    plt.bar(
        indices_incorrect + bar_width,
        avg_unique_incorrect,
        bar_width,
        label="Incorrect Answers",
        color="red",
    )

    plt.xlabel("Round Number", fontsize=20)
    plt.ylabel("Average Number of Unique Answers", fontsize=20)
    plt.title("Average Number of Unique Answers Per Round", fontsize=22)
    plt.xticks(indices_correct + bar_width / 2, round_nums_correct, fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=18)

    plt.tight_layout(pad=0)

    # Save the plot
    plt.savefig("./data/charts/answers_diversity.pdf", bbox_inches="tight")


def sample_debate_outputs_for_question_i(df, question_index):
    print("=== Fetching row for given question index... ===")
    row = df.iloc[question_index]
    agent_answers = row["answer"]
    correct_answer = row["solution"]  # Changed this line to use 'solution'

    options = row["options"]

    # options = {"A": "Intrafascicular ...", "B": "Inflammation .."}
    # Convert options to a string
    options = "\n".join([f"{k}: {v}" for k, v in options.items()])

    print(
        f"\nQuestion: {row['question']}\n\n{options}\n\nCorrect Answer: {correct_answer}"
    )

    print("\n=== Starting to sample debate outputs... ===")

    # Assuming that all agents have the same rounds available
    sample_agent = next(iter(agent_answers.keys()))
    rounds_available = agent_answers[sample_agent].keys()

    for round_number in rounds_available:
        print(f"\n--- {round_number} ---")

        for agent in agent_answers.keys():
            round_data = agent_answers[agent].get(round_number, {})
            response = round_data.get("response", "")
            predicted_answer = round_data.get("extracted_answer", "-1")

            print(
                f"\n{agent}:\n  Response: {response}\n  ",
                f"Extracted: {predicted_answer}\n  Correct: {correct_answer}",
            )

    print("\n=== Finished sampling debate outputs. ===")


def plot_extracted_answer_frequency(df):
    """
    Plots the frequency of extracted answers across all agents and rounds in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the debate data.
    """
    # Initialize a Counter to store the frequencies of extracted answers
    extracted_answer_freq = Counter()

    # Iterate through the DataFrame to collect extracted answers
    for _, row in df.iterrows():
        for _, rounds in row["answer"].items():
            for _, round_data in rounds.items():
                extracted_answer = round_data.get("extracted_answer", "-1")
                extracted_answer_freq[extracted_answer] += 1

    # Prepare data for plotting
    labels = list(extracted_answer_freq.keys())
    frequencies = list(extracted_answer_freq.values())

    # Sort the data in descending order
    sorted_indices = np.argsort(frequencies)[::-1]
    labels = np.array(labels)[sorted_indices]
    frequencies = np.array(frequencies)[sorted_indices]

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=frequencies, palette="viridis")
    plt.title("Frequency of Extracted Answers", fontsize=24)  # Adjusted fontsize
    plt.xlabel("Extracted Answers", fontsize=16)  # Adjusted fontsize
    plt.ylabel("Frequency", fontsize=16)  # Adjusted fontsize
    plt.xticks(rotation=45, fontsize=14)  # Adjusted fontsize
    plt.yticks(fontsize=14)  # Adjusted fontsize

    # Save the plot
    plt.savefig("./data/charts/extracted_frequency.pdf", bbox_inches="tight")


def plot_actual_answer_frequency(df):
    """
    Plots the frequency of actual answers (solutions) in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the debate data.
    """
    # Initialize a Counter to store the frequencies of actual answers
    actual_answer_freq = Counter(df["solution"])

    # Prepare data for plotting
    labels = list(actual_answer_freq.keys())
    frequencies = list(actual_answer_freq.values())

    # Sort the data in descending order
    sorted_indices = np.argsort(frequencies)[::-1]
    labels = np.array(labels)[sorted_indices]
    frequencies = np.array(frequencies)[sorted_indices]

    # Sort the data in descending order
    sorted_indices = np.argsort(frequencies)[::-1]
    labels = np.array(labels)[sorted_indices]
    frequencies = np.array(frequencies)[sorted_indices]

    # Create the plot
    plt.figure(figsize=(5, 6))
    sns.barplot(x=labels, y=frequencies, palette="viridis")
    plt.title("Frequency of Actual Answers")
    plt.xlabel("Actual Answers")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

    plt.tight_layout(pad=0)
    # Save the plot
    plt.savefig("./data/charts/actual_frequency.pdf", bbox_inches="tight")


######################
# Main Functionality #
######################

# Load the file into a Pandas DataFrame
# file_path = "data/results/medqa/2023-10-20_14:22:26_multiagentdebategoogle/
# multiagentdebategoogle.json" # GPT-3
# file_path = "data/results/medqa/2023-10-23_09:18:14_multiagentdebategoogle/
# multiagentdebategoogle.json" # GPT-4
# file_path = "data/results/medqa/2023-10-27_11:39:42_singleagentqa/
# singleagentqa.json" # Single agent GPT-4
# Self-consistent GPT-4
# file_path = (
#     "data/results/medqa/2023-10-29_08:56:44_ensemblerefinementdebate"
#     "/ensemblerefinementdebate.json"
# )

# file_path = (
#     "data/results/usmle/roundrobindebateqa_04-11-2023_08:03:32/"
#     "roundrobindebateqa.json"
# )

# file_path = (
#     "data/results/usmle/2023-11-13_09:04:00_ensemblerefinementdebate/"
#     "ensemblerefinementdebate.json"
# )

file_path = (
    "data/results/cosmosqa/2023-12-04_08:42:33_chatevaldebate/" "chatevaldebate.json"
)

df = pd.read_json(file_path)

# Check the first few rows to understand the structure
# print("Data: ", df.head())

######################
# Data Preprocessing #
######################
df = extract_answers(df)

# Show the first few rows of the updated DataFrame to confirm the changes
# print(df.head())

######################
# Data Visualization #
######################
# print("Tree Structure: \n", create_tree_structure(df))
plot_answers_diversity(df)
plot_extracted_answer_frequency(df)
plot_actual_answer_frequency(df)

# Print random debate outputs

# Test the function
sample_debate_outputs_for_question_i(df, 0)
