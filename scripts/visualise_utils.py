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
import re
from collections import defaultdict
from math import sqrt
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import seaborn as sns
from sigfig import round as round_sigfig


def custom_sort_key(s):
    match = re.search(r"\d+", s)
    if match:
        number = int(match.group(0))
        prefix = s[: match.start()]
        return (prefix, number)
    else:
        return (s, 0)


SPINE_COLOR = "gray"


def latexify(fig_width=7, fig_height=5, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # Installs:
    # Website: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    # sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng
    # pip install latex
    # sudo apt install cm-super

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0  # noqa
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES
    size = 14
    params = {
        "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb}",
        "axes.labelsize": size,  # fontsize for x and y labels (was 10)
        "axes.titlesize": size,
        "font.size": size,  # was 8
        "legend.fontsize": 10,  # was 8
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


latexify()

name_mapping = {
    "debatellm.systems.ChatEvalDebate": "ChatEval",
    "debatellm.systems.EnsembleRefinementDebate": "Ensemble Refinement",
    "debatellm.systems.MultiAgentDebateGoogle": "Society of Mind",
    "debatellm.systems.MultiAgentDebateTsinghua": "Multi-Persona",
    "debatellm.systems.SingleAgentQA": "Single Agent",
    "debatellm.systems.SelfConsistency": "Self-Consistency",
    "debatellm.systems.Medprompt": "Medprompt",
}

marker_dict = {
    "Society of Mind": "^",
    "Ensemble Refinement": "s",
    "Multi-Persona": "x",
    "ChatEval": "o",
    "Single Agent": "*",
    "Self-Consistency": "D",
    "Medprompt": "P",
}

# options: usmle, medmcqa, mmlu, pubmedqa, medqa, ciar, cosmosqa, gpqa
dataset_to_name = {
    "usmle": "USMLE",
    "medqa": "MedQA",
    "medmcqa": "MedMCQA",
    "mmlu": "MMLU",
    "pubmedqa": "PubMedQA",
    "ciar": "CIAR",
    "cosmosqa": "CosmosQA",
    "gpqa": "GPQA",
}

muted_color_dict = {
    "Society of Mind": "#1f77b4",
    "Ensemble Refinement": "#ff7f0e",
    "ChatEval": "#2ca02c",
    "Self-Consistency": "#d62728",
    "Single Agent": "#9467bd",
    "Multi-Persona": "#8c564b",
    "Medprompt": "#e377c2",
}
unmuted_color_dict = muted_color_dict
# unmuted_color_dict = {
#     "Society of Mind": "#4c72b0",
#     "Ensemble Refinement": "#dd8452",
#     "Multi-Persona": "#8172b3",
#     "ChatEval": "#55a868",
#     "Single Agent": "#8B0000",
# }


metrics = [
    "eval/Agent_0/answered_correctly",
    "eval/Agent_0/avg_prompt_tokens",
    "eval/Agent_0/avg_response_length",
    "eval/Agent_0/avg_response_tokens",
    "eval/Agent_0/bullied_by_other",
    "eval/Agent_0/changed_answer",
    "eval/Agent_0/incorrect_parsed_rounds/Round_0",
    "eval/Agent_0/incorrect_parsed_rounds/Round_1",
    "eval/Agent_0/incorrectly_parsed_final_answer",
    "eval/Agent_0/any_incorrectly_parsed_answer",
    "eval/Agent_0/num_of_correct_rounds_when_correct",
    "eval/Agent_0/number_of_answers",
    "eval/Agent_0/percentage_of_correct_rounds_when_correct",
    "eval/Agent_0/prompt_tokens/Round_0",
    "eval/Agent_0/prompt_tokens/Round_1",
    "eval/Agent_0/relied_on_other",
    "eval/Agent_0/response_length_rounds/Round_0",
    "eval/Agent_0/response_length_rounds/Round_1",
    "eval/Agent_0/response_tokens/Round_0",
    "eval/Agent_0/response_tokens/Round_1",
    "eval/Agent_0/round_costs/Round_0",
    "eval/Agent_0/round_costs/Round_1",
    "eval/debate/agents_in_agreement",
    "eval/debate/any_correct_answer",
    "eval/debate/how_many_agents_changed",
    "eval/total_cost",
    "eval/avg_sec_per_question",
    "eval/debate/unique_first_answers",
    "eval/percent_complete",
    "eval/score/Step1_acc",
    "eval/score/Step1_count",
    "eval/score/Step2&3_acc",
    "eval/score/Step2&3_count",
    "eval/score/total_acc",
    "eval/score/total_count",
]

fixed_sorted_systems = [
    "Medprompt",
    "Society of Mind",
    "Ensemble Refinement",
    "Self-Consistency",
    "ChatEval",
    "Single Agent",
    "Multi-Persona",
]


def sigfig_round_small(num):
    if num >= 10.0:
        return int(round(num, 0))
    else:
        return round_sigfig(num, 2)


# Generate a unique discription depending on the system name
def get_unique_description(data, unique_id, include_prompts=False):
    full_system_name = data["config/system/_target_"][unique_id]
    display_name = name_mapping[full_system_name]
    system_name = full_system_name.split(".")[-1]
    if "MultiAgentDebateGoogle" in system_name:

        # Count the number of agents
        num_agents = 0
        while f"config/system/agents/Agent_{num_agents}/_target_" in data.keys():
            entry = data[f"config/system/agents/Agent_{num_agents}/_target_"][unique_id]

            # Check for NaN
            if type(entry) == float:
                break

            num_agents += 1

        rounds = int(data["config/system/num_rounds"][unique_id])

        summarize_answers = data["config/system/summarize_answers"][unique_id]

        descripter = f"{num_agents} agents, {rounds} rounds"
        if summarize_answers:
            descripter += ", summarized answers"
    elif "ChatEvalDebate" in system_name:
        debate_setting = data["config/system/debate_setting"][unique_id]
        rounds = int(data["config/system/num_rounds"][unique_id])
        descripter = f"{rounds} rounds, {debate_setting}"
    elif "SingleAgentQA" in system_name:
        if "spp_synergy" in data["config/system/name"][unique_id]:
            descripter = "SPP Synergy"
        else:
            agent_name = data["config/system/agents/Agent_0/_target_"][unique_id].split(
                "."
            )[-1]

            descripter = agent_name
    elif "EnsembleRefinementDebate" in system_name:
        agent_name = data["config/system/agents/Agent_0/_target_"][unique_id].split(
            "."
        )[-1]
        agg_steps = int(data["config/system/num_aggregation_steps"][unique_id])
        reason_steps = int(data["config/system/num_reasoning_steps"][unique_id])
        name = data["config/system/name"][unique_id]

        descripter = f"{reason_steps}:{agg_steps} - {name}, {agent_name}"
    elif "MultiAgentDebateTsinghua" in system_name:
        max_rounds = int(data["config/system/max_num_rounds"][unique_id])
        descripter = f"{max_rounds} rounds max"
    elif "Medprompt" in system_name:
        temp = data["config/system/agents/Agent_0/sampling/temperature"][unique_id]
        top_p = data["config/system/agents/Agent_0/sampling/top_p"][unique_id]
        descripter = f"temp: {temp}, top_p: {top_p}"
    else:
        descripter = ""

    # Add debate and agent prompts to the description
    if include_prompts:
        debate_prompt = data["config/system/debate_prompts/name"][unique_id]
        descripter += f", debate prompt: {debate_prompt}"

        agent_prompt = data["config/system/agents/Agent_0/prompt/name"][unique_id]
        descripter += f", agent prompt: {agent_prompt}"

    # full_descripter = full_descripter.replace("MultiAgentDebateGoogle", "MAD-Google")
    # full_descripter = full_descripter.replace(
    #     "MultiAgentDebateTsinghua", "MAD-Tsinghua"
    # )
    # full_descripter = full_descripter.replace("ChatEvalDebate", "ChatEval")
    # full_descripter = full_descripter.replace("SingleAgentQA", "SingleAgent")
    # full_descripter = full_descripter.replace("EnsembleRefinementDebate", "ER-Google")
    descripter = descripter.replace("ensemble_refinement", "ER")

    if descripter:
        return f"{display_name} - {descripter}"

    return display_name


def scale_min_values(value, min_scale=0.2):
    return min_scale + (1 - min_scale) * value


# Generate spider chart
def generate_spider_chart(
    data,
    metrics,
    save_path=None,
    add_legend=True,
    add_name=False,
    min_values=None,
    max_values=None,
):
    """
    Generates a spider chart based on the given data and metrics.
    """

    # Normalize the data by dividing by the max
    if min_values is None:
        min_values = data[metrics].min()
        max_values = data[metrics].max()

    range_values = max_values - min_values
    normalized_data = (data[metrics] - min_values) / range_values
    normalized_data = normalized_data.applymap(scale_min_values)  # New line

    # Prepare for plotting
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Use the last part of metric as label
    labels = [m.split("/")[-1] for m in metrics]
    # Place metric names with larger font, black color and increased radial distance
    for angle, label, min_val, max_val in zip(
        angles[:-1], labels, min_values, max_values
    ):
        if label == "percentage_of_correct_rounds_when_correct":
            label = "%_correct_rounds_when_correct"
        elif label == "num_of_correct_rounds_when_correct":
            label = "#_correct_rounds_when_correct"

        # Remove underscore and uppercase all first letters
        label = label.replace("_", " ").title()

        label = label.replace("Total Acc", "Accuracy")
        label = label.replace("Agents In Agreement", "Final Round Consensus")

        ax.text(
            angle,
            1.7,
            f"{label}\n(Min: {sigfig_round_small(min_val)}, Max: {sigfig_round_small(max_val)})",
            ha="center",
            va="top",
            color="black",
            size=30,
        )
        # Add light grey line to each axis
        ax.axvline(angle, color="lightgrey", linestyle="--", lw=1)

    ax.set_rlabel_position(30)
    ax.set_xticks([])  # Remove the degrees (tick marks)
    ax.set_yticklabels([])  # Remove the y-axis labels (radii)
    ax.spines["polar"].set_visible(False)  # Remove the outer circle

    # Plot each run
    for idx, run in normalized_data.iterrows():
        values = run.tolist()
        values += values[:1]
        descriptor = get_unique_description(data, idx)

        if not add_name:
            descriptor = descriptor.split(" - ", 1)[-1]

        ax.plot(angles, values, linewidth=2, linestyle="solid", label=descriptor)
        ax.fill(angles, values, alpha=0.25)

    if add_legend:
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0.0), fontsize=30)

    # Save the figure with high resolution if save_path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


# Generate scatter chart
def generate_scatter_chart(
    data,
    x_metric,
    x_label,
    title_name,
    save_path,
    dataset,
    sparse_legend=False,
    # marker_list=None,
    font_size=14,
):
    plt.figure(figsize=(15, 6))

    # Organize data by system name
    organized_data = defaultdict(list)

    # Filter out all runs with scores less than 0.45
    # data = data[data["eval/score/total_acc"] > 0.45]

    # If the number of aggregations steps is 0, then the system is a single agent
    aggr_name = "config/system/num_aggregation_steps"
    data["config/system/_target_"] = data.apply(
        lambda row: "debatellm.systems.SingleAgentQA"
        if aggr_name in row and row[aggr_name] == 0
        else row["config/system/_target_"],
        axis=1,
    )

    # If the number of aggregations steps is 0 and reasoning steps is 1,
    # then we consider it a single agent protocol.
    data = update_self_consistency_names(data)

    # Assuming the unique experiment IDs and 'config/system/_target_' values are aligned
    for exp_id in data.index.tolist():
        system_name = data["config/system/_target_"][exp_id]
        organized_data[system_name].append(exp_id)

    # Sort by descriptor to keep the system colors consistent
    sorted_system_name = sorted(organized_data.keys())
    for _, system_name in enumerate(sorted_system_name):
        descriptors = [
            get_unique_description(data, unique_id)
            for unique_id in organized_data[system_name]
        ]

        # Sort by descriptor to keep the system markers consistent
        zip_list = zip(descriptors, organized_data[system_name])
        zip_list = sorted(zip_list, key=lambda x: x[0])

        for j, (discriptor, unique_id) in enumerate(zip_list):
            x = data[x_metric][unique_id]
            y = data["eval/score/total_acc"][unique_id]

            if sparse_legend:
                marker = marker_dict[name_mapping[system_name]]
                if j == 0:
                    label = name_mapping[system_name]
                else:
                    label = None
            else:
                # marker = marker_list[j % len(marker_list)]
                label = discriptor
            plt.scatter(
                x,
                y,
                label=label,
                marker=marker,
                c=unmuted_color_dict[name_mapping[system_name]],
                s=100,
            )

    plt.xlabel(x_label, fontsize=font_size)

    dataset_name = dataset_to_name[dataset]
    plt.ylabel(f"Accuracy {dataset_name} (out of 1.0)", fontsize=font_size)
    # plt.title(title_name)

    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort them using the custom_sort_key function
    sorted_indices = sorted(
        range(len(labels)), key=lambda k: custom_sort_key(labels[k])
    )

    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Add sorted legend
    plt.legend(sorted_handles, sorted_labels, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Log-transform x-axis
    plt.gca().set_xscale("log")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def get_paper_dataset_ranges(dataset):
    if dataset == "all":
        run_range = [
            [3886, 4065],
            [3832, 3885],
            [3727, 3780],
            [3904, 4058],
            [4116, 4163],
        ]
    elif dataset == "medqa":
        run_range = [
            [3886, 4065],
            [4116, 4163],
        ]
    elif dataset == "pubmedqa":
        run_range = [
            [3832, 3885],
            [4116, 4163],
        ]
    elif dataset == "mmlu":
        run_range = [
            [3727, 3780],
            [4116, 4163],
        ]
    elif dataset in ["ciar", "gpqa", "cosmosqa"]:
        run_range = [
            [3904, 4058],
            [4116, 4163],
        ]
    return run_range


def filter_results_for_paper(runs_table_df, dataset):
    # Medical datasets should have 44 runs.
    # Non-medical datasets should have 36 runs.

    dataset = dataset.lower()

    # Filter out GPT-4 and PaLM
    runs_table_df = runs_table_df[
        runs_table_df["config/system/agents/Agent_0/engine"] != "chat-bison@001"
    ]
    runs_table_df = runs_table_df[
        runs_table_df["config/system/agents/Agent_0/engine"] != "gpt-4"
    ]

    # Filter out Multi-Persona agreement intensities greater than 0
    # Filter out runs where 'ai' is greater than -1, ignoring NaN values
    runs_table_df = runs_table_df[
        (runs_table_df["config/system/agreement_intensity"] <= -1)
        | runs_table_df["config/system/agreement_intensity"].isna()
    ]

    # Filter out duplicates
    # Create a mask for rows with agreement_intensity of -1 and max_num_rounds of 3
    mask = (runs_table_df["config/system/agreement_intensity"] == -1) & (
        runs_table_df["config/system/max_num_rounds"] == 3
    )

    # Apply the mask and drop duplicates based on 'config/dataset/eval_dataset'
    # This operation only affects rows that meet the condition, keeping the first occurrence
    filtered_duplicates = runs_table_df[mask].drop_duplicates(
        subset=["config/dataset/eval_dataset"], keep="first"
    )

    # Combine the filtered duplicates with the rest of the DataFrame, excluding the original
    # duplicates
    runs_table_df = pd.concat(
        [runs_table_df[~mask], filtered_duplicates], ignore_index=True
    )

    # if dataset == "medqa":
    # Filter out scores less than 0.3
    # TODO: Is this a good idea?
    # runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] >= 0.55]
    # elif dataset == "pubmedqa":
    #     # Filter out scores less than 0.4:
    #     runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] >= 0.4]
    # elif dataset == "mmlu":
    #     # Filter out scores less than 0.6:
    #     runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] >= 0.6]
    # elif dataset == "ciar":
    #     # Filter out scores less than 0.3:
    #     runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] >= 0.3]
    # elif dataset == "gpqa":
    #     # Filter out scores less than 0.2:
    #     runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] >= 0.2]
    return runs_table_df


def get_dataset_runs(run_range: List[int], dataset: str = None, engine: str = None):
    # Later runs: [2244, 2381]  # Faithful runs: 2244-2323

    api_token = os.environ["TM_NEPTUNE_API_TOKEN"]
    project = neptune.init_project(
        project="Anon/debatellm",
        api_token=api_token,
        mode="read-only",
    )

    def fetch_runs_in_batches(run_range, batch_size=1000):
        all_dataframes = []
        start, end = run_range
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            batch_ids = [f"TRUEM-{run_id}" for run_id in range(batch_start, batch_end)]
            batch_df = project.fetch_runs_table(id=batch_ids).to_pandas()
            all_dataframes.append(batch_df)

        return pd.concat(all_dataframes, ignore_index=True)

    runs_table_df = fetch_runs_in_batches(run_range)

    # Set the index based on the custom names
    runs_table_df = runs_table_df.set_index("sys/id")

    # Filter out all runs not initiated by k.tessara or anon
    # runs_table_df = runs_table_df[runs_table_df["sys/owner"] == "anon"]
    runs_table_df = runs_table_df[
        runs_table_df["sys/owner"].isin(["anon", "anon"])
    ]

    # Filter out all runs that where less than 80% completed
    # runs_table_df = runs_table_df[runs_table_df["eval/percent_complete"] >= 40.0]
    runs_table_df = runs_table_df[runs_table_df["eval/percent_complete"] >= 100.0]

    # At least 10 evaluations
    runs_table_df["config/max_eval_count"] = pd.to_numeric(
        runs_table_df["config/max_eval_count"], errors="coerce"
    )
    condition = pd.isna(runs_table_df["config/max_eval_count"]) | (
        runs_table_df["config/max_eval_count"] > 20
    )
    runs_table_df = runs_table_df[condition]

    # TODO: This is only temporary needed
    runs_table_df = filter_results_for_paper(runs_table_df, dataset)

    if dataset is not None:
        # Filter out all runs that are not from the current dataset
        runs_table_df = runs_table_df[
            runs_table_df["config/dataset/eval_dataset"] == dataset
        ]

    if engine is not None:
        runs_table_df = runs_table_df[
            runs_table_df["config/system/agents/Agent_0/engine"].apply(
                lambda x: engine in x if isinstance(x, str) else False
            )
        ]

    # Discard all keys with monitoring in their name.
    runs_table_df = runs_table_df[
        [key for key in runs_table_df.keys() if "eval/" in key or "config/" in key]
    ]

    name_mapping = {
        "debatellm.systems.ChatEvalDebate": "ChatEval",
        "debatellm.systems.EnsembleRefinementDebate": "Ensemble Refinement",
        "debatellm.systems.MultiAgentDebateGoogle": "Society of Mind",
        "debatellm.systems.MultiAgentDebateTsinghua": "Multi-Persona",
        "debatellm.systems.SingleAgentQA": "Single Agent",
        "debatellm.systems.Medprompt": "Medprompt",
    }

    runs_table_df = runs_table_df[runs_table_df["config/system/_target_"].notna()]
    runs_table_df["system_name"] = runs_table_df["config/system/_target_"].apply(
        lambda x: name_mapping.get(x, x)
    )

    create_num_rounds_column(runs_table_df)
    create_num_agents_column(runs_table_df)

    create_use_summarizer(runs_table_df)
    create_use_judge(runs_table_df)
    runs_table_df.loc[
        (runs_table_df["num_agents"] == 1)
        & (runs_table_df["system_name"] != "Medprompt"),
        "system_name",
    ] = "Single Agent"
    runs_table_df.loc[
        (runs_table_df["system_name"] == "Ensemble Refinement")
        & (runs_table_df["config/system/num_aggregation_steps"] == 0),
        "system_name",
    ] = "Self-Consistency"

    create_use_examples(runs_table_df)
    create_num_api_call(runs_table_df)

    return runs_table_df


def get_scatter_plot(metrics, runs_table_df, legend=True, dataset="", save_path=None):
    if metrics == "Average tokens per question":
        key = "eval/avg_tokens_per_question"
    elif metrics == "Total cost":
        key = "eval/total_cost"
    elif metrics == "Average seconds per question":
        key = "eval/avg_sec_per_question"
    else:
        raise ValueError("Invalid metrics")
    # system_order = ['Society of Mind', 'Single Agent']
    system_order = [
        "Society of Mind",
        "Ensemble Refinement",
        "ChatEval",
        "Self-Consistency",
        "Single Agent",
        "Multi-Persona",
        "Medprompt",
    ]
    runs_table_df_plot = runs_table_df

    runs_table_df_plot.rename(
        columns={
            "num_agents": "Number of agents",
            "system_name": r"\textbf{System name}",
            "num_api_calls": r"\textbf{Average API calls}",
        },
        inplace=True,
    )

    # plt.figure(figsize=(10, 6))
    if legend:
        plt.figure(figsize=(8.2, 4))
    else:
        plt.figure(figsize=(6, 4))

    sns.scatterplot(
        x=key,
        y="eval/score/total_acc",
        hue=r"\textbf{System name}",
        size=r"\textbf{Average API calls}",
        sizes=(50, 200),
        palette="tab10",
        marker="o",
        hue_order=system_order,
        data=runs_table_df_plot,
    )
    plt.xscale("log")
    plt.xlabel(f"{metrics}")
    plt.ylabel(f"{dataset} Accuracy (out of 1.0)")
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        # pass
    else:
        plt.legend().remove()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_line_chart_with_error_bar(
    runs_table_df, system_name, metric_name, save_path, font_size=14
):
    # Filter out all runs that does not have the correct system name
    runs_table_df = runs_table_df[
        runs_table_df["config/system/_target_"] == f"debatellm.systems.{system_name}"
    ]

    if system_name == "EnsembleRefinementDebate":
        runs_table_df = runs_table_df[
            runs_table_df["config/system/num_aggregation_steps"] > 0
        ]
    elif system_name == "MultiAgentDebateTsinghua":
        print(runs_table_df["config/system/agreement_intensity"])

        # Agreement intensity must be -1
        runs_table_df = runs_table_df[
            runs_table_df["config/system/agreement_intensity"] == -1
        ]

    if metric_name == "num_agents":

        # Count the number of agents
        for unique_id in runs_table_df.index.tolist():
            num_agents = 0
            while (
                f"config/system/agents/Agent_{num_agents}/_target_"
                in runs_table_df.keys()
            ):
                entry = runs_table_df[
                    f"config/system/agents/Agent_{num_agents}/_target_"
                ][unique_id]

                # Check for NaN
                if type(entry) == float:
                    break

                num_agents += 1
            runs_table_df.loc[unique_id, metric_name] = num_agents

    # Group by metric_name and calculate mean and standard deviation
    grouped = runs_table_df.groupby(metric_name)["eval/score/total_acc"].agg(
        ["mean", "std"]
    )

    # Replace NaN values in 'std' column with 0
    grouped["std"].fillna(0, inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"], fmt="-o", capsize=5
    )
    plt.fill_between(
        grouped.index,
        (grouped["mean"] - grouped["std"]),
        (grouped["mean"] + grouped["std"]),
        color="b",
        alpha=0.1,
    )

    # metric_title = metric_name.split("/")[-1].replace("_", " ").title()
    # plt.title(system_name)
    # plt.xlabel(metric_title, fontsize=font_size)
    plt.ylabel("Accuracy MedQA", fontsize=font_size)
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def generate_line_chart(
    runs_table_df, system_names, metric_name, save_path, font_size=14
):
    plt.figure(figsize=(10, 6))

    for system_name in system_names:
        full_name = f"debatellm.systems.{system_name}"

        # Filter by system name
        filtered_df = runs_table_df[
            runs_table_df["config/system/_target_"] == full_name
        ]

        # Group by metric_name and calculate mean
        grouped = filtered_df.groupby(metric_name)["eval/score/total_acc"].mean()

        # Plot line for each system
        label = name_mapping[full_name]
        plt.plot(grouped.index, grouped, label=label, color=unmuted_color_dict[label])

    metric_title = metric_name.split("/")[-1].replace("_", " ").title()

    plt.xlabel(metric_title, fontsize=18)
    plt.ylabel("USMLE accuracy (out of 1.0)", fontsize=18)
    plt.legend()
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)


def update_self_consistency_names(runs_table_df, name_mapping=None):
    # If the number of aggregations steps is 0 and reasoning steps is 1,
    # then we consider it a single agent protocol.

    single_agent_name = "debatellm.systems.SingleAgentQA"
    single_agent_name = (
        single_agent_name if name_mapping is None else name_mapping[single_agent_name]
    )

    self_consistency_name = "debatellm.systems.SelfConsistency"
    self_consistency_name = (
        self_consistency_name
        if name_mapping is None
        else name_mapping[self_consistency_name]
    )

    aggr_name = "config/system/num_aggregation_steps"
    runs_table_df["config/system/_target_"] = runs_table_df.apply(
        lambda row: single_agent_name
        if aggr_name in row
        and row[aggr_name] == 0
        and row["config/system/num_reasoning_steps"] == 1
        else row["config/system/_target_"],
        axis=1,
    )

    # If the number of aggregations steps is 0 and reasoning steps is more than 1,
    # the system is considered Self-Consitency.
    runs_table_df["config/system/_target_"] = runs_table_df.apply(
        lambda row: self_consistency_name
        if aggr_name in row
        and row[aggr_name] == 0
        and row["config/system/num_reasoning_steps"] > 1
        else row["config/system/_target_"],
        axis=1,
    )
    return runs_table_df


def generate_box_chart(
    runs_table_df,
    dataset,
    y_metric="eval/score/total_acc",
    save_path="./data/charts/total_acc_box.pdf",
):
    plt.figure(figsize=(6, 4))
    sns.set_palette("tab10")

    # Filter out all runs with scores less than 0.45
    # runs_table_df = runs_table_df[runs_table_df["eval/score/total_acc"] > 0.45]

    # Replace system names in the dataframe using the mapping
    runs_table_df = (
        runs_table_df.copy()
    )  # Create a copy to avoid modifying the original dataframe
    runs_table_df["config/system/_target_"] = runs_table_df[
        "config/system/_target_"
    ].replace(name_mapping)

    # If the number of aggregations steps is 0 and reasoning steps is 1,
    # then we consider it a single agent protocol.
    runs_table_df = update_self_consistency_names(runs_table_df, name_mapping)

    # Calculate average values and sort systems by highest to lowest average
    avg_values = (
        runs_table_df.groupby("config/system/_target_")[y_metric]
        .mean()
        .sort_values(ascending=False)
    )
    all_sorted_systems = avg_values.index.tolist()

    sorted_systems = []

    # Add systems in the order of fixed_sorted_systems
    for system in fixed_sorted_systems:
        if system in all_sorted_systems:
            sorted_systems.append(system)

    # Create the boxplot
    # Create the boxplot
    sns.boxplot(
        x=runs_table_df["config/system/_target_"],
        y=runs_table_df[y_metric],
        order=sorted_systems,
        palette=muted_color_dict,
        linewidth=1,
        fliersize=5,
        saturation=0.5,
        showmeans=True,
        meanprops={
            "marker": ".",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": "3",
        },
    )

    # Adjust x-axis labels
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 2 != 0:  # For every second label
            label.set_position((label.get_position()[0], label.get_position()[1] - 0.1))

    plt.tight_layout()
    plt.show()

    # Soften edges and remove top and right spines
    # sns.despine(left=True, bottom=True)
    # ax.set_facecolor('.9')

    # Soften text color
    plt.xticks(fontsize=11, color="0.3")  # 0.3 is light gray
    plt.yticks(fontsize=13, color="0.3")  # 0.3 is light gray

    dataset_name = dataset_to_name[dataset]

    plt.xlabel("", color="0.3")
    plt.ylabel(f"{dataset_name} Accuracy (out of 1.0)", color="0.3")

    # Add a point that indicates what we achieved when improving Tshinghua

    # Add a point for a specific protocol
    # protocol_name = "Multi-Persona"
    # point_value = 0.665  # Replace with the actual point value

    # # Get x-coordinate of the protocol
    # protocol_x_coord = sorted_systems.index(protocol_name)

    # plt.scatter(
    #     x=[protocol_x_coord],  # x-coordinate of the box
    #     y=[point_value],  # y-coordinate (point value)
    #     color="red",
    #     s=100,  # Size of the point
    #     label='Special Point',  # Label for legend
    #     zorder=5,  # Make sure point appears on top
    #     marker='x'  # Diamond shape
    # )

    # plt.xlabel("")
    # plt.ylabel("Accuracy Accuracy (out of 1.0)")
    # plt.title("Total Accuracy by System")

    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def create_num_rounds_column(df):
    # Use 'config/system/num_rounds' if it exists and is not NaN
    if "config/system/num_rounds" in df.columns:
        df["num_rounds"] = df["config/system/num_rounds"]

    # Use 'config/system/max_num_rounds' if 'config/system/num_rounds' is NaN or doesn't exist
    if "config/system/max_num_rounds" in df.columns:
        df["num_rounds"].fillna(df["config/system/max_num_rounds"], inplace=True)

    # Use 1 if both columns are NaN or don't exist
    df["num_rounds"].fillna(1, inplace=True)


def create_num_agents_column(df):
    num_agents_list = []

    for _, row in df.iterrows():
        max_agent_index = -1
        system_name = row.get("system_name", None)

        # Find the greatest 'i' such that 'config/system/agents/Agent_{i}/_target_' is not NaN
        for i in range(100):  # Assuming an upper limit of 100 agents
            agent_target_key = f"config/system/agents/Agent_{i}/_target_"
            if agent_target_key in df.columns and not pd.isna(row[agent_target_key]):
                max_agent_index = i
            else:
                break

        max_agent_index += 1  # +1 as the index starts from 0

        # Apply the special conditions based on the system name
        if (
            system_name == "ChatEval"
            and row.get("config/system/debate_setting", None)
            != "simultaneous_talk_with_summarizer"
        ):  # because of summarizer
            max_agent_index -= 1
        elif system_name == "Ensemble Refinement":
            num_reasoning_steps = row.get("config/system/num_reasoning_steps", 0)
            num_aggregation_steps = row.get("config/system/num_aggregation_steps", 0)
            max_agent_index = num_reasoning_steps + num_aggregation_steps
        elif system_name == "Society of Mind" and row.get(
            "config/system/summarize_answers", None
        ):
            max_agent_index += 1

        num_agents_list.append(max_agent_index)

    df["num_agents"] = num_agents_list


def create_num_api_call(df):
    num_api_list = []

    for _, row in df.iterrows():

        system_name, num_agents, num_rounds = (
            row.get("system_name", None),
            row.get("num_agents", None),
            row.get("config/system/num_rounds", None),
        )

        if system_name in ["Ensemble Refinement", "Self-Consistency"]:
            num_api_calls = num_agents

        elif system_name == "Single Agent" or system_name == "SPP":
            num_api_calls = 1

        elif system_name == "Multi-Persona":
            num_api_calls = num_agents * (row.get("eval/debate/num_rounds", None) + 1)

        elif system_name == "Society of Mind":
            num_api_calls = num_agents * num_rounds

        elif system_name == "ChatEval":
            num_api_calls = num_agents * num_rounds

        elif system_name == "Medprompt":
            num_api_calls = row.get("config/system/num_reasoning_steps", None)

        else:
            raise ValueError(f"Unknown system name: {system_name}")

        num_api_list.append(num_api_calls)

    df["num_api_calls"] = num_api_list


def create_use_summarizer(df):
    use_summarizer_list = []
    for _, row in df.iterrows():
        system_name = row.get("system_name", None)

        if system_name == "ChatEval":
            use_summarizer = (
                row.get("config/system/debate_setting", None)
                == "simultaneous_talk_with_summarizer"
            )
        elif system_name == "Society of Mind":
            use_summarizer = row.get("config/system/summarize_answers", None)
        elif system_name == "Ensemble Refinement":
            use_summarizer = row.get("config/system/num_aggregation_steps", None) > 0
        else:
            use_summarizer = False

        use_summarizer_list.append(use_summarizer)

    df["use_summarizer"] = use_summarizer_list


def create_use_judge(df):
    use_judge_list = []
    for _, row in df.iterrows():
        system_name = row.get("system_name", None)

        if system_name == "Ensemble Refinement":
            use_judge = row.get("config/system/num_aggregation_steps", None) > 0
        elif system_name == "Multi-Persona":
            use_judge = True
        else:
            use_judge = False

        use_judge_list.append(use_judge)

    df["use_judge"] = use_judge_list


def create_use_examples(df):
    use_examples_list = []
    for _, row in df.iterrows():
        system_name = row.get("system_name", None)

        if (
            system_name == "Ensemble Refinement"
            or system_name == "Self-Consistency"
            or system_name == "Single Agent"
        ):
            use_examples = row.get("config/system/use_few_shot_examples", None)
            if isinstance(use_examples, str):
                use_examples = True
            else:
                use_examples = False

        elif system_name == "SPP":
            use_examples = True
        else:
            use_examples = False

        use_examples_list.append(use_examples)

    df["use_examples"] = use_examples_list
