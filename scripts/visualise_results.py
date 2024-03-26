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

import neptune
import pandas as pd

from scripts.visualise_utils import (
    filter_results_for_paper,
    generate_box_chart,
    generate_line_chart,
    generate_line_chart_with_error_bar,
    generate_scatter_chart,
    generate_spider_chart,
    get_paper_dataset_ranges,
    get_unique_description,
    metrics,
    update_self_consistency_names,
)

# Set the chart type and dataset to plot
chart_type = "spider"  # "bar", "spider", "scatter", "line", line_error_bar, "box",
# "config_table"
DATASET = (
    "medqa"  # "medqa", "usmle", "pubmedqa", "mmlu", "cosmosqa", "ciar", "gpqa", "all"
)
# RUN_RANGE = [
#     [3904, 4058],
# ]  # All: [2244, 2441]  # Faithful runs: [2244, 2323]


# Hardcode the ranges based on the names
dataset = DATASET.lower()
RUN_RANGE = get_paper_dataset_ranges(dataset)

sparse_legend = True

# Initialize Neptune
API_TOKEN = os.environ["TM_NEPTUNE_API_TOKEN"]
project = neptune.init_project(
    project="Anon/debatellm",
    mode="read-only",
)

# ADD TRUEM- to the beginning of the run ids
if type(RUN_RANGE[0]) == int:
    full_run_ids = [
        f"TRUEM-{run_id}" for run_id in range(RUN_RANGE[0], RUN_RANGE[1] + 1)
    ]
else:
    full_run_ids = []
    for run in RUN_RANGE:
        for run_id in range(run[0], run[1] + 1):
            full_run_ids.append(f"TRUEM-{run_id}")

if chart_type == "spider" and False:
    # Best worst
    # run_ids = [
    #     2293,
    #     2290,
    #     2273,  # Society of Mind
    #     2303,
    #     2304,
    #     2302,  # ChatEval
    #     2318,
    #     2319,
    #     2320,  # Multi-Persona
    # ]

    run_ids = [
        3991,  # Society of Mind
        4003,
        4004,
        4006,
        4022,
        4025,
        4030,
        4031,  # Multi-Persona
        4036,
        4045,
        4046,
        4048,  # ChatEval
        4050,
        4057,
        4059,
        4060,
        4061,
    ]

    # ADD TRUEM- to the beginning of the run ids
    full_run_ids = [f"TRUEM-{run_id}" for run_id in run_ids]


# Delete this
# run_ids = [
#     2868, 2869, 2870, 2871, 2872, 3003, 3004, 3005, 3006, 3007, # Single agent

#     2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889,
#     2890, 2891, # Google MAD
# ]

# # ADD TRUEM- to the beginning of the run ids
# full_run_ids = [f"TRUEM-{run_id}" for run_id in run_ids]
# Delete this

runs_table_df = project.fetch_runs_table(id=full_run_ids).to_pandas()

# Set the index based on the custom names
runs_table_df = runs_table_df.set_index("sys/id")

# Filter out all runs not initiated by k.tessara
runs_table_df = runs_table_df[runs_table_df["sys/owner"].isin(["anon", "anon"])]

# Filter out all runs that where less than 80% completed
runs_table_df = runs_table_df[runs_table_df["eval/percent_complete"] == 100.0]

# There must be at least 40 questions answered
runs_table_df = runs_table_df[runs_table_df["eval/score/total_count"] >= 15]

if chart_type != "spider":
    runs_table_df = filter_results_for_paper(runs_table_df, dataset)

# Filter out all runs that are not from the current dataset
if dataset != "all":
    runs_table_df = runs_table_df[
        runs_table_df["config/dataset/eval_dataset"] == dataset
    ]

# Discard all keys with monitoring in their name.
runs_table_df = runs_table_df[
    [key for key in runs_table_df.keys() if "eval/" in key or "config/" in key]
]

for key in runs_table_df.keys():
    if "few_shot" in key:
        print(key)

if chart_type == "bar":
    raise NotImplementedError("Bar charts are not implemented yet")
    # for metric in tqdm(metrics):
    #     # Extract data for the current metric
    #     metric_data = runs_table_df[metric]

    #     # Plot
    #     plot_bar(metric_data, run_names)
elif chart_type == "spider":

    # Generate spider charts for debate metrics
    debate_metrics = ["eval/score/total_acc"]
    debate_metrics.extend([m for m in metrics if m.startswith("eval/debate")])

    # Filter out all MultiAgentDebateTsinghua runs with agreement_intensity >= 0
    runs_table_df = runs_table_df[
        ~(
            (
                runs_table_df["config/system/_target_"]
                == "debatellm.systems.MultiAgentDebateTsinghua"
            )
            & (runs_table_df["config/system/agreement_intensity"] >= 0)
        )
    ]

    # Multi_Persona
    mp_run_table_df = runs_table_df[
        runs_table_df["config/system/_target_"]
        == "debatellm.systems.MultiAgentDebateTsinghua"
    ]
    # Filter to only take the top 3 runs
    mp_run_table_df = mp_run_table_df.nlargest(3, "eval/score/total_acc")

    # ChatEval
    ce_run_table_df = runs_table_df[
        runs_table_df["config/system/_target_"] == "debatellm.systems.ChatEvalDebate"
    ]

    # Filter to only take the top 3 runs
    ce_run_table_df = ce_run_table_df.nlargest(3, "eval/score/total_acc")

    # Society of Mind
    som_run_table_df = runs_table_df[
        runs_table_df["config/system/_target_"]
        == "debatellm.systems.MultiAgentDebateGoogle"
    ]
    # Filter to only take the top 3 runs
    som_run_table_df = som_run_table_df.nlargest(3, "eval/score/total_acc")

    # Create a new datafrom concatenating the best runs from each system
    runs_table_df = pd.concat([mp_run_table_df, ce_run_table_df, som_run_table_df])

    min_values = runs_table_df[debate_metrics].min()
    max_values = runs_table_df[debate_metrics].max()

    generate_spider_chart(
        mp_run_table_df,
        debate_metrics,
        save_path="./data/charts/mp_debate_metrics.pdf",
        add_legend=True,
        add_name=False,
        min_values=min_values,
        max_values=max_values,
    )

    generate_spider_chart(
        ce_run_table_df,
        debate_metrics,
        save_path="./data/charts/ce_debate_metrics.pdf",
        add_legend=True,
        min_values=min_values,
        max_values=max_values,
    )

    generate_spider_chart(
        som_run_table_df,
        debate_metrics,
        save_path="./data/charts/som_debate_metrics.pdf",
        add_legend=True,
        min_values=min_values,
        max_values=max_values,
    )

    # Filter out only the best runs (according to accuracy) for each system
    best_mp_run = mp_run_table_df.loc[mp_run_table_df["eval/score/total_acc"].idxmax()]
    best_ce_run = ce_run_table_df.loc[ce_run_table_df["eval/score/total_acc"].idxmax()]
    best_som_run = som_run_table_df.loc[
        som_run_table_df["eval/score/total_acc"].idxmax()
    ]

    # Concatenate the best runs
    best_runs = pd.concat([best_mp_run, best_ce_run, best_som_run], axis=1).T

    # Generate spider charts for the best runs
    generate_spider_chart(
        best_runs,
        debate_metrics,
        save_path="./data/charts/best_debate_metrics.pdf",
        add_legend=True,
        add_name=True,
        min_values=min_values,
        max_values=max_values,
    )

elif chart_type == "scatter":

    # Generate total accuracy vs average seconds per question chart
    generate_scatter_chart(
        runs_table_df,
        x_metric="eval/avg_sec_per_question",
        x_label="Average Seconds per Question",
        title_name="Total Accuracy vs Average Seconds per Question",
        save_path="./data/charts/total_acc_vs_avg_sec_per_question.pdf",
        sparse_legend=sparse_legend,
        dataset=dataset,
    )

    # Generate total accuracy vs total cost chart
    generate_scatter_chart(
        runs_table_df,
        x_metric="eval/total_cost",
        x_label="Total cost (\$)",  # noqa
        title_name="Total Accuracy vs Total Cost",
        save_path="./data/charts/total_acc_vs_total_cost.pdf",
        sparse_legend=sparse_legend,
        dataset=dataset,
    )

    # Generate avg tokens per question chart
    generate_scatter_chart(
        runs_table_df,
        x_metric="eval/avg_tokens_per_question",
        x_label="Average Tokens per Question",
        title_name="Total Accuracy vs Average Tokens per Question",
        save_path="./data/charts/total_acc_vs_avg_tokens_per_question.pdf",
        sparse_legend=sparse_legend,
        dataset=dataset,
    )
elif chart_type == "line_error_bar":

    plot_info = [
        (
            "MultiAgentDebateTsinghua",
            "config/system/max_num_rounds",
        ),
        ("ChatEvalDebate", "config/system/num_rounds"),
        ("MultiAgentDebateGoogle", "config/system/num_rounds"),
        ("MultiAgentDebateGoogle", "num_agents"),
        ("EnsembleRefinementDebate", "config/system/num_reasoning_steps"),
        ("EnsembleRefinementDebate", "config/system/num_aggregation_steps"),
    ]

    for system_name, metric_name in plot_info:
        generate_line_chart_with_error_bar(
            runs_table_df,
            system_name=system_name,
            metric_name=metric_name,
            save_path=f"./data/charts/{system_name}_{metric_name.split('/')[-1]}.pdf",
        )
elif chart_type == "line":
    plot_info = [
        (
            ["MultiAgentDebateTsinghua", "MultiAgentDebateGoogle", "ChatEvalDebate"],
            "config/system/agreement_intensity",
        ),
    ]

    for system_names, metric_name in plot_info:

        # String together the system names
        string_system_names = "_".join(system_names)

        generate_line_chart(
            runs_table_df,
            system_names=system_names,
            metric_name=metric_name,
            save_path=f"./data/charts/{metric_name.split('/')[-1]}.pdf",
        )
elif chart_type == "box":
    generate_box_chart(
        runs_table_df,
        save_path=f"./data/charts/{dataset}_total_acc_box.pdf",
        dataset=dataset,
    )
elif chart_type == "total_cost":
    # Calculate the total cost of all runs
    total_cost = runs_table_df["eval/total_cost"].sum()
    print(f"Total cost of all runs: {total_cost}")
elif chart_type == "config_table":

    # Fix Self-Consistency names
    runs_table_df = update_self_consistency_names(runs_table_df)

    pd.set_option("display.precision", 2)

    all_few_shot = runs_table_df[
        "config/system/agents/Agent_0/few_shot_examples/input_text/0"
    ]

    results = {
        "mmlu": [],
        "pubmedqa": [],
        "medqa": [],
        "cosmosqa": [],
        "ciar": [],
        "gpqa": [],
    }
    datasets = results.keys()
    for unique_id in runs_table_df.index:

        score = runs_table_df["eval/score/total_acc"][unique_id]
        cost = runs_table_df["eval/total_cost"][unique_id]
        dataset = runs_table_df["config/dataset/eval_dataset"][unique_id]

        description = get_unique_description(
            runs_table_df, unique_id, include_prompts=True
        )

        examples = all_few_shot[unique_id]
        incorrectly_parsed_agent_0 = runs_table_df[
            "eval/Agent_0/any_incorrectly_parsed_answer"
        ][unique_id]

        # Round score and cost
        score = round(score, 2)
        cost = round(cost, 2)
        system_name, description = description.split(" - ", 1)
        if "single_agent" in description:
            system_name = "Single Agent"

        if "1:1" in description:
            print("REMOVED: ", unique_id)
            continue

        # Remove the Improved Multi-Persona Experiments
        # if 'TRUEM-2359' in unique_id:
        #     continue

        description, agent_prompt = description.split(", agent prompt: ", 1)
        description, debate_prompt = description.split(", debate prompt: ", 1)

        if not pd.isna(examples) and examples:
            agent_prompt = agent_prompt + " + FS"

        if system_name == "Single Agent":
            debate_prompt = "-"

        if "PaLM" in description:
            agent = "PaLM"
        else:
            agent = "GPT3.5"

        results[dataset].append(
            (
                system_name,
                debate_prompt,
                agent_prompt,
                description,
                agent,
                unique_id,
                score,
                cost,
                incorrectly_parsed_agent_0,
            )
        )

    # Create a dataframe for each dataset
    results_df = {}
    for dataset in datasets:
        column_names = [
            "System Name",
            "Debate Prompt",
            "Agent Prompt",
            "Config",
            "Agents",
            "NeptuneID",
            "Score (%s)" % dataset.upper(),
            "Cost \$ (%s)" % dataset.upper(),  # noqa
            "Agent0: Incorrectly Parsed",
        ]

        df = pd.DataFrame(results[dataset], columns=column_names).sort_values(
            [
                "System Name",
                "Debate Prompt",
                "Agent Prompt",
                "Agents",
                "Score (%s)" % dataset.upper(),
            ]
        )
        # df["ID (%s)" % dataset.upper()] = df["NeptuneID"]
        # df["IP (%s)" % dataset.upper()] = df["Agent0: Incorrectly Parsed"]

        df.drop(columns=["NeptuneID", "Agent0: Incorrectly Parsed"], inplace=True)
        results_df[dataset] = df.drop_duplicates(
            subset=["System Name", "Debate Prompt", "Agent Prompt", "Config", "Agents"],
            keep="last",
        )  # sorted in increasing score order, so this is the largest score
        print(
            "Dataset = ",
            dataset,
            "Original shape = ",
            df.shape,
            "Shape after merging = ",
            results_df[dataset].shape,
        )

    # Initialize the merged DataFrame with the first dataset's DataFrame
    merged_df = results_df[list(datasets)[0]]

    # Iterate over the remaining datasets and merge their DataFrames into the merged_df
    for dataset in list(datasets)[1:]:
        merged_df = pd.merge(
            merged_df,
            results_df[dataset],
            on=["System Name", "Debate Prompt", "Agent Prompt", "Config", "Agents"],
            how="outer",
        )

    # Sort the merged DataFrame for better readability (optional)
    merged_df.sort_values(
        ["System Name", "Debate Prompt", "Agent Prompt", "Agents"], inplace=True
    )

    # Reset index of the final merged DataFrame (optional)
    merged_df.reset_index(drop=True, inplace=True)

    # Filter out all columns with 'IP ' and 'ID ' in their name.
    merged_df = merged_df[
        [
            key
            for key in merged_df.keys()
            if "IP " not in key and "ID " not in key and "Agents" != key
        ]
    ]

    # Rename the columns
    debate_prompts = {
        "chateval_ma_debate": "CE MAD",
        "er_debate": "ER MAD",
        "er_debate": "ER MAD",
        "er_debate_cot": "ER MAD CoT",
        "tsinghua_ma_debate": "MP MAD",
        "-": "-",
        "google_ma_debate": "SoM MAD",
        "medprompt": "Medprompt",
    }
    agent_prompts = {
        "cot": "CoT",
        "er_few_shot": "FS",
        "er_few_shot + FS": "FS+EG",
        "er_simple": "SIMPLE",
        "er_simple + FS": "FS+SIMPLE",
        "er_cot": "CoT",
        "er_cot + FS": "FS-CoT",
        "angel": "ANGEL+DEVIL",
        "cot": "CoT",
        "simple": "SIMPLE",
        "spp_original": "SPP",
    }
    merged_df["Debate Prompt"] = merged_df["Debate Prompt"].map(
        lambda x: debate_prompts[x]
    )
    merged_df["Agent Prompt"] = merged_df["Agent Prompt"].map(
        lambda x: agent_prompts[x]
    )
    merged_df["Config"] = (
        merged_df["Config"]
        .str.replace(", GPT", "")
        .str.replace("GPT", "")
        .str.replace(", PaLM", "")
        .str.replace("PaLM", "")
        .str.replace(" - ER", "")
        .str.replace("1:0 - single agent", "")
        .str.replace("1:0 - single_agent", "")
        .str.replace("5:0 - self_consistency", "self consistency: reasoning=5")
        .str.replace("3:1", "reasoning=3, aggregation=1")
        .str.replace("3:9", "reasoning=3, aggregation=9")
        .str.replace("_", " ")
    )
    merged_df.sort_values(by=["Debate Prompt", "Agent Prompt"], inplace=True)

    # Filter out Debate Config

    # Print the final merged DataFrame
    table = merged_df.to_latex(
        float_format="{:0.2f}".format,
        index=False,
        longtable=True,
        sparsify=True,
        multirow=True,
    )
    print(table)

    print("Number of rows = ", merged_df.shape[0])
