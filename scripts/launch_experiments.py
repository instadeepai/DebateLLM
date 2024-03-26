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

from scripts.experiments_utils import gen_agent_config, run_experiments

exp_table = []

# Single agent
exp_table.append(
    {
        "system": "single_agent",
        "system.agents": gen_agent_config(
            1, use_gpt=[False, True], prompt=["simple", "cot"]
        ),
    }
)

# Single Agent Ensemble Refinement
exp_table.extend(
    [
        {  # debate prompt: er_debate
            "system": "ensemble_refinement",
            "system.name": "single_agent",
            "system.num_reasoning_steps": 1,
            "system.num_aggregation_steps": 0,
            "system.agents": gen_agent_config(
                1,
                use_gpt=[False, True],
                prompt="er_simple",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
        {  # debate prompt: er_debate_cot
            "system": "ensemble_refinement_er_debate",
            "system.name": "single_agent",
            "system.num_reasoning_steps": 1,
            "system.num_aggregation_steps": 0,
            "system.agents": gen_agent_config(
                1,
                use_gpt=[False, True],
                prompt="er_cot",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
    ]
)

# Self Consistency Ensemble Refinement
exp_table.extend(
    [
        {
            "system": "ensemble_refinement",
            "system.name": "self_consistency",
            "system.num_reasoning_steps": 5,
            "system.num_aggregation_steps": 0,
            "system.agents": gen_agent_config(
                1,
                use_gpt=[False, True],
                prompt="er_simple",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
        {
            "system": "ensemble_refinement_er_debate",
            "system.name": "self_consistency",
            "system.num_reasoning_steps": 5,
            "system.num_aggregation_steps": 0,
            "system.agents": gen_agent_config(
                1,
                use_gpt=[False, True],
                prompt="er_cot",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
    ]
)

# Ensemble Refinement
exp_table.extend(
    [
        {
            "system": "ensemble_refinement",
            "system.name": "ensemble_refinement",
            "system.num_reasoning_steps": 3,
            "system.num_aggregation_steps": [1, 9],
            "system.agents": gen_agent_config(
                1,
                use_gpt=True,
                prompt="er_simple",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
        {
            "system": "ensemble_refinement_er_debate",
            "system.name": "ensemble_refinement",
            "system.num_reasoning_steps": 3,
            "system.num_aggregation_steps": [1, 9],
            "system.agents": gen_agent_config(
                1,
                use_gpt=True,
                prompt="er_cot",
                is_es=True,
                use_few_shot_examples=[False, True],
            ),
        },
    ]
)

# SPP Synergy
exp_table.append(
    {
        "system": "spp_synergy",
        "system.agents": gen_agent_config(1, use_gpt=True, prompt="spp_original"),
    }
)


# Google MAD
# exp_table.append(
#     {
#         "system": "google_mad",
#         "system.summarize_answers": False,
#         "system.agents": gen_agent_config(3, use_gpt=True, prompt="cot"),
#         "system.num_rounds": 2,
#         "system.agreement_intensity": [2, 6],
#     }
# )
exp_table.append(
    {
        "system": "google_mad",
        "system.summarize_answers": [True, False],
        "system.agents": gen_agent_config(
            [2, 3, 4],
            use_gpt=True,
            prompt="cot",
        ),
        "system.num_rounds": [2, 3],
    }
)


# Tsinghua MAD
# few_shot_examples=None
exp_table.append(
    {
        "system": "tsinghua_mad",
        "system.max_num_rounds": [2, 3, 4],
        # "system.agreement_intensity": [-1, 3, 6, 8],
    }
)
# exp_table.append(
#     {
#         "system": "tsinghua_mad",
#         "system.max_num_rounds": 3,  # [2, 3, 4],
#         "system.agreement_intensity": [-1, 3, 6, 8],
#     }
# )

# ChatEval
exp_table.append(
    {
        "system": "chateval",
        "system.debate_setting": [
            "one_by_one",
            "simultaneous_talk",
            "simultaneous_talk_with_summarizer",
        ],
        "system.num_rounds": [2, 3],
    }
)

# Medprompt
exp_table.append(
    {
        "system": "medprompt",
        "system.num_reasoning_steps": 5,
        "system.agents": gen_agent_config(
            1,
            use_gpt=True,
            sampling={
                "temperature": [0.5, 0.7],
                "top_p": [0.5, 0.8],
            },
        ),
    }
)

# Add all 3 datasets to the experiments
for exp in exp_table:
    exp["dataset"] = [
        # "cosmosqa",
        # "ciar",
        # "gpqa",
        "medqa",
        "pubmedqa",
        "mmlu",
    ]  # "medqa", "pubmedqa", "mmlu", "cosmosqa", "ciar", "gpqa", medmcqa
run_experiments(exp_table, parallel_workers=4, shuffle=False, sort_by_dataset=False)
