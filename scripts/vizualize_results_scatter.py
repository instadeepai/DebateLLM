# Copyright 2023  All rights reserved.
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

from visualise_utils import get_dataset_runs, get_paper_dataset_ranges, get_scatter_plot
import pandas as pd

DATASET = "MedQA"  # MedQA, USMLE, PubMedQA, MMLU, CosmosQA, Ciar, GPQA, Chess
# RUN_RANGE = [3904, 4058]

# Hardcode the ranges based on the names
RUN_RANGE = get_paper_dataset_ranges(DATASET.lower())

# METRIC can be "Average seconds per question", "Total cost", "Average tokens per question"
METRICS = ["Total cost", "Average tokens per question", "Average seconds per question"]

for METRIC in METRICS:
    print(f"Plotting {METRIC}")

    # save the figure
    SAVE_PATH = f"./data/charts/{DATASET}_{METRIC}_scatter_plots.pdf"  # None if you don't want to

    LEGEND = True  # whether to show the legend or not
    
    run_table = None

    for run_range in RUN_RANGE:
        if run_table is None:
            run_table = get_dataset_runs(run_range, DATASET.lower())
        else:
            run_table = pd.concat([run_table, get_dataset_runs(run_range, DATASET.lower())], ignore_index=True)

    print(f"Number of runs: {len(run_table)}")

    get_scatter_plot(METRIC, run_table, LEGEND, DATASET, save_path=SAVE_PATH)
