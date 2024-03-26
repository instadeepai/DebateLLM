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

import base64
import itertools
import logging
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def generate_combinations(settings):
    keys = settings.keys()
    values = (
        settings[key] if isinstance(settings[key], list) else [settings[key]]
        for key in keys
    )
    combinations = [
        dict(zip(keys, combination)) for combination in itertools.product(*values)
    ]
    return combinations


def encode_config(config):
    return base64.b32encode(str(config).encode()).decode().replace("=", "-")


def gen_agent_config(
    num_agents=1,
    use_gpt=True,
    prompt="cot",
    is_es=False,
    use_few_shot_examples=None,
    sampling=None,
):

    settings = {
        "num_agents": num_agents,
        "use_gpt": use_gpt,
        "prompt": prompt,
        "use_few_shot_examples": use_few_shot_examples,
    }

    if sampling is not None:
        settings["sampling_temperature"] = sampling["temperature"]
        settings["sampling_top_p"] = sampling["top_p"]

    # Generate all combinations of the settings
    exps = generate_combinations(settings)

    encodings = []
    for exp in exps:
        prompt = "${system.agent_prompts." + exp["prompt"] + "}"
        agent = "${system.gpt}" if exp["use_gpt"] else "${system.palm}"
        base_agent = [agent, {"prompt": prompt}]

        if is_es:
            base_agent.append({"sampling": {"temperature": 0.7, "top_p": 0.5}})

            if exp["use_few_shot_examples"]:
                if "cot" in prompt:
                    base_agent.append(
                        {"few_shot_examples": "${system.medpalm_examples.cot_few_shot}"}
                    )
                else:
                    base_agent.append(
                        {"few_shot_examples": "${system.medpalm_examples.few_shot}"}
                    )
            else:
                base_agent.append({"few_shot_examples": False})
        elif sampling is not None:
            base_agent.append(
                {
                    "sampling": {
                        "temperature": exp["sampling_temperature"],
                        "top_p": exp["sampling_top_p"],
                    }
                }
            )
        encodings.append(encode_config([base_agent for _ in range(exp["num_agents"])]))
    return encodings


# Shared flag to indicate if any experiment has failed.
experiment_failed = False
experiment_failed_lock = threading.Lock()


def run_experiment(exp):
    global experiment_failed
    if experiment_failed:
        return

    try:
        logging.info(f"Starting experiment with config: {exp}")
        cmd = ["python", "experiments/evaluate.py"]
        for k, v in exp.items():
            cmd.append(f"{k}={v}")
        subprocess.run(cmd, check=True)
        logging.info(f"Completed experiment with config: {exp}")
    except subprocess.CalledProcessError:
        logging.error(f"Experiment failed with config {exp}")
        with experiment_failed_lock:
            experiment_failed = True
    except Exception as e:
        logging.error(f"Error in experiment with config {exp}: {e}")
        with experiment_failed_lock:
            experiment_failed = True


def run_experiments(
    exp_table, parallel_workers=2, verbose=True, shuffle=False, sort_by_dataset=False
):
    global experiment_failed
    experiment_failed = False  # Reset the flag before starting a new batch.

    experiments = [
        item for setting in exp_table for item in generate_combinations(setting)
    ]

    if shuffle:
        random.shuffle(experiments)
    elif sort_by_dataset:
        experiments = sorted(experiments, key=lambda x: x["dataset"])

    if verbose:
        print(f"Launching {len(experiments)} experiments...")

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        list(tqdm(executor.map(run_experiment, experiments), total=len(experiments)))

        if experiment_failed:
            executor.shutdown(wait=False)
