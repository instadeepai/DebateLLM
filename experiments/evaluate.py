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
import logging
import os

import hydra
import neptune
import omegaconf
from rich.logging import RichHandler

from debatellm.eval.eval_system import eval_system
from debatellm.utils.eval import apply_config_overwrites, replace_none_and_listconfig

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

AICHOR_PATH = os.environ.get("AICHOR_INPUT_PATH")


def display_info(config):
    log.info(f"AIChor Path: {AICHOR_PATH}")
    log.info("Working directory : {}".format(os.getcwd()))
    log.info(f"\nConfig:\n{omegaconf.OmegaConf.to_yaml(config)}")


@hydra.main(
    config_path="conf",
    version_base=None,
    config_name="config",
)
def evaluate(cfg: omegaconf.DictConfig):
    if type(cfg.system.agents) == str:
        # Decode the agents config string from the experiment launcher
        message = base64.b32decode(cfg.system.agents.replace("-", "=")).decode()
        cfg.system.agents = eval(message)

    cfg = apply_config_overwrites(cfg)

    # Uncomment to display the config
    # display_info(cfg)

    system_fn = lambda: hydra.utils.instantiate(cfg.system, verbose=cfg.verbose)
    system_name = cfg.system._target_.split(".")[-1].lower()

    agents = ":"
    agents += "-".join([agent._target_.split(".")[-1] for agent in cfg.system.agents])

    eval_dataset = cfg.dataset.eval_dataset
    if os.getenv("TM_NEPTUNE_API_TOKEN"):
        logger = neptune.init_run(
            project="Anon/debatellm",
            api_token=os.getenv("TM_NEPTUNE_API_TOKEN"),
            tags=[f"{eval_dataset}", f"{system_name}{agents}"],
        )

        logger["config"] = replace_none_and_listconfig(
            omegaconf.OmegaConf.to_container(cfg, resolve=True)
        )
        params = {
            "system_name": system_name,
            "path_to_exams": cfg.dataset.path_to_exams,
            "dataset": eval_dataset,
            "max_eval_count": 0
            if cfg.max_eval_count == "None"
            else int(cfg.max_eval_count),
            "num_eval_workers": cfg.num_eval_workers,
            "batch_size": 1
            if cfg.eval_batch_size == "None"
            else int(cfg.eval_batch_size),
        }
        logger["parameters"] = params
    else:
        logger = None

    eval_system(
        system_fn=system_fn,
        system_name=system_name,
        dataset=eval_dataset,
        path_to_exams=cfg.dataset.path_to_exams,
        dataset_settings=cfg.dataset.dataset_settings,
        max_eval_count=None
        if cfg.max_eval_count == "None"
        else int(cfg.max_eval_count),
        num_eval_workers=cfg.num_eval_workers,
        eval_batch_size=1
        if cfg.eval_batch_size == "None"
        else int(cfg.eval_batch_size),
        logger=logger,
        seed=cfg.seed,
        save_question_answer_mode=cfg.save_question_answer_mode,
    )


if __name__ == "__main__":
    evaluate()
