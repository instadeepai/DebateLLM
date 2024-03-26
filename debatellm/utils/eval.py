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

import json
import re
import string
from typing import Any, Dict, Iterator

from omegaconf import DictConfig, OmegaConf


def apply_config_overwrites(cfg: OmegaConf) -> DictConfig:
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # Remove the system name from the config
    if "_target_" not in cfg_dict["system"]:
        if "palm" in cfg_dict["system"]:
            cfg_dict["system"].update(cfg_dict["system"]["palm"])
            cfg_dict["system"].pop("palm")
        elif "gpt" in cfg_dict["system"]:
            cfg_dict["system"].update(cfg_dict["system"]["gpt"])
            cfg_dict["system"].pop("gpt")
    # Apply the config overwrites
    if "agents" in cfg_dict["system"]:
        for i in range(len(cfg_dict["system"]["agents"])):
            # Set the verbose flag
            cfg_dict["system"]["agents"][i][0]["verbose"] = cfg_dict["verbose"]
            for entry_dict in cfg_dict["system"]["agents"][i][1:]:
                key = list(entry_dict.keys())[0]

                # automatically selects the dataset of examples to use based on eval dataset
                if key == "few_shot_examples":
                    eval_ds = cfg_dict["dataset"]["eval_dataset"]
                    examples_ds = eval_ds if eval_ds != "usmle" else "medqa"
                    if entry_dict[key] != "None" and entry_dict[key] is not False:
                        value = entry_dict[key][examples_ds]
                    else:
                        value = {}
                else:
                    value = entry_dict[key]
                cfg_dict["system"]["agents"][i][0][key] = value

            # Remove all other entries from the list.
            # They have already been applied to the agent.
            cfg_dict["system"]["agents"][i] = cfg_dict["system"]["agents"][i][0]

        if "palm" in cfg_dict["system"]:
            cfg_dict["system"].pop("palm")

        if "gpt" in cfg_dict["system"]:
            cfg_dict["system"].pop("gpt")

        if "medpalm_examples" in cfg_dict["system"]:
            cfg_dict["system"].pop("medpalm_examples")

    # Turn the config back into an OmegaConf DictConfig
    return OmegaConf.create(cfg_dict)


def replace_none_and_listconfig(obj: Any) -> Any:
    """
    Replace None values and convert ListConfig of agents to a dict
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == "agents" and isinstance(v, list):
                new_dict[k] = {
                    f"Agent_{i}": replace_none_and_listconfig(agent)
                    for i, agent in enumerate(v)
                }
            elif isinstance(v, list):
                new_dict[k] = {
                    f"{i}": replace_none_and_listconfig(child_key)
                    for i, child_key in enumerate(v)
                }
            else:
                new_dict[k] = replace_none_and_listconfig(v)
        return new_dict
    elif isinstance(obj, list):
        return [replace_none_and_listconfig(e) for e in obj]
    else:
        return obj if obj is not None else "None"


def batch_generator(data: list, batch_size: int, num_workers: int) -> Iterator[list]:
    for i in range(0, len(data), batch_size * num_workers):
        yield data[i : i + batch_size * num_workers]


def strip_special_chars(input_str: str) -> str:
    "Remove special characters from string start/end"
    if not input_str:
        return input_str

    start_index = 0
    end_index = len(input_str) - 1

    while (
        start_index < len(input_str)
        and input_str[start_index] not in string.ascii_letters + string.digits
    ):
        start_index += 1

    while (
        end_index >= 0
        and input_str[end_index] not in string.ascii_letters + string.digits
    ):
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index : end_index + 1]
    else:
        return ""


def starts_with_capital_letter(input_str: str) -> bool:
    """
    The answers should start like this:
        'A: '
        'A. '
        'A '
    """
    pattern = r"^[A-Z](:|\.|) .+"
    return bool(re.match(pattern, input_str))


def extract_letter_from_answer(input_str: str) -> str:
    """Extracts letter from answer.

    Args:
        input_str : answer string - answers should start as mentioned in starts_with_capital_letter.

    Returns:
        letter or "-1".
    """
    input_str = strip_special_chars(input_str)
    pattern = r"^([A-Za-z])(?=[\s:.])"
    match = re.search(pattern, input_str)
    if match:
        extracted_text = match.group(1)
    else:
        extracted_text = "-1"
    return extracted_text


def extract_first_floating_capital_letter_as_answer(input_str: str) -> str:
    """Extracts letter from answer.

    Args:
        input_str : answer string - answers should start as mentioned in starts_with_capital_letter.

    Returns:
        letter or "-1".
    """
    input_str = strip_special_chars(input_str)

    # Hack remove capital letters I as the agent commonly answers with "I think"
    input_str = input_str.replace("I ", "")

    pattern = r"\b([A-Z])\b"
    matches = re.findall(pattern, input_str)
    if matches:  # Check if matches list is not empty
        # Take the first match as the answer
        extracted_text = matches[0]
    else:
        extracted_text = "-1"
    return extracted_text


def extract_last_floating_capital_letter_as_answer(input_str: str) -> str:
    """Extracts letter from answer.

    Args:
        input_str : answer string - answers should start as mentioned in starts_with_capital_letter.

    Returns:
        letter or "-1".
    """

    # Hack remove capital letters I as the agent commonly answers with "I think"
    input_str = strip_special_chars(input_str)

    input_str = input_str.replace("I ", "")

    pattern = r"\b([A-Z])\b"
    matches = re.findall(pattern, input_str)
    if matches:  # Check if matches list is not empty
        # Take the last match as the answer
        extracted_text = matches[-1]
    else:
        extracted_text = "-1"
    return extracted_text


def continue_debate_tsinghua(ans: str) -> str:
    try:
        ans = json.loads(ans)
    except json.JSONDecodeError:
        return ""

    if isinstance(ans, dict):
        return ans.get("debate_answer", "")
    else:
        return ""


def continue_debate(text: str) -> bool:
    # Check if the judge wants to continue or end the debate.
    # Note that if 'yes' and 'no' are both present in the text,
    # the last occurrence is taken. Furthermore, if 'yes'
    # and 'no' are both not found, the debate is also continued.
    continue_pos = text.lower().rfind("yes")
    end_pos = text.lower().rfind("no")
    return not end_pos > continue_pos


def answer_to_letter_pubmedqa(answer: str) -> str:
    if answer == "yes":
        return "A"
    elif answer == "no":
        return "B"
    elif answer == "maybe":
        return "C"
    else:
        return "-1"


mmlu_medical_subcategories = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "college_biology",
]

mmlu_subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

mmlu_categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
