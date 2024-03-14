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
import re
from typing import Any, List, Optional, Tuple


def partial_format(input_str: str, **kwargs: Any) -> str:
    formatted_str = input_str
    for key, value in kwargs.items():
        formatted_str = formatted_str.replace(f"{{{key}}}", str(value))
    return formatted_str


def remove_spaces_in_name(messages: List[dict]) -> List[dict]:
    for message in messages:
        if "name" in message:
            message["name"] = message["name"].replace(" ", "_")
    return messages


def construct_summary_message(summary: str, prompts: dict) -> str:

    # Use introspection in the case in which there are no other agents.
    if not summary:
        return (
            "Can you verify that your answer is correct. Please reiterate your answer."
        )

    prompt = prompts["summary_prefix_seperator"] + summary + prompts["suffix_seperator"]
    return prompt


def construct_message(
    agents: List[List[str]], question: str, prompts: dict, summary_mode: bool = False
) -> str:

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return (
            "Can you verify that your answer is correct. Please reiterate your answer."
        )

    other_agent_responses = ""
    for agent_response in agents:
        agent_response = remove_question(agent_response[-1], question)  # type: ignore
        other_agent_responses += prompts["agent_response"].format(agent_response)

    prompt = prompts["prefix_seperator"] + other_agent_responses

    if summary_mode:
        prompt += prompts["summary_suffix_seperator"]
    else:
        prompt += prompts["suffix_seperator"]

    return prompt


def construct_message_from_history(
    message_history: List[dict],
    agent_name: Optional[str] = "",
    mode: Optional[str] = "assistant_list",
) -> List[dict]:

    if mode == "assistant_list":
        # default mode: history is given as a list of assistant messages
        messages = []
        for message in message_history:
            messages.append(
                {
                    "role": "assistant",
                    "name": message["agent_name"],
                    "content": message["content"],
                }
            )
        return messages

    elif mode == "one_prompt":
        # history is aggregated into a single user prompt
        user_message = ""
        for message in message_history:
            user_message += (
                "\n\n" + message["agent_name"] + " arguing: " + message["content"]
            )
        user_message = user_message.strip("\n")
        return [
            {
                "role": "user",
                "name": "user",
                "content": user_message,
            }
        ]

    elif mode == "tsinghua_judge":
        # combines one_prompt for each round, separated by previous Judge messages
        current_segment = []
        result = []

        for message in message_history:
            if message["agent_name"] != "Judge":
                current_segment.append(message)
            else:
                result += construct_message_from_history(
                    current_segment, mode="one_prompt"
                )
                result.append(
                    {
                        "role": "assistant",
                        "name": agent_name,
                        "content": message["content"],
                    }
                )
                current_segment = []

        result += construct_message_from_history(current_segment, mode="one_prompt")

        return result

    elif mode == "tsinghua_mad":
        # the agent messages are passed as assistant messages
        # and other agent messages are passed as user messages
        messages = []
        user_message = ""

        two_agent_debate = (
            len(
                {
                    message["agent_name"]
                    for message in message_history
                    if message["agent_name"] != "Judge"
                }
            )
            <= 2
        )

        for message in message_history:

            if message["agent_name"] == "Judge":
                pass

            elif message["agent_name"] == agent_name:

                messages.append(
                    {
                        "role": "user",
                        "name": "user",
                        "content": user_message.strip("\n"),
                    }
                )

                messages.append(
                    {
                        "role": "assistant",
                        "name": message["agent_name"],
                        "content": message["content"],
                    }
                )
                user_message = ""
            else:
                prefix = (
                    ""
                    if two_agent_debate
                    else "\n\n" + message["agent_name"] + " arguing: "
                )
                user_message += prefix + message["content"]

        messages.append(
            {
                "role": "user",
                "name": "user",
                "content": user_message.strip("\n"),
            }
        )
        return messages
    else:
        raise ValueError("Invalid mode")


def remove_question(string: str, question: str) -> str:
    pattern = f"(?=({re.escape(question)}))"
    matches = re.findall(pattern, string)

    for match in matches:
        string = string.replace(match, "", 1)

    return string


def most_frequent(list: List[str]) -> Tuple[str, int]:
    counter = 0
    num = list[0]

    for i in list:
        current_frequency = list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter
