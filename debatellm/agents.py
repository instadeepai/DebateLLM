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

import abc
import datetime
import time
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple

import google
import numpy as np
import openai
import vertexai
from vertexai.preview.language_models import (
    ChatModel,
    InputOutputTextPair,
    TextEmbeddingModel,
    TextGenerationModel,
)

from debatellm.utils.debate import (
    construct_message_from_history,
    partial_format,
    remove_spaces_in_name,
)
from debatellm.utils.gcloud import load_gcloud_credentials
from debatellm.utils.openai import load_openai_api_key


# Try except decorator
def try_except_decorator(func: Callable) -> Callable:
    def func_wrapper(*args: Any, **kwargs: Any) -> Callable:
        history_counter = 0
        while True:
            try:
                return func(*args, **kwargs, history_counter=history_counter)
            except (
                google.api_core.exceptions.InternalServerError,
                google.api_core.exceptions.ResourceExhausted,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIError,
                openai.error.Timeout,
                openai.error.APIConnectionError,
                openai.error.InvalidRequestError,
            ) as e:
                print("API error occurred:", str(e), ". Retrying in 1 second...")

                # If the error is not an invalid request error, then wait for 1 second;
                # otherwise, increment the history counter to discard the oldest message.
                if type(e) == openai.error.InvalidRequestError:
                    history_counter += 1
                else:
                    time.sleep(1)

    return func_wrapper


class BaseAgent(abc.ABC):
    def __init__(
        self,
        cost_per_prompt_token: Optional[float],
        cost_per_response_token: Optional[float],
        prompt: Dict[str, Any],
        verbose: bool,
        agent_name: Optional[str],
        engine: Optional[str],
        few_shot_examples: Optional[Dict],
        agent_prompts: Optional[Dict] = None,  # Unused
    ) -> None:
        self._prompt = prompt
        self._verbose = verbose
        self._agent_name = agent_name
        self._engine = engine
        self._cost_per_prompt_token = cost_per_prompt_token
        self._cost_per_response_token = cost_per_response_token
        self._few_shot_examples = few_shot_examples

        # Extract the answer extractor functions from the prompt.
        self._extractor_fns = {
            key.split("_extractor_fn")[0]: getattr(
                import_module(value.rsplit(".", 1)[0]), value.rsplit(".", 1)[1]
            )
            for key, value in self._prompt.items()
            if key.endswith("_extractor_fn")
        }

    @abc.abstractmethod
    def _infer(
        self,
        context: str,
        instruction: str,
        message_history: list,
        history_counter: int = 0,
        examples: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        pass

    def answer(
        self,
        prompt: Optional[str] = None,
        question: str = "",
        system_message: str = "",
        message_history: list = [],  # noqa: B006
        allow_examples: Optional[bool] = True,
        instruct_prompt: str = "context",
    ) -> Tuple[str, Any]:
        """
        This function takes as input either a prompt (the entire model prompt)
        or a question, system message and message_history that then gets formatted
        internally into a prompt.
        """
        self.allow_examples = allow_examples
        start_time = datetime.datetime.now().timestamp()
        if not prompt:
            assert (
                len(question) != 0
            ), "The question cannot be empty if there is no prompt."

            if not message_history:
                instruct_prompt = (
                    "context_first" if "context_first" in self._prompt else "context"
                )

            instruction = partial_format(
                self._prompt[instruct_prompt], question=question
            )
            system_message = partial_format(system_message, question=question)
            # Print the instruction if verbose
            if self._verbose:
                print("\n System message: " + system_message)

        else:
            instruction = prompt
            assert len(question) == 0, "Cannot use question and pure prompt together."
            assert (
                len(system_message) == 0
            ), "Cannot use system message and pure prompt together."
            assert (
                len(message_history) == 0
            ), "Cannot use message history and pure prompt together."

        if self._verbose:
            print("\nInstruction: " + instruction)

        response, usage_info = self._infer(
            context=system_message,
            instruction=instruction,
            message_history=message_history,
        )

        answer = self._extractor_fns[instruct_prompt](response)

        if self._verbose:
            print("\n---- PROPOSED ANSWER ----")
            print(response)
            print("\n------- DETECTED ANSWER ---------")
            print(answer)
            print("-----------------------------------")

        usage_info.update(
            {
                "response": response,
                "agent_class": self._agent_name,
                "engine": self._engine,
                "answer_duration": datetime.datetime.now().timestamp() - start_time,
            }
        )
        return answer, usage_info


class GPT(BaseAgent):
    def __init__(
        self,
        prompt: Dict[str, Any],
        engine: str,
        few_shot_examples: Optional[Dict],
        cost_per_prompt_token: Optional[float] = 0.0015,
        cost_per_response_token: Optional[float] = 0.002,
        sampling: Optional[dict] = None,
        mock: bool = False,
        verbose: bool = False,
        agent_name: str = "GPT",
        prompt_from_history: str = "assistant_list",
        agent_prompts: Optional[Dict] = None,  # Unused
    ) -> None:
        super().__init__(
            prompt=prompt,
            verbose=verbose,
            agent_name=agent_name,
            engine=engine,
            cost_per_prompt_token=cost_per_prompt_token,
            cost_per_response_token=cost_per_response_token,
            few_shot_examples=few_shot_examples,
        )

        cost_per_prompt_token_dict = {
            "gpt-3.5-turbo-1106": 0.001,
            "gpt-3.5-turbo-0613": 0.001,  # TODO: Assume this is the same as gpt-3.5-turbo-1106?
            "gpt-4": 0.03,
            "gpt-4-1106-preview": 0.01,
            "gpt-4-32k": 0.06,
        }

        cost_per_response_token_dict = {
            "gpt-3.5-turbo-1106": 0.002,
            "gpt-3.5-turbo-0613": 0.002,  # TODO: Assume this is the same as gpt-3.5-turbo-1106?
            "gpt-4": 0.06,
            "gpt-4-1106-preview": 0.03,
            "gpt-4-32k": 0.12,
        }

        assert (
            cost_per_prompt_token == cost_per_prompt_token_dict[engine]
        ), f"cost_per_prompt_token must be {cost_per_prompt_token_dict[engine]} for {engine}"
        assert (
            cost_per_response_token == cost_per_response_token_dict[engine]
        ), f"cost_per_response_token must be {cost_per_response_token_dict[engine]} for {engine}"

        self._mock = mock
        if not self._mock:
            openai.api_key = load_openai_api_key()
        self._engine = engine

        if sampling is None:
            # default sampling parameters
            sampling = {"max_tokens": 300}
        self._sampling = sampling
        self._prompt_from_history = prompt_from_history

    @try_except_decorator
    def _infer(
        self,
        context: str,
        instruction: str,
        message_history: Optional[list] = None,
        history_counter: int = 0,
    ) -> Tuple[str, dict]:
        """
        The history counter is used to limit the number of messages used when the prompt
        length exceeds the model's limit.
        """
        if not self._mock:
            messages = []

            if context:
                messages.append({"role": "system", "content": context})

            if self._few_shot_examples and self.allow_examples:
                for i in range(len(self._few_shot_examples["input_text"])):
                    messages.append(
                        {
                            "role": "user",
                            "name": "example",
                            "content": self._few_shot_examples["input_text"][i],
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "name": "example",
                            "content": self._few_shot_examples["output_text"][i],
                        }
                    )

            if message_history:
                agent_role = (
                    self._prompt["agent_role"]
                    if "agent_role" in self._prompt
                    else "assistant"
                )
                prompts_from_history = construct_message_from_history(
                    message_history,
                    agent_role,
                    mode=self._prompt_from_history,
                )
                messages.extend(prompts_from_history)

            # We want the last message to be a user message:
            # if it's already a user message, we append the instruction to it
            # otherwise we create a new user message with the instruction
            if message_history and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + instruction
            else:
                messages.append({"role": "user", "content": instruction})

            # Remove the oldest history_counter number of messages that is not the system message.
            # This is only done if the prompt length exceeds the model's limit.
            if history_counter > 0:
                print(f"Exceeding context length! Popping {history_counter} messages")

            for _ in range(history_counter):
                messages.pop(1)

            assert self._engine in [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-3.5-turbo-0613",
                "gpt-4-1106-preview",
                "gpt-4-32k",
            ]

            response = openai.ChatCompletion.create(
                model=self._engine,
                messages=remove_spaces_in_name(messages),
                **self._sampling,
            )

            prompt_cost = (
                np.ceil(response["usage"]["prompt_tokens"] / 1000)
                * self._cost_per_prompt_token
            )
            response_cost = (
                np.ceil(response["usage"]["completion_tokens"] / 1000)
                * self._cost_per_response_token
            )
            usage_info = {
                "prompt_tokens": int(response["usage"]["prompt_tokens"]),
                "response_tokens": int(response["usage"]["completion_tokens"]),
                "cost": prompt_cost + response_cost,
                "num_messages_removed": history_counter,
            }
            response = response["choices"][0]["message"]["content"]  # type: ignore
        else:
            response = "This is a mock output."
            usage_info = {"prompt_tokens": 0, "response_tokens": 0, "cost": 0}
        return response, usage_info


class PaLM(BaseAgent):
    def __init__(
        self,
        prompt: Dict[str, Any],
        few_shot_examples: Optional[Dict],
        engine: str = "chat-bison@001",
        cost_per_prompt_token: Optional[float] = 0.0005,
        cost_per_response_token: Optional[float] = 0.0005,
        sampling: Optional[dict] = None,
        mock: bool = False,
        verbose: bool = False,
        agent_name: str = "PaLM",
        agent_prompts: Optional[Dict] = None,  # Unused
    ) -> None:
        super().__init__(
            prompt=prompt,
            verbose=verbose,
            agent_name=agent_name,
            engine=engine,
            cost_per_prompt_token=cost_per_prompt_token,
            cost_per_response_token=cost_per_response_token,
            few_shot_examples=few_shot_examples,
        )

        self._agent_name = "PaLM"
        if sampling is None:
            # default sampling parameters
            sampling = {
                "temperature": 0.2,
                "max_output_tokens": 256,
                "top_p": 0.95,
                "top_k": 40,
            }
        self._sampling = sampling

        self._mock = mock
        if not self._mock:
            credentials, json = load_gcloud_credentials()
            vertexai.init(
                project=json["quota_project_id"],
                location="us-central1",
                credentials=credentials,
            )
        self._engine = engine

    @try_except_decorator
    def _infer(
        self,
        context: str,
        instruction: str,
        message_history: list,
        history_counter: int = 0,
    ) -> Tuple[str, dict]:
        if not self._mock:

            user_message = ""
            for message in message_history:
                user_message += (
                    "\n" + message["agent_name"] + " response: " + message["content"]
                )
            user_message += "\n" + instruction

            examples_list = []
            examples_str = ""

            if self._few_shot_examples and self.allow_examples:

                for i in range(len(self._few_shot_examples["input_text"])):
                    examples_list.append(
                        InputOutputTextPair(
                            input_text=self._few_shot_examples["input_text"][i],
                            output_text=self._few_shot_examples["output_text"][i],
                        )
                    )
            elif (
                "example" in self._prompt.keys()
            ):  # Single generic example in agent prompt
                examples_list.append(
                    InputOutputTextPair(
                        input_text=self._prompt["example"]["input_text"],
                        output_text=self._prompt["example"]["output_text"],
                    )
                )

            examples_str += "".join(
                [example.input_text + example.output_text for example in examples_list]
            )
            full_user_message = context + examples_str + user_message

            # TODO: Try using the message history for PaLM.
            # Currently it only supports two agents.
            if self._engine == "chat-bison@001":
                model = ChatModel.from_pretrained(self._engine)
                chat = model.start_chat(
                    context=context,
                    examples=examples_list,
                )
                output_response = chat.send_message(user_message, **self._sampling)

            elif self._engine == "text-bison@001":
                model = TextGenerationModel.from_pretrained(self._engine)
                output_response = model.predict(full_user_message, **self._sampling)

            output = output_response.text

            embedding_model = TextEmbeddingModel.from_pretrained(
                "textembedding-gecko@001"
            )

            prompt_embedding = embedding_model.get_embeddings([full_user_message])[0]
            num_prompt_tokens = prompt_embedding._prediction_response.predictions[0][
                "embeddings"
            ]["statistics"]["token_count"]

            response_embedding = embedding_model.get_embeddings([output])[0]
            num_response_tokens = response_embedding._prediction_response.predictions[
                0
            ]["embeddings"]["statistics"]["token_count"]

            prompt_cost = (
                np.ceil(num_prompt_tokens / 1000) * self._cost_per_prompt_token
            )
            response_cost = (
                np.ceil(num_response_tokens / 1000) * self._cost_per_response_token
            )
            usage_info = {
                "prompt_tokens": num_prompt_tokens,
                "response_tokens": num_response_tokens,
                "cost": prompt_cost + response_cost,
                "num_messages_removed": 0,  # TODO: Add support for message history
            }
        else:
            output = "This is a mock output."
            usage_info = {"promt_tokens": 0, "response_tokens": 0, "cost": 0}
        return output, usage_info
