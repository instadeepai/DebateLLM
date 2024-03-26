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
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from debatellm.utils.debate import (
    construct_message,
    construct_summary_message,
    most_frequent,
)


def print_agent_contexts(messages: List[Dict[str, Any]]) -> None:
    for message in messages:
        if message["role"] in ["system", "user"]:
            print(message["content"])
        else:
            assert message["role"] == "assistant"
            print(message["name"] + ": " + message["content"])


class QASystem(abc.ABC):
    def __init__(self, verbose: bool) -> None:
        self._verbose = verbose

    @abc.abstractmethod
    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        pass

    def metrics(
        self, info: Dict[str, Any], _: Callable[..., Any], __: str
    ) -> Dict[str, Any]:
        return info


def construct_agent_metrics(
    info: Dict[str, Any],
    format_solution_fn: Callable,
    solution: str,
    verbose: bool,
    agents: List[str],
    agent_names: List[str],
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Agent Statistics generated:
     - "answered_correctly" # bool: Agent's final answer was correct
     - "first_correct_round_when_correct" # int: Round which agent gave a first
                         correct answer (else NaN)
     - "num_of_correct_rounds_when_correct" # int: Number of rounds agent is correct (else NaN)
     - "percentage_of_correct_rounds_when_correct"   # float: Percent of rounds agent is
                         correct (else NaN)
     - "changed_answer" # bool: whether the agent changed their answer during debate
     - "number_of_answers" # int: how many different answers did the agent give
     - "relied_on_other" # bool: was the agent initially incorrect (in round 0),
                         but gave the correct answer in the final round
     - "bullied_by_other" # bool: was the agent initially correct (in round 0),
                         but gave an incorrect answer in the final round
     - "agent_name" # str: name of agent
     - "agent_engine # str: engine used in the QA Agent
     - "incorrectly_parsed_answer" # bool: an answer not properly parsed during debate
     - "avg_response_length" # float: average length of agent responses during debate
     - "avg_messages_removed" # float: average number of messages removed due to
                        exceeding context length
     - "incorrect_parsed_rounds" # dict: {round: bool} whether a round was incorrectly parsed
     - "response_length_rounds # dict: {round: int} length of agent response per round
     - "time_per_question" # float: total time taken to answer the question for the agent

     Args:
         info (Dict[str, Any]): question answer info dictionary
         format_solution_fn (Callable): function to check a given answer is correct
         solution (str): correct answer to the question
         verbose (bool): whether to print the agent metrics
         agents: List[str]: list of agent keys
         agent_names List[str]: list of agent names
         num_rounds int: number of rounds in the the QA system

     Returns:
         info (Dict[str, Any]): question answer info dictionary including agent metrics
    """

    agent_responses = info["response"]
    agent_answers = info["agent_answers"]
    api_info = info["agent_info"]

    if verbose:
        print(f"\nCORRECT ANSWER: {solution}\nAGENT ANSWER: {agent_answers} \n")

    metrics: Dict = {}
    round_metrics: Dict = {}

    # Per Agent Metrics
    for agent, agent_name in zip(agents, agent_names):

        unparsed_answers: Dict = {}
        response_length: Dict = {}
        prompt_tokens: Dict = {}
        response_tokens: Dict = {}
        round_costs: Dict = {}
        round_durations: Dict = {}
        num_messages_removed: Dict = {}
        rounds = []
        for round_name in list(agent_answers[agent].keys()):
            round_answer = agent_answers[agent][round_name]
            rounds.append(format_solution_fn(round_answer, solution))
            unparsed_answers[round_name] = round_answer == "-1"

            round_response = agent_responses[agent][round_name]
            response_length[round_name] = len(round_response)
            prompt_tokens[round_name] = api_info[agent][round_name]["prompt_tokens"]
            response_tokens[round_name] = api_info[agent][round_name]["response_tokens"]
            round_costs[round_name] = api_info[agent][round_name]["cost"]
            num_messages_removed[round_name] = api_info[agent][round_name][
                "num_messages_removed"
            ]
            round_durations[round_name] = api_info[agent][round_name]["answer_duration"]

        metrics[agent] = {
            # Agent specific metrics
            "agent_name": agent_name,  # str
            "agent_engine": api_info[agent][round_name]["engine"],  # str
            "answered_correctly": rounds[-1],  # bool
            "any_incorrectly_parsed_answer": bool(
                np.any(list(unparsed_answers.values()))
            ),  # bool
            "incorrectly_parsed_final_answer": round_answer == "-1",  # bool
            "avg_response_length": np.mean(list(response_length.values())),  # float
            "avg_round_cost": np.mean(list(round_costs.values())),  # float
            "avg_prompt_tokens": np.mean(list(prompt_tokens.values())),  # float
            "avg_messages_removed": np.mean(
                list(num_messages_removed.values())
            ),  # float
            "avg_response_tokens": np.mean(list(response_tokens.values())),  # float
            "total_prompt_tokens": float(np.sum(list(prompt_tokens.values()))),  # float
            "total_response_tokens": float(
                np.sum(list(response_tokens.values()))
            ),  # float
            "cost_per_question": np.sum(list(round_costs.values())),  # float
            "time_per_question": np.sum(list(round_durations.values())),  # float
        }

        # For debates with more than one round
        if num_rounds > 1:
            # which round was the agent correct in
            first_correct_round = next(
                (i for i, correct in enumerate(rounds) if correct), None
            )
            # Exclude if any round is incorrectly parsed
            if metrics[agent]["any_incorrectly_parsed_answer"]:
                relied_on_other = np.NaN
                bullied_answer = np.NaN
            else:
                # Did the agent change their answer after being incorrect
                if not rounds[0] and isinstance(first_correct_round, int):
                    relied_on_other = True
                else:
                    relied_on_other = False

                # Did the agent change their answer after being correct
                if not rounds[-1] and isinstance(first_correct_round, int):
                    bullied_answer = True
                else:
                    bullied_answer = False

            number_of_answers = len(set(agent_answers[agent].values()))

            metrics[agent]["changed_answer"] = number_of_answers > 1  # bool
            metrics[agent]["number_of_answers"] = number_of_answers  # int
            metrics[agent]["relied_on_other"] = relied_on_other  # bool
            metrics[agent]["bullied_by_other"] = bullied_answer  # bool
            metrics[agent]["first_correct_round_when_correct"] = (
                first_correct_round if isinstance(first_correct_round, int) else np.NaN
            )  # int
            metrics[agent]["num_of_correct_rounds_when_correct"] = (
                int(np.sum(rounds)) if np.sum(rounds) > 0 else np.NaN
            )  # int
            metrics[agent]["percentage_of_correct_rounds_when_correct"] = (
                float(np.mean(rounds)) if np.mean(rounds) > 0 else np.NaN
            )  # float

            round_metrics[agent] = {
                "incorrect_parsed_rounds": unparsed_answers,  # dict
                "response_length_rounds": response_length,  # dict
                "round_costs": round_costs,  # dict,
                "num_messages_removed": num_messages_removed,  # dict,
                "prompt_tokens": prompt_tokens,  # dict,
                "response_tokens": response_tokens,  # dict,
                "round_durations": round_durations,  # dict,
            }

    info["agent_metrics"] = metrics

    if num_rounds > 1:
        info["round_metrics"] = round_metrics

    return info


def construct_debate_metrics(
    info: Dict[str, Any],
    format_solution_fn: Callable,
    solution: str,
    verbose: bool,
    agents: List[str],
    agent_names: List[str],
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Agent Statistics generated:
    - "answered_correctly" # bool: Agent's final answer was correct
    - "first_correct_round_when_correct" # int: Round which agent gave a first
                        correct answer (else NaN)
    - "num_of_correct_rounds_when_correct" # int: Number of rounds agent is correct (else NaN)
    - "percentage_of_correct_rounds_when_correct"   # float: Percent of rounds agent is
                        correct (else NaN)
    - "changed_answer" # bool: whether the agent changed their answer during debate
    - "number_of_answers" # int: how many different answers did the agent give
    - "relied_on_other" # bool: was the agent initially incorrect (in round 0),
                        but gave the correct answer in the final round
    - "bullied_by_other" # bool: was the agent initially correct (in round 0),
                        but gave an incorrect answer in the final round
    - "agent_name" # str: name of agent
    - "agent_engine # str: engine used in the QA Agent
    - "incorrectly_parsed_answer" # bool: an answer not properly parsed during debate
    - "avg_response_length" # float: average length of agent responses during debate
    - "incorrect_parsed_rounds" # dict: {round: bool} whether a round was incorrectly parsed
    - "response_length_rounds # dict: {round: int} length of agent response per round
    - "time_per_question" # float: total time taken to answer the question for the agent

    Debate Statistics generated:
    - "any_correct_answer" # bool: whether any debate agent answer correct during the debate
    - "agents_in_agreement" # bool: do the agents agree in the final round of debate
    - "agents_in_agreement_no_neg_one" # bool: do the agents agree in the final round of debate
    (excluding -1)
    - "how_many_agents_changed" # int: how many agents changed their answer during the debate
    - "how_many_agents_changed_no_neg_one" # int: how many agents changed their answer
     during the debate (excluding -1)
    - "unique_first_answers" # int: how many unique first round answers are there.
    - "unique_first_answers_no_neg_one" # int: how many unique first round answers are there
    (excluding -1).

    Args:
        info (Dict[str, Any]): question answer info dictionary
        format_solution_fn (Callable): function to check a given answer is correct
        solution (str): correct answer to the question
        verbose (bool): whether to print the agent metrics
        agents (List[str]): list of agent keys
        agent_names (List[str]): list of agent names
        num_rounds (int): number of rounds in the debate
    Returns:
        info (Dict[str, Any]): question answer info dictionary including metrics
    """

    # First contruct the agent metrics
    if "agent_metrics" not in info.keys():
        info = construct_agent_metrics(
            info, format_solution_fn, solution, verbose, agents, agent_names, num_rounds
        )

    agent_answers = info["agent_answers"]

    if verbose:
        print(f"\nCORRECT ANSWER: {solution}\nAGENT ANSWERS: {agent_answers} \n")

    debate_metrics: Dict = {}
    any_correct_answer = 0

    for agent in agents:
        rounds = [
            format_solution_fn(round_answer, solution)
            for round_answer in agent_answers[agent].values()
        ]

        # was any agent ever correct
        if "judge" not in agent.lower():
            any_correct_answer += any(rounds)

    # Debate level statistics
    debate_metrics["num_rounds"] = max(
        [int(k.split("_")[1]) for k in agent_answers[agent].keys()]
    )  # int
    debate_metrics["any_correct_answer"] = bool(any_correct_answer)  # bool

    # Metrics that include -1
    debate_metrics["agents_in_agreement"] = (
        len(
            {
                list(agent_answers[agent].values())[-1]
                for agent in agent_answers.keys()
                if "judge" not in agent.lower()
            }
        )
        == 1
    )  # bool
    debate_metrics["how_many_agents_changed"] = sum(
        [
            len(set(agent_answers[agent].values())) > 1
            for agent in agent_answers.keys()
            if "judge" not in agent.lower()
        ]
    )  # int

    debate_metrics["unique_first_answers"] = len(
        {
            list(agent_answers[agent].values())[0]
            for agent in agent_answers.keys()
            if "judge" not in agent.lower()
        }
    )  # int

    # Metrics that exclude -1
    last_answers = {
        list(agent_answers[agent].values())[-1]
        for agent in agent_answers.keys()
        if "judge" not in agent.lower()
    }
    last_answers.discard("-1")
    debate_metrics["agents_in_agreement_no_neg_one"] = len(last_answers) == 1  # bool

    agent_changed_count = 0
    for agent in agent_answers.keys():
        if "judge" not in agent.lower():
            agent_set = set(agent_answers[agent].values())
            agent_set.discard("-1")
            if len(agent_set) > 1:
                agent_changed_count += 1
    debate_metrics["how_many_agents_changed_no_neg_one"] = agent_changed_count  # int

    first_answers = {
        list(agent_answers[agent].values())[0]
        for agent in agent_answers.keys()
        if "judge" not in agent.lower()
    }
    first_answers.discard("-1")
    debate_metrics["unique_first_answers_no_neg_one"] = len(first_answers)  # int

    info["debate_metrics"] = debate_metrics
    return info


class SingleAgentQA(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: Optional[dict] = None,  # Unused
        verbose: bool = False,
        name: Optional[str] = None,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)
        self._agent = agents[0]
        self.agent_name = type(self._agent).__name__
        # self._system_message = debate_prompts["system_message"]

    """
    This implements a simple single Agent QA system.

    The system uses one agent and one round. The agent is given
    the question and it alone generates a single response that is taken
    as the answer.

    Returns:
        answer: str: single char answer
        info: Dict[str, Any]: API info dictionary
    """

    # Setup single agent metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_agent_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=["Agent_0"],
            agent_names=[self.agent_name],
            num_rounds=1,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        if self._verbose:
            print("---- QUESTON ----")
            print(question)
            print("-----------------------------------")
            print("")

        answer, info = self._agent.answer(question=question)

        # Return the answer and the API info consistent with the other QA systems.
        return answer, {
            "response": {"Agent_0": {"Round_0": info["response"]}},
            "agent_answers": {"Agent_0": {"Round_0": answer}},
            "agent_info": {"Agent_0": {"Round_0": info}},
        }


class RoundRobinDebateQA(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: dict,
        num_rounds: int,
        verbose: bool = False,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)
        self._agents = agents
        self._agent_names = [type(agent).__name__ for agent in agents]
        self._num_agents = len(agents)
        self._num_rounds = num_rounds
        self._system_message = debate_prompts["system_message"]

    """
    This is an implementation of a custom round robin debate system.

    The system is a multi-round debate system where each agent is given
    the question and responses generated by all agents in the previous
    round. The agent is then prompted to provide an updated explanation
    and answer. The final answer is determined by taking the most frequent
    answer provided by the agents in the final round of debate.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_debate_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=[f"Agent_{agent}" for agent in range(self._num_agents)],
            agent_names=self._agent_names,
            num_rounds=self._num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        agent_answers: Any = {f"Agent_{agent}": {} for agent in range(self._num_agents)}
        agent_info: Any = {f"Agent_{agent}": {} for agent in range(self._num_agents)}
        agent_responses: Any = {
            f"Agent_{agent}": {} for agent in range(self._num_agents)
        }

        if self._verbose:
            print("---- SYSTEM MESSAGE ----")
            print(self._system_message)
            print("-----------------------------------")
            print("")

            print("---- QUESTON ----")
            print(question)
            print("-----------------------------------")
            print("")

        message_history: List[Dict[str, str]] = []
        for round_index in range(self._num_rounds):
            if self._verbose:
                print("#######################")
                print(f"DEBATING ROUND {round_index}")
                print("#######################")
                print("")

            for i in range(self._num_agents):

                if self._verbose:
                    print(f"---- AGENT {i} ----")

                answer, info = self._agents[i].answer(
                    question=question,
                    system_message=self._system_message,
                    message_history=message_history,
                )

                agent_answers[f"Agent_{i}"][f"Round_{round_index}"] = answer
                agent_responses[f"Agent_{i}"][f"Round_{round_index}"] = info["response"]
                message_history.append(
                    {"agent_name": f"Agent_{i}", "content": info["response"]}
                )
                agent_info[f"Agent_{i}"][f"Round_{round_index}"] = info

        final_answers = [
            agent_answers[f"Agent_{agent}"][f"Round_{self._num_rounds - 1}"]
            for agent in range(self._num_agents)
        ]

        # TODO: End the debate early if all agents agree on the answer
        answer, _ = most_frequent(final_answers)

        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }


class MultiAgentDebateGoogle(QASystem):
    def __init__(
        self,
        agents: dict,
        debate_prompts: dict,
        answer_aggregation: str,
        num_rounds: int,
        agreement_intensity: int,
        summarize_answers: bool,
        verbose: bool = False,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)
        self._agents = agents
        self._agent_names = [type(agent).__name__ for agent in agents]
        self._answer_aggregation = answer_aggregation
        self._num_agents = len(agents)
        self._summarize_answers = summarize_answers
        self._num_rounds = num_rounds

        if agreement_intensity >= 0:
            debate_prompts["suffix_seperator"] = debate_prompts[
                f"suffix_seperator_{agreement_intensity}"
            ]

        self.prompts = debate_prompts

    """
    This is an implementation of the multi-agent debate system take
    from https://arxiv.org/pdf/2305.14325.pdf.

    The system is a multi-round debate system where each agent is given
    the question and responses generated by all agents in the previous
    round. The agent is then prompted to provide an updated explanation
    and answer. The final answer is determined by aggregating the answers
    provided in the final round of debate. The default aggregation method
    is to take the most frequent answer.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_debate_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=[f"Agent_{agent}" for agent in range(self._num_agents)],
            agent_names=self._agent_names,
            num_rounds=self._num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        agent_contexts = [[question] for _ in range(self._num_agents)]
        agent_answers: Any = {f"Agent_{agent}": {} for agent in range(self._num_agents)}
        agent_info: Any = {f"Agent_{agent}": {} for agent in range(self._num_agents)}
        agent_responses: Any = {
            f"Agent_{agent}": {} for agent in range(self._num_agents)
        }

        summary = ""
        for round_index in range(self._num_rounds):
            if self._verbose:
                print("#######################")
                print(f"DEBATING ROUND {round_index}")
                print("#######################")
                print("")

            for i, agent_context in enumerate(agent_contexts):
                if round_index != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]

                    if self._summarize_answers:
                        # Summarize the agent responses for the next round
                        message = construct_summary_message(summary, self.prompts)
                        agent_context[-1] = agent_context[0] + "\n\n" + message

                    else:
                        message = construct_message(
                            agent_contexts_other, question, self.prompts
                        )
                        agent_context[-1] += "\n\n" + message

                if self._verbose:
                    print(f"\n---- AGENT {i} ----")

                answer, info = self._agents[i].answer(prompt=agent_context[-1])
                response = info["response"]
                agent_answers[f"Agent_{i}"][f"Round_{round_index}"] = answer
                agent_responses[f"Agent_{i}"][f"Round_{round_index}"] = response
                agent_info[f"Agent_{i}"][f"Round_{round_index}"] = info
                agent_context.append(agent_context[0] + "\n\n" + response)

            if self._summarize_answers and round_index < self._num_rounds - 1:
                # Summarize the agent responses for the next round

                message = construct_message(
                    agent_contexts,
                    question,
                    self.prompts,
                    summary_mode=True,
                )

                summary = self._agents[0].answer(prompt=message)[1]["response"]

        if self._answer_aggregation == "most_frequent":
            final_answers = [
                agent_answers[f"Agent_{agent}"][f"Round_{self._num_rounds - 1}"]
                for agent in range(self._num_agents)
            ]

            answer, _ = most_frequent(final_answers)
        else:
            raise NotImplementedError(
                "Only most_frequent answer aggregation is currently supported."
            )
        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_contexts": agent_contexts,
            "agent_info": agent_info,
        }


class EnsembleRefinementDebate(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: dict,
        num_reasoning_steps: int = 1,
        num_aggregation_steps: int = 3,
        verbose: bool = False,
        name: Optional[str] = None,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)

        assert len(agents) == 1
        self._agent = agents[0]
        self._num_reasoning_steps = num_reasoning_steps
        assert self._num_reasoning_steps > 0
        self._num_aggregation_steps = num_aggregation_steps
        self._agent_names = [type(agent).__name__ for agent in agents]
        self.prompts = debate_prompts

    """
    This is an implementation of the ensemble refinement debate system take
    from https://arxiv.org/pdf/2305.09617.pdf.

    The system is comprised of a single agent prompted to provide multiple
    answers and explainations in reasoning steps via temperature sampling.
    This is followed by multiple aggregation steps. The reasoning steps are
    used to generate a set of potential answers to the question which are then
    concatenated and the agent is prompted to aggregate the answers during
    aggregation steps. The final answer is determined by taking the
    most frequent answer provided by the agent during the aggregation.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        pseudo_num_rounds = self._num_reasoning_steps + self._num_aggregation_steps
        return construct_agent_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=["Agent_0"],
            agent_names=self._agent_names,
            num_rounds=pseudo_num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:

        agent_answers: Any = {"Agent_0": {}}
        agent_info: Any = {"Agent_0": {}}
        agent_responses: Any = {"Agent_0": {}}
        if self._verbose:
            print("#######################")
            print("REASONING STEP")
            print("#######################")

        message_history: List[Dict[str, str]] = []
        for i in range(self._num_reasoning_steps):
            answer, info = self._agent.answer(
                question=question, system_message=self.prompts["reasoning_step_message"]
            )
            message_history.append(
                {"agent_name": f"Reasoning_{i}", "content": info["response"]}
            )
            agent_answers["Agent_0"][f"Reasoning_{i}"] = answer
            agent_responses["Agent_0"][f"Reasoning_{i}"] = info["response"]
            agent_info["Agent_0"][f"Reasoning_{i}"] = info

        if self._verbose:
            print("#######################")
            print("AGGREGATION STEP")
            print("#######################")
            print("")

        # Decrease the temperature for less stochastic aggregation steps
        temp_decrease = (
            0.2
            if self._agent._sampling.temperature > 0.2
            else self._agent._sampling.temperature
        )
        self._agent._sampling.temperature -= temp_decrease

        for i in range(self._num_aggregation_steps):
            answer, info = self._agent.answer(
                question=question,
                system_message=self.prompts["aggregation_step_message"],
                allow_examples=False,
                message_history=message_history,
            )
            agent_answers["Agent_0"][f"Aggregation_{i}"] = answer
            agent_responses["Agent_0"][f"Aggregation_{i}"] = info["response"]
            agent_info["Agent_0"][f"Aggregation_{i}"] = info

        # Increase the temperature back to the default
        self._agent._sampling.temperature += temp_decrease

        if self._num_aggregation_steps > 0:
            final_answers = [
                agent_answers["Agent_0"][f"Aggregation_{i}"]
                for i in range(self._num_aggregation_steps)
            ]
        else:
            final_answers = [
                agent_answers["Agent_0"][f"Reasoning_{i}"]
                for i in range(self._num_reasoning_steps)
            ]
        answer, _ = most_frequent(final_answers)

        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }


class MultiAgentDebateTsinghua(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: dict,
        judge_name: str,
        max_num_rounds: int,
        agreement_intensity: int = 1,
        verbose: bool = False,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)

        self._judge_name = judge_name
        self._agent_roles = []
        agent_list = []
        for agent in agents:
            if agent._prompt.agent_role != judge_name:
                if (
                    agent._prompt.agent_role == "Negative side"
                    and agreement_intensity >= 0
                ):
                    agent._prompt["context"] = agent._prompt[
                        f"context_{agreement_intensity}"
                    ]
                self._agent_roles.append(agent._prompt.agent_role)
                agent_list.append(agent)
            else:
                self.judge = agent

        self._agents = dict(zip(self._agent_roles, agent_list))
        self._agent_names = [type(agent).__name__ for agent in self._agents.values()]

        self._max_num_rounds = max_num_rounds
        self._agent_system_message = debate_prompts["agent_system_message"]
        self._judge_system_message = debate_prompts["judge_system_message"]

    """
    This is an implementation of Tsinghua Multi-Agent Debate take
    from https://arxiv.org/pdf/2305.19118.pdf.


    The system is a multi-round debate system where each agent is given the
    question and responses generated by all agents. For each round, a judge
    analyzes the responses provided determines whether to terminate the
    debate or keep going. At the end of the debate the judge is also responsible
    for determining the final answer.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_debate_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=self._agent_roles + [self._judge_name],
            agent_names=self._agent_names + [self._judge_name],
            num_rounds=self._max_num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        agent_answers: Any = {
            agent_role: {} for agent_role in self._agent_roles + [self._judge_name]
        }
        agent_info: Any = {
            agent_role: {} for agent_role in self._agent_roles + [self._judge_name]
        }
        agent_responses: Any = {
            agent_role: {} for agent_role in self._agent_roles + [self._judge_name]
        }

        if self._verbose:
            print("---- AGENT SYSTEM MESSAGE ----")
            print(self._agent_system_message)
            print("-----------------------------------")
            print("")

            print("---- JUDGE SYSTEM MESSAGE ----")
            print(self._judge_system_message)
            print("-----------------------------------")
            print("")

            print("---- QUESTON ----")
            print(question)
            print("-----------------------------------")
            print("")

        message_history: List[Dict[str, str]] = []
        for round_index in range(self._max_num_rounds):
            if self._verbose:
                print("#######################")
                print(f"DEBATING ROUND {round_index}")
                print("#######################")
                print("")

            for agent_role in self._agent_roles:

                if self._verbose:
                    print(f"---- {agent_role} ----")

                if not message_history:
                    system_message = ""
                else:
                    system_message = self._agent_system_message

                answer, info = self._agents[agent_role].answer(
                    question=question,
                    system_message=system_message,
                    message_history=message_history,
                )

                agent_answers[agent_role][f"Round_{round_index}"] = answer
                agent_responses[agent_role][f"Round_{round_index}"] = info["response"]
                message_history.append(
                    {"agent_name": agent_role, "content": info["response"]}
                )
                agent_info[agent_role][f"Round_{round_index}"] = info

            # Judge decides whether to continue the debate
            if self._verbose:
                print(f"---- {self._judge_name} ----")

            judge_answer, info = self.judge.answer(
                question=question,
                system_message=self._judge_system_message,
                message_history=message_history,
                instruct_prompt="universal_mode",
            )

            # If the judge decides to end the debate, then break
            if judge_answer:
                break
            else:
                message_history.append(
                    {"agent_name": self._judge_name, "content": info["response"]}
                )
        # debate was inconclusive, so hard stop it with a final prompt
        if not judge_answer:
            # remove last unconclusive answer
            message_history.pop(-1)
            judge_answer, info = self.judge.answer(
                question=question,
                system_message=self._judge_system_message,
                message_history=message_history,
                instruct_prompt="final_mode",
            )

        # Log the judge's answer
        agent_answers[self._judge_name][f"Round_{round_index}"] = judge_answer
        agent_responses[self._judge_name][f"Round_{round_index}"] = info["response"]
        agent_info[self._judge_name][f"Round_{round_index}"] = info

        return judge_answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }


class DebateWithJudge(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: dict,
        judge_name: str,
        num_rounds: int,
        verbose: bool = False,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)

        self._agent_roles = [
            agent._prompt.agent_role if "agent_role" in agent._prompt else f"Agent_{i}"
            for i, agent in enumerate(agents)
        ]
        self._agent_names = [type(agent).__name__ for agent in agents]
        self._judge_name = judge_name
        self._agents = dict(zip(self._agent_roles, agents))
        self._num_rounds = num_rounds
        self._system_message = debate_prompts["system_message"]

    """
    This is an implementation of a multi-agent debate setup with a designated
    judge agent.

    The system is a multi-round debate system where each agent is given the
    question and responses generated by all agents and the judge from the
    previous round. For each round, a judge analyzes the responses provided
    and guides the debate. The judge is also responsible for determining the
    final answer.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_debate_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=self._agent_roles,
            agent_names=self._agent_names,
            num_rounds=self._num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:
        agent_answers: Any = {agent_role: {} for agent_role in self._agent_roles}
        agent_info: Any = {agent_role: {} for agent_role in self._agent_roles}
        agent_responses: Any = {agent_role: {} for agent_role in self._agent_roles}

        if self._verbose:
            print("---- SYSTEM MESSAGE ----")
            print(self._system_message)
            print("-----------------------------------")
            print("")

            print("---- QUESTON ----")
            print(question)
            print("-----------------------------------")
            print("")

        message_history: List[Dict[str, str]] = []
        for round_index in range(self._num_rounds):
            if self._verbose:
                print("#######################")
                print(f"DEBATING ROUND {round_index}")
                print("#######################")
                print("")

            for agent_role in self._agent_roles:

                if self._verbose:
                    print(f"---- {agent_role} ----")

                answer, info = self._agents[agent_role].answer(
                    question=question,
                    system_message=self._system_message,
                    message_history=message_history,
                )

                agent_answers[agent_role][f"Round_{round_index}"] = answer
                agent_responses[agent_role][f"Round_{round_index}"] = info["response"]
                message_history.append(
                    {"agent_name": agent_role, "content": info["response"]}
                )
                agent_info[agent_role][f"Round_{round_index}"] = info

        # Take the answer provided by the judge.
        answer = agent_answers[self._judge_name][f"Round_{self._num_rounds - 1}"]

        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }


class ChatEvalDebate(QASystem):
    def __init__(
        self,
        agents: list,
        debate_prompts: dict,
        debate_setting: str,
        summarizer_name: str,
        num_rounds: int,
        agreement_intensity: int,
        verbose: bool = False,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)

        self._debate_setting = debate_setting
        self._summarizer_name = summarizer_name
        agent_list = []
        for agent in agents:
            if (
                "agent_role" in agent._prompt
                and agent._prompt.agent_role == summarizer_name
            ):
                self.summarizer = agent
            else:
                agent_list.append(agent)

        self._agent_roles = [f"Agent_{i}" for i in range(len(agent_list))]
        self._agents = dict(zip(self._agent_roles, agent_list))
        self._agent_names = [type(agent).__name__ for agent in self._agents.values()]

        self._num_rounds = num_rounds

        if agreement_intensity >= 0:
            self._agent_system_message = debate_prompts[
                f"agent_system_message_{agreement_intensity}"
            ]
        else:
            self._agent_system_message = debate_prompts["agent_system_message"]
        self._summarizer_system_message = debate_prompts["summarizer_system_message"]

    """
    This is an implementation of the three debating protocols in ChatEval take
    from https://arxiv.org/pdf/2308.07201.pdf.


    The system is a multi-round debate system where each agent is given the
    question and responses generated by other agents. There are three available
    debatting settings.
    one_by_one: Each agent is given the question and responses generated by
        all previous agents.
    simultaneous_talk: Each agent is given the question and simultaneously
        generates a response. At the end of each round all responses are
        aggregated and the agents are given the aggregated responses.
    simultaneous_talk_with_summarizer: Each agent is given the question and
        simultaneously generates a response. At the end of each round all
        responses are aggregated and summarized by a summarizer agent. The
        agents are then given the summarized response. The summariser prompt
        is taken from https://shorturl.at/doCWY.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:

        agent_roles = self._agent_roles
        agent_names = self._agent_names
        if self._debate_setting == "simultaneous_talk_with_summarizer":
            agent_roles = self._agent_roles + [self._summarizer_name]
            agent_names = self._agent_names + [self._summarizer_name]

        return construct_debate_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=agent_roles,
            agent_names=agent_names,
            num_rounds=self._num_rounds,
        )

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:

        agent_roles = self._agent_roles
        if self._debate_setting == "simultaneous_talk_with_summarizer":
            agent_roles = self._agent_roles + [self._summarizer_name]

        agent_answers: Any = {agent_role: {} for agent_role in agent_roles}
        agent_info: Any = {agent_role: {} for agent_role in agent_roles}
        agent_responses: Any = {agent_role: {} for agent_role in agent_roles}

        if self._verbose:
            print("---- AGENT SYSTEM MESSAGE ----")
            print(self._agent_system_message)
            print("-----------------------------------")
            print("")

            if self._debate_setting == "simultaneous_talk_with_summarizer":
                print("---- SUMMARIZER SYSTEM MESSAGE ----")
                print(self._summarizer_system_message)
                print("-----------------------------------")
                print("")

            print("---- QUESTON ----")
            print(question)
            print("-----------------------------------")
            print("")

        message_history: List[Dict[str, str]] = []
        for round_index in range(self._num_rounds):
            temp_message_history = message_history.copy()
            if self._verbose:
                print("#######################")
                print(f"DEBATING ROUND {round_index}")
                print("#######################")
                print("")

            for agent_role in self._agent_roles:
                if self._verbose:
                    print(f"---- {agent_role} ----")

                answer, info = self._agents[agent_role].answer(
                    question=question,
                    system_message=self._agent_system_message,
                    message_history=message_history,
                )

                agent_answers[agent_role][f"Round_{round_index}"] = answer
                agent_responses[agent_role][f"Round_{round_index}"] = info["response"]
                agent_info[agent_role][f"Round_{round_index}"] = info

                message = {"agent_name": agent_role, "content": info["response"]}
                if self._debate_setting == "one_by_one":
                    message_history.append(message)
                elif self._debate_setting in [
                    "simultaneous_talk",
                    "simultaneous_talk_with_summarizer",
                ]:
                    temp_message_history.append(message)
                else:
                    raise ValueError(f"Invalid debate setting: {self._debate_setting}")

            if self._debate_setting == "simultaneous_talk":
                message_history = temp_message_history
            elif (
                self._debate_setting == "simultaneous_talk_with_summarizer"
                and round_index < self._num_rounds - 1
            ):
                if self._verbose:
                    print(f"---- {self._summarizer_name} ----")

                answer, info = self.summarizer.answer(
                    question=question,
                    system_message=self._summarizer_system_message,
                    message_history=temp_message_history,
                )

                # Add the summarizer's response to the message history
                message_history = [
                    {"agent_name": self._summarizer_name, "content": info["response"]}
                ]

                # Log the summarizer's answer
                agent_answers[self._summarizer_name][f"Round_{round_index}"] = answer
                agent_responses[self._summarizer_name][f"Round_{round_index}"] = info[
                    "response"
                ]
                agent_info[self._summarizer_name][f"Round_{round_index}"] = info

        # Get the final answer using a mejority vote
        final_answers = [
            agent_answers[agent_role][f"Round_{self._num_rounds - 1}"]
            for agent_role in self._agent_roles
        ]
        answer, _ = most_frequent(final_answers)

        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }


class Medprompt(QASystem):
    def __init__(
        self,
        agents: list,
        num_reasoning_steps: int,
        debate_prompts: dict,
        verbose: bool = False,
        name: Optional[str] = None,
        mock: bool = False,  # Unused
        agent_prompts: Optional[dict] = None,  # Unused
    ):
        super().__init__(verbose=verbose)

        assert len(agents) == 1
        self._num_reasoning_steps = num_reasoning_steps
        self._agent = agents[0]
        self._agent_names = [type(agent).__name__ for agent in agents]
        self.prompts = debate_prompts

    """
    This is an implementation of the Medprompt system take
    from https://arxiv.org/abs/2311.16452

    The system is comprised of a single agent prompted to provide multiple
    answers and explainations via temperature sampling and question shuffling.
    The final answer is determined by taking the most frequent answer provided
    by the agent during the aggregation.

    IMPORTANT: The current implementation only contains the first three steps
    of the Medprompt setup. Therefore additional improvements can be made
    by including the kNN and Ensemble with choice shuffling as well.
    """

    # Setup debate metrics
    def metrics(
        self, info: Dict[str, Any], format_solution_fn: Callable, solution: str
    ) -> Dict[str, Any]:
        return construct_agent_metrics(
            info=info,
            format_solution_fn=format_solution_fn,
            solution=solution,
            verbose=self._verbose,
            agents=["Agent_0"],
            agent_names=self._agent_names,
            num_rounds=self._num_reasoning_steps,
        )

    @staticmethod
    def shuffle_answers(question: str) -> Tuple[str, Any]:
        """
        Takes in a multiple choice question string and shuffles only the answer texts,
        keeping the answer labels (A, B, C, etc.) intact.
        Also returns a mapping of shuffled choices to original choices.
        """
        # Find the start of the answer section (e.g., '\nA:')
        answer_section_start = re.search(r"\n[A-Z]:", question).start()  # type: ignore

        # Split the question from the answers
        main_question = question[:answer_section_start]
        answers = question[answer_section_start + 1 :].split("\n")

        # Filter out answers that are not in the correct format
        # answers = [answer for answer in answers if ": " == answer[1:3]]

        # Extract answer texts
        answer_texts = [answer.split(": ", 1)[1] for answer in answers]

        # assert len(answer_texts) > 0

        # Shuffle the answer texts and create a mapping to original answers
        shuffled_texts = answer_texts.copy()
        random.shuffle(shuffled_texts)
        answer_mapping = {
            chr(65 + i): answers[answer_texts.index(text)][0]
            for i, text in enumerate(shuffled_texts)
        }

        # Reassemble the shuffled answers with original labels
        shuffled_answers = [
            f"{chr(65 + i)}: {text}" for i, text in enumerate(shuffled_texts)
        ]

        # Reassemble the question
        shuffled_question = main_question + "\n" + "\n".join(shuffled_answers)
        return shuffled_question, answer_mapping

    def answer(
        self,
        question: str,
    ) -> Tuple[str, Any]:

        agent_answers: Any = {"Agent_0": {}}
        agent_info: Any = {"Agent_0": {}}
        agent_responses: Any = {"Agent_0": {}}
        if self._verbose:
            print("#######################")
            print("REASONING STEP")
            print("#######################")

        message_history: List[Dict[str, str]] = []

        for i in range(self._num_reasoning_steps):

            try:
                # TODO: Provide the options to the system as well. This would
                # make it much easier to shuffle the answers. Furthermore, remove
                # all questions without options in load_datasets.py.
                shuffled_question, answer_mapping = self.shuffle_answers(question)
            except Exception as e:
                shuffled_question = question
                answer_mapping = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
                print("question: ", question)
                print("Shuffling failed, using original question: ", e)

            answer, info = self._agent.answer(
                question=shuffled_question,
                system_message=self.prompts["system"],
            )

            # Dummy data to check the suffler.
            # answer = "A"
            # info = {"prompt_tokens": 1234, "response_tokens": 1234,
            #       "response": "I don't know, A.",
            #       "cost": 0.0, "num_messages_removed": 0.0,
            #       "answer_duration": 1.0, "engine": "Diesel"}

            # Map the answer back to the original answer
            if answer in answer_mapping:
                answer = answer_mapping[answer]

            message_history.append(
                {"agent_name": f"Reasoning_{i}", "content": info["response"]}
            )
            agent_answers["Agent_0"][f"Reasoning_{i}"] = answer
            agent_responses["Agent_0"][f"Reasoning_{i}"] = info["response"]
            agent_info["Agent_0"][f"Reasoning_{i}"] = info

        final_answers = [
            agent_answers["Agent_0"][f"Reasoning_{i}"]
            for i in range(self._num_reasoning_steps)
        ]
        answer, _ = most_frequent(final_answers)

        return answer, {
            "response": agent_responses,
            "agent_answers": agent_answers,
            "agent_info": agent_info,
        }
