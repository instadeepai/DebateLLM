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

from debatellm.agents import GPT  # GPT, PaLM

# from debatellm.systems import RoundRobinDebateQA


def evaluate() -> None:
    # For local debugging
    mock = False
    prompt = {
        "context_extractor_fn": "debatellm.utils.eval.extract_last_floating_capital_letter_as_answer",  # noqa
        "context": "",
        "example": {"input_text": "", "output_text": ""},
    }

    # debate_prompts = {"system_message": ""}

    # Select the system to use
    system = GPT(
        prompt=prompt, engine="gpt-3.5-turbo-0613", mock=mock, few_shot_examples={}
    )
    # system = PaLM(prompt=prompt, mock=mock)

    # EXAMPLE: PaLM debating itself
    # palm = PaLM(prompt=prompt, mock=mock)
    # agents_to_debate = [palm] * 2
    # system = RoundRobinDebateQA(
    #     agents=agents_to_debate,
    #     debate_prompts=debate_prompts,
    #     num_rounds=2,
    #     verbose=True,
    #     mock=mock,
    # )

    # EXAMPLE: GPT debating PaLM
    # gpt = GPT(prompt=prompt, engine="gpt-3.5-turbo-0613", mock=mock)
    # palm = PaLM(prompt=prompt, mock=mock)
    # agents_to_debate = [gpt, palm]
    # system = RoundRobinDebateQA(
    #    agents=agents_to_debate,
    #     debate_prompts=debate_prompts,
    #    num_rounds=2,
    #    verbose=True,
    #    mock=mock,
    # )

    # Ask a question
    question = (
        "Question: A 27-year-old woman comes to the office for counseling prior to "
        "conception. She states that a friend recently delivered a newborn with a neural tube "
        "defect and she wants to decrease her risk for having a child with this condition. She "
        "has no history of major medical illness and takes no medications. Physical examination"
        " shows no abnormalities. It is most appropriate to recommend that this patient begin "
        "supplementation with a vitamin that is a cofactor in which of the following processes?"
        " \n\n(A) Biosynthesis of nucleotides \n\n(B) Protein gamma glutamate carboxylation "
        "\n\n(C) Scavenging of free radicals \n\n(D) Transketolation \n\n(E) Triglyceride "
        "lipolysis \n\nWhat is the correct answer (A, B, C, D, or E)?"
    )

    answer, info = system.answer(
        question=question,
    )

    print("---- FINAL ANSWER ----")
    print(answer)
    print("------------------------")
    print("")
    print("---- INFO ----")
    print(info)


if __name__ == "__main__":
    evaluate()
