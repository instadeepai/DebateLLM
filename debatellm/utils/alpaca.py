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

# Sampling strategies
greedy_search = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": False,
}

beam_serach = {
    "num_beams": 4,
    "do_sample": False,
    "max_new_tokens": 128,
    "early_stopping": True,
}

sampling_top_k = {
    "do_sample": True,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_k": 50,
}

sampling_top_p = {
    "do_sample": True,
    "top_k": 0,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

default_sampling = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9,
}
