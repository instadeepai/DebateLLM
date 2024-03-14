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

import os

from debatellm.utils.s3_io import FileHandler

if __name__ == "__main__":
    s3_endpoint = os.environ.get("S3_ENDPOINT")

    # Create file handler that set up the access to the s3 bucket if needed.
    path = "data/datasets"
    file_handler = FileHandler(
        s3_endpoint=s3_endpoint, bucket="input", path_to_exams=path
    )

    # Read USMLE dataset
    data_path = "usmle/step1.json"
    data_loaded = file_handler.read_json(path=data_path)
    print(f"\nUSMLE questions loaded: {data_loaded[0]}")

    # Read MMLU dataset
    data_path = "mmlu/dev/anatomy_dev.csv"
    df_loaded = file_handler.read_csv(path=data_path)
    print(f"\nMMLU questions loaded: {df_loaded.head()}")

    # Read PubMedQA dataset
    data_path = "pubmedqa/dev.json"
    dict_loaded = file_handler.read_json(path=data_path)
    example_q = "10808977"
    print(f"\nPubMedQA questions loaded: {dict_loaded[example_q]}")

    # Read MedMCQA dataset
    data_path = "medmcqa/dev.json"
    list_loaded = file_handler.read_json_lines(path=data_path)
    print(f"\nMedMCQA questions loaded: {list_loaded[0]}")
