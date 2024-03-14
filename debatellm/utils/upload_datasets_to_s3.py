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

from s3fs.core import S3FileSystem

AICHOR_INPUT_PATH = "s3://debatellm-de65283dde19416d-inputs/data/datasets/"


def upload_data(local_filepath: str, remote_filename: str) -> None:
    s3_endpoint = os.environ.get("S3_ENDPOINT")
    s3 = S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})
    remote_filepath = os.path.join(AICHOR_INPUT_PATH, remote_filename)
    s3.put(local_filepath, remote_filepath, recursive=True)


print(f"\nUploading datasets to: {AICHOR_INPUT_PATH}\n")

# # Using input path for both input and output since it is easier to track.
for dataset_name in ["usmle", "medmcqa", "pubmedqa", "mmlu"]:
    local_location = f"data/datasets/{dataset_name}"
    print(f"Uploading {local_location} to S3...{dataset_name}")
    upload_data(local_location, dataset_name)
