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
from typing import Any, Dict, Tuple

from google.oauth2.credentials import Credentials

from debatellm.utils.s3_io import FileHandler


def load_gcloud_credentials(
    path: str = "application_default_credentials.json",
) -> Tuple[Any, Dict[Any, Any]]:
    # Check if GOOGLE_APPLICATION_CREDENTIALS is set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        warning_msg = (
            f"{path} not found. Please specify a valid path or set the"
            " GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
        assert os.path.exists(path), warning_msg
    else:
        path = str(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    # Load credentials using a FileHandler with S3 endpoint
    fh = FileHandler(s3_endpoint=os.environ.get("S3_ENDPOINT"), bucket="input")
    data = fh.read_json(path)

    # Create credentials object from json data
    cls = Credentials(None)
    credentials = cls.from_authorized_user_info(data, None)
    return credentials, data
