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

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem


class FileHandler:
    """Wrapper that gets either an s3 endpoint or None. If an s3 endpoint is provided,
    all methods will read and save the file to the indicated s3 bucket,
    otherwise the local disk is used. The goal is to use the same functions when
    the script is run on Ichor or on a local machine.
    """

    def __init__(
        self,
        s3_endpoint: Optional[str],
        bucket: Optional[str],
        path_to_exams: str = "",
        save_question_mode: Optional[bool] = False,
    ) -> None:
        self.s3_endpoint = s3_endpoint
        self.bucket = bucket
        self.save_question_mode = save_question_mode

        # Assert that bucket is in {"input", "output"}. This could change along with
        # Ichor's infrastructure but is for now structured as an input and ouput
        # bucket.
        # WARNING: if "bucket" == "ouput", the files will be stored under a folder
        # named after the experiment whereas saving in the "input" bucket is more
        # straightforward as the path is directly used.

        if s3_endpoint:
            self.s3 = S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})

            if bucket == "input":
                bucket_path = os.environ.get("AICHOR_INPUT_PATH", "./")
            elif bucket == "output":
                bucket_path = os.environ.get("AICHOR_OUTPUT_PATH", "./")
            else:
                raise ValueError("bucket should be in {input, output}")
        else:
            bucket_path = "./"

        if path_to_exams:
            self.bucket_path = os.path.join(bucket_path, path_to_exams)
            path_to_results = path_to_exams.replace("datasets", "results")
            self.results_path = os.path.join(bucket_path, path_to_results)
        else:
            # edge case to load the GOOGLE_APPLICATION_CREDENTIALS
            self.bucket_path = ""

    def make_results_folder(
        self,
        results_folder: str,
    ) -> None:
        """Create and maintain the results folder.

        Args:
            path (str): Path to the results folder.
        """
        path = os.path.join(self.results_path, results_folder)
        self.makedir(path, exist_ok=True)
        self.results_path = path

    def dump_batch_of_question_and_answers(self, outname: str, results: Any) -> None:
        if self.save_question_mode:
            path = os.path.join(self.results_path, outname)

            try:
                if self.s3_endpoint:
                    with self.s3.open(os.path.join(path)) as f:  # type: ignore
                        existing_data = json.load(f)
                else:
                    with open(os.path.join(path)) as f:  # type: ignore
                        existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []

        existing_data.extend(results)
        self.save_json(path, existing_data)

    def makedir(self, path: str, exist_ok: bool) -> None:
        """Wrapper around the python makedirs method.

        Args:
            path (str): Path to the folder to be created.
        """
        if self.s3_endpoint:
            self.s3.mkdir(path, exist_ok=exist_ok)
        else:
            os.makedirs(path, exist_ok=exist_ok)

    # Handling numpy
    def read_numpy(self, path: str) -> np.ndarray:
        """Wrapper around the numpy load method.

        Args:
            path (str): Path to where the array is to be read.

        Returns:
            np.ndarray: Array.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                array = np.load(f, allow_pickle=True)
        else:
            array = np.load(path, allow_pickle=True)
        return array

    def save_numpy(self, path: str, array: np.ndarray) -> None:
        """Wrapper around the numpy save method.

        Args:
            path (str): Path where the array is to be saved.
            array (np.ndarray): Array to be saved.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path), "wb"  # type: ignore
            ) as f:
                f.write(pickle.dumps(array))
        else:
            np.save(path, array)

    # Handling text files
    def read_text(self, path: str) -> List[str]:
        """Wrapper around the python text read method.

        Args:
            path (str): Path to which the text is to be read.

        Returns:
            List[str]: lines read from the text file.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                lines: List[str] = f.readlines()
        else:
            with open(os.path.join(self.bucket_path, path)) as f:  # type: ignore
                lines = f.readlines()

        return lines

    def save_text(self, path: str, lines: List[str]) -> None:
        """Wrapper around the python text save method.

        Args:
            path (str): Path to which the text is to be saved.
            lines (np.ndarray): Lines to be written in the text file.
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path), "w"  # type: ignore
            ) as f:
                for line in lines:
                    f.write(line)
        else:
            with open(os.path.join(self.bucket_path, path), "w") as f:  # type: ignore
                for line in lines:
                    f.write(line)

    # Handling json files
    def read_json(self, path: str) -> Dict:
        """Wrapper around the python json load method

        Args:
            path (str): Path to which the text is to be read.

        Returns:
            Dict: Dictionary written in the json file;
        """
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                data: Dict = json.load(f)
        else:
            with open(os.path.join(self.bucket_path, path)) as f:  # type: ignore
                data = json.load(f)

        return data

    def read_json_lines(self, path: str) -> List[Dict]:
        """Wrapper around the python json load method

        Args:
            path (str): Path to which the text is to be read.

        Returns:
            List[Dict]: List of dictionaries read from the json file;
        """
        data = []
        if self.s3_endpoint:
            with self.s3.open(
                os.path.join(self.bucket_path, path)  # type: ignore
            ) as f:
                for line in f:
                    data.append(json.loads(line))

        else:
            with open(os.path.join(self.bucket_path, path)) as f:  # type: ignore
                for line in f:
                    data.append(json.loads(line))
        return data

    def save_json(self, path: str, data: Dict) -> None:
        """Wrapper around the python json save method

        Args:
            path (str): Path to which the text is to be saved.
            data (Dict): Dictionary to be saved.
        """
        if self.s3_endpoint:
            with self.s3.open(path, "w") as f:  # type: ignore
                json.dump(data, f)
        else:
            with open(path, "w") as f:  # type: ignore
                json.dump(data, f)

    # Handling csv files
    def read_csv(
        self,
        path: str,
        col_names: Optional[List[str]] = None,
        header: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        delimiter: str = ",",
        quotechar: str = '"',
    ) -> pd.DataFrame:
        """Wrapper around the pd read_csv method

        Args:
            path (str): Path to which the text is to be saved.
            header (np.ndarray): Row to be used for the columns of the resulting
                DataFrame. defaults to None.

        Returns:
            pd.DataFrame: pandas Dataframe.
        """
        if self.s3_endpoint:
            df = pd.read_csv(
                os.path.join(self.bucket_path, path),  # type: ignore
                names=col_names,
                header=header,
                usecols=usecols,
                delimiter=delimiter,
                quotechar=quotechar,
                storage_options={"client_kwargs": {"endpoint_url": self.s3_endpoint}},
            )
        else:
            df = pd.read_csv(
                os.path.join(self.bucket_path, path),  # type: ignore
                names=col_names,
                header=header,
                usecols=usecols,
                delimiter=delimiter,
                quotechar=quotechar,
            )
        return df

    def save_csv(
        self, path: str, df: pd.DataFrame, header: bool = True, index: bool = False
    ) -> None:
        """Wrapper around the pd read_csv method

        Args:
            path (str): Path to which the text is to be saved.
            df (pd.DataFrame): DataFrame to be saved.
            header (bool, optional): Whether headers should be saved. Defaults to True.
            index (bool, optional): Whether index should be saved. Defaults to False.
        """

        if self.s3_endpoint:
            df.to_csv(
                os.path.join(self.bucket_path, path),  # type: ignore
                index=index,
                header=header,
                storage_options={"client_kwargs": {"endpoint_url": self.s3_endpoint}},
            )

        else:
            df.to_csv(
                os.path.join(self.bucket_path, path),  # type: ignore
                index=index,
                header=header,
            )

    # Handling os operations
    def listdir(self, path: str) -> List[str]:
        """Wrapper around the listdir command.

        Args:
            path (str): Path to the folder that needs to be inspected

        Returns:
            List[str]: List of filenames in the folder.
        """
        if self.s3_endpoint:
            # Gets the list of paths from root for all files in the folder
            list_files = []
            for file in self.s3.ls(os.path.join(self.bucket_path, path)):  # type: ignore
                list_files.append(file)
            # Trim paths to get only the file names as in os.listdir
            list_files = [file.split(os.path.sep)[-1] for file in list_files]
            return list_files

        else:
            return os.listdir(os.path.join(self.bucket_path, path))
