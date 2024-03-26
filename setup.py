# python3
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

"""Install script for setuptools."""
import os

import setuptools
from setuptools import setup

build_root = os.path.dirname(__file__)


def requirements() -> list:
    """Get package requirements"""
    with open(os.path.join(build_root, "requirements.txt")) as f:
        return [pname.strip() for pname in f.readlines()]


with open("README.md") as tmp:
    readme = tmp.read()

setup(
    name="id-debatellm",
    version="0.0.0",
    description="A Python library for constitutional LLMs in the medical field.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords="medical conversational LLMs",
    packages=setuptools.find_packages(),
    install_requires=requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
