# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

PORT=8888

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)


VOLUME_SETUP = -v ${PWD}:/home/app/debatellm

DOCKER_BUILD_ARGS = \
	--build-arg NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN} \
	--build-arg GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS} \
	--build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID)

DOCKER_RUN_FLAGS = --rm --privileged -p ${PORT}:${PORT} --network host
DOCKER_IMAGE_NAME = debatellm
DOCKER_CONTAINER_NAME = debatellm_container


run_debate_venv:
	./venv/bin/python ./debatellm/run_eval.py

build_venv:
	python -m venv venv
	venv/bin/pip install -r requirements.txt

.PHONY: docker_build
docker_build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f ./docker/Dockerfile . \
	--build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) \
	-v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_enter
docker_enter:
	sudo docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app -it $(DOCKER_IMAGE_NAME)

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

upload_datasets_to_s3:
	export S3_ENDPOINT="http://storage.googleapis.com" \
	&& python debatellm/utils/upload_datasets_to_s3.py
