PROJ_NAME=sisr
CONTAINER_NAME="${PROJ_NAME}-${USER}"

ifndef DS_VOLUME
	DS_VOLUME=/scratch
endif

ifndef NB_PORT
	NB_PORT=8888
endif

ifndef MLF_PORT
	MLF_PORT=5000
endif

ifndef SHM_SIZE
	SHM_SIZE="8g"
endif

PROJ_NAME=sisr
CONTAINER_NAME="${PROJ_NAME}-${USER}"

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"
	@echo "notebookshell -- launches a notebook server"
	@echo "mlflow -- launches an mlflow server"

build:
	docker build -t sisr .

download:
	chmod 775 ./download.sh
	./download.sh

setup:
	chmod 775 ./models/car/adaptive_gridsampler/setup.sh
	./models/car/adaptive_gridsampler/setup.sh

dockershell:
	docker run --rm --name $(CONTAINER_NAME) --gpus all -p 9198:9198 \
	-v $(shell pwd):/sisr -v $(DS_VOLUME):/scratch -v \
	-it sisr \
	--shm-size ${SHM_SIZE}

notebookshell:
	docker run --gpus all --privileged -itd --name $(CONTAINER_NAME)-nb \
	-p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/sisr -v $(DS_VOLUME):/scratch \
	sisr \
	jupyter notebook \
	--NotebookApp.token='sisr' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT} \
	--shm-size ${SHM_SIZE}

mlflow:
	docker run --privileged -itd --rm --name $(CONTAINER_NAME)-mlf \
	-p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/sisr -v $(DS_VOLUME):/scratch \
	sisr \
	mlflow ui --host 0.0.0.0:${MLF_PORT}
