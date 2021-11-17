#!/usr/bin/bash

echo "## Installing Packages for Conda"
conda update -y conda
conda activate
conda create --name venv
conda activate venv
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
conda install -y --file FedAdapt/requirements.txt