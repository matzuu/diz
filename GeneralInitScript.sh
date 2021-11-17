#!/usr/bin/bash
echo "## Starting General Initialization Script"
echo "## Installing Updates"
sudo apt-get update
echo "## Installing C libraries"
sudo apt --assume-yes install g++
sudo apt --assume-yes install gcc
sudo apt --assume-yes install make
chmod +x Start_FL_training_Client.sh
export USE_CUDA=0
export USE_ROCM=0
#sudo apt --assume-yes install python3.8-venv

echo "## Installing Conda"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

echo "## Maybe need to restart shell after conda install?"
echo "## Installing Packages"
conda update -y conda
conda activate
conda create --name venv
conda activate venv
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
conda install -y --file FedAdapt/requirements.txt

echo "## Finished Conda"
#cd FedAdapt
#echo "## Creating python venv"
#python3 -m venv venv
#source venv/bin/activate
#echo "## Installing pip packages"
#pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements.txt