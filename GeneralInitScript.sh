#!/usr/bin/bash
echo "## Starting General Initialization Script"
echo "## Installing Updates"
sudo apt-get update
echo "## Installing Python"
sudo apt --assume-yes install python3.8-venv
cd FedAdapt
echo "## Creating python venv"
python3 -m venv venv
source venv/bin/activate
echo "## Installing pip packages"
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt