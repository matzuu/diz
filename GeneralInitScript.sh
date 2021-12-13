#!/usr/bin/bash
echo "## Starting General Initialization Script"
echo "## Installing Updates"
sudo apt-get update
echo "## Installing C libraries"
sudo apt --assume-yes install g++
sudo apt --assume-yes install gcc
sudo apt --assume-yes install make
sudo apt --assume-yes install python3-pip
export USE_CUDA=0
export USE_ROCM=0

##Anything else?


