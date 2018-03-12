#!/bin/bash

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-8-0; then
    # The 16.04 installer works with 16.10.
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    apt-get update
    apt-get install cuda-8-0 -y
fi

echo "Installing python3.6..."
add-apt-repository ppa:jonathonf/python-3.6 -y
apt-get update -y
apt-get install python3.6 python3.6-dev python-apt -y
ln -s `which python3.6` /bin/python
echo "Installing pip..."
wget https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py
ln -s `which pip` /bin/pip
