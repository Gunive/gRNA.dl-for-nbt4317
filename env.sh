# Ubuntu 16.04
sudo su
apt --assume-yes install tmux build-essential gcc g++ make binutils unzip git
	
# install virtualenv
pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./venv

# install Anaconda
wget "https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh"
bash "Anaconda3-5.0.1-Linux-x86_64.sh" -b
	
cd ~
	
echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
conda install -y bcolz
conda upgrade -y --all
	
# install cuda 9.0
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install cuda
apt-get --assume-yes upgrade
apt-get --assume-yes autoremove
apt-get install cuda-toolkit-9.0 cuda-command-line-tools-9-0）

cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
	
source ~/.bashrc
	
# install cudnn libraries
wget http://files.fast.ai/files/cudnn-9.1-linux-x64-v7.tgz
tar xf cudnn-9.1-linux-x64-v7.tgz
sudo cp cuda/include/*.* /usr/local/cuda/include/
sudo cp cuda/lib64/*.* /usr/local/cuda/lib64/

# Add NVIDIA package repositories
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1604/x86_64/cuda-9-0_9.0.176-1_amd64.deb
dpkg -i cuda-9-0_9.0.176-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
wget https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
apt-get update
	
# Install NVIDIA driver
apt-get install --no-install-recommends nvidia-driver-410

# install tensorflow (mirrors in China)
pip install tensorflow-gpu --ignore-installed --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple/

# install keras
pip install keras
