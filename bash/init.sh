echo 'Begin to install python packages...'
nvidia-smi
conda init
source ~/.bashrc
echo "conda activate env-novelai"
conda activate env-novelai 
cd /group/40034/jerryxwli/code/VideoCrafter_Share/
pip install -r requirements.txt