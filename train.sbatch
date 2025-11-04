#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

module purge
module load Anaconda3/2023.09-0

source activate ms_sd_infer

echo "当前激活的 Conda 环境：$(conda info --envs | grep '*' | awk '{print $1}')"
echo "当前使用的 Python 路径：$(which python)"
echo "当前 Python 版本：$(python --version)"
echo "当前环境已安装的包：$(pip list | grep -E 'transformers|pillow|gradio|torch')"  # 列出关键依赖

# python utils/poster_generate.py
python utils/slogans_generate.py 
# nvidia-smi