#!/bin/bash
#SBATCH --job-name=gai_inference  # Create a short name for your job
#SBATCH --output=logs/gai_output_%j.log  # Log output file, saved in the logs directory (%j is the job ID)
#SBATCH --error=logs/gai_error_%j.log   # Error log file
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=1                 # Number of GPUs per node (only valid for large/normal partitions)
#SBATCH --time=00:20:00         # Total run time limit (HH:MM:SS) (2 hours, sufficient for one inference)
#SBATCH --partition=normal  # Partition (large/normal/cpu) to submit to
#SBATCH --account=mscaisuperpod      # Required only for multiple projects

# Navigate to the project directory (replace with your gai code path)
cd /home/hxiaoap/Cross-Border_AdGen
echo "当前激活的 Conda 环境：$(conda info --envs | grep '*' | awk '{print $1}')"
echo "当前使用的 Python 路径：$(which python)"
echo "当前 Python 版本：$(python --version)"
echo "当前环境已安装的包：$(pip list | grep -E 'transformers|pillow|gradio|torch')"  # 列出关键依赖

# Load environment
module purge                     # Clear inherited environment modules
module load Anaconda3/2023.09-0  # Load the required modules
echo "Initializing and activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gai

# Prepare log directory
mkdir -p logs

# Execute inference command
echo "Starting inference with image input..."
python ./utils/poster_gen.py

if [ $? -eq 0 ]; then
  # Output on success
  echo "Inference task completed."
else
  # Output on failure
  echo "Inference task failed! Check the error log: cat logs/gai_error_"$SLURM_JOB_ID".log"
fi