git clone https://github.com/tencent/HunyuanVideo-I2V
cd HunyuanVideo-I2V
# 1. Create conda environment
conda create -n HunyuanVideo-I2V python==3.11.9

# 2. Activate the environment
conda activate HunyuanVideo-I2V

# 3. Install PyTorch and other dependencies using conda
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.0

cd HunyuanVideo-I2V

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --prompt "一个25 岁女性，身材纤细，自然淡妆，表情松弛自信模特穿着图片中的衣服。模特站在极简明亮的摄影棚内，左侧柔和自然光打光，8K 高清，焦点清晰对准服装，阴影柔和自然，专业时尚大片风格，全身构图，纯白色背景（突出服装主体），无任何多余元素，要求人物脸部完整，身体躯干完整。" \
    --i2v-mode \
    --i2v-image-path example/input/skirt.png \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results

