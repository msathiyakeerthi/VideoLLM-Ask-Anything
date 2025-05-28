
# VideoChat Setup Guide

This guide will help you set up and run the **Ask Anything Video Chat** environment with all required dependencies.

---

## 📁 Step 1: Create Project Directory

```bash
mkdir VideoChat
cd VideoChat/
```

---

## 🐍 Step 2: Manage Conda Environments

Check existing environments:

```bash
conda env list
```

Deactivate any active environment:

```bash
conda deactivate
```

Create a new environment (replace if already exists):

```bash
conda create -n chatvideo python=3.8.16
```

> If prompted, choose `y` to remove the existing environment.

Activate the environment:

```bash
conda activate chatvideo
```

---

## 📦 Step 3: Clone Repository

```bash
git clone https://github.com/OpenGVLab/Ask-Anything.git
cd Ask-Anything/video_chat_with_ChatGPT/
```

---

## 🔧 Step 4: Install Dependencies

### PyTorch with CUDA 12.1:

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### Requirements:

Create or update `requirements.txt` with the following content:

```text
openai==0.27.10
einops
gradio==3.38.0
imageio==2.25.1
decord
simplet5
imageio-ffmpeg
Pillow
scipy
numpy
openmim
mmcv-full==1.6.1
spacy==3.7
omegaconf
lvis
boto3
jsonschema
entrypoints
nltk
webdataset
accelerate
bitsandbytes
langchain==0.0.101
timm==0.4.12
transformers
fairscale==0.4.4
pycocoevalcap
torch==1.13.1
torchvision==0.14.1
wget==3.2
setuptools==59.5.0
```

Then install:

```bash
pip install -r requirements.txt
```

Re-run the torch install again (if needed):

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 📦 Step 5: Additional Packages

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchmetrics==0.6.0
pip install --upgrade langchain
pip install langchain-community
```

---

## 📁 Step 6: Download Pretrained Models

```bash
mkdir pretrained_models
wget -P ./pretrained_models https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth

cd pretrained_models/
wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
tar -xvf azcopy.tar.gz
cd azcopy_linux_amd64_10.29.1/

./azcopy copy "https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth?...<truncated>..." ../ --recursive
```

Clone and pull summarization model:

```bash
cd ..
git clone https://huggingface.co/mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback ./flan-t5-large-finetuned-openai-summarize_from_feedback
cd flan-t5-large-finetuned-openai-summarize_from_feedback
git lfs pull
cd ../..
```

---

## ⚙️ Step 7: Environment Variables

```bash
mkdir -p /tmp/huggingface_cache
export TRANSFORMERS_CACHE=/tmp/huggingface_cache
export MPLCONFIGDIR=/tmp/mpl_cache
export OPENAI_API_KEY="YOUR_API_KEY"
```

---

## 🚀 Step 8: Run the App

Navigate to:

```bash
cd /mnt/itcus2/VideoChat/Ask-Anything/video_chat_with_ChatGPT
```

Start the app:

```bash
python app.py
```

---

## 🖥️ Step 9: Access the App Locally via SSH Tunnel

In your **local terminal**, run:

```bash
ssh -i "C:\Users\itcus\Downloads\itcusnew3_hyperstack.txt" -L 7861:localhost:7860 ubuntu@185.216.22.87
```

Then open in browser:

```
http://localhost:7861/
```
