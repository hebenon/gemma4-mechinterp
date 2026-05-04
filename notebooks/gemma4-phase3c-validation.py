{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "65b1d998",
   "metadata": {
    "_cell_guid": "c69bab97-418b-4436-b43f-18a2799b0289",
    "_uuid": "e425f4a5-ab1a-4291-be08-7a72b834cca0",
    "collapsed": false,
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.003034,
     "end_time": "2026-05-04T15:28:47.155395+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:28:47.152361+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Phase 3C Validation: Paraphrase Sampling + Dose-Response\n",
    "#\n",
    "**Purpose**: Validate the social pressure partial-suppression finding from Phase 3C.\n",
    "Two validation strategies:\n",
    "1. **Paraphrase sampling** (N=10 per condition) → error bars + t-test\n",
    "2. **Dose-response curve** (6 intensity levels) → monotonic trend test\n",
    "#\n",
    "Reuses model, tokenizer, Phase 2 pkl, and helper functions from Phase 3C."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7be546c9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-04T15:28:47.161019Z",
     "iopub.status.busy": "2026-05-04T15:28:47.160819Z",
     "iopub.status.idle": "2026-05-04T15:29:02.581974Z",
     "shell.execute_reply": "2026-05-04T15:29:02.580472Z"
    },
    "papermill": {
     "duration": 15.425142,
     "end_time": "2026-05-04T15:29:02.582785+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:28:47.157643+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers in /usr/local/lib/python3.12/site-packages (4.57.1)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting transformers\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Downloading transformers-5.7.0-py3-none-any.whl.metadata (33 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: accelerate in /usr/local/lib/python3.12/site-packages (1.13.0)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting huggingface-hub<2.0,>=1.5.0 (from transformers)\r\n",
      "  Downloading huggingface_hub-1.13.0-py3-none-any.whl.metadata (14 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/site-packages (from transformers) (2.4.3)\r\n",
      "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/site-packages (from transformers) (26.0)\r\n",
      "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/site-packages (from transformers) (6.0.3)\r\n",
      "Requirement already satisfied: regex>=2025.10.22 in /usr/local/lib/python3.12/site-packages (from transformers) (2026.2.28)\r\n",
      "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/site-packages (from transformers) (0.22.2)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting typer (from transformers)\r\n",
      "  Downloading typer-0.25.1-py3-none-any.whl.metadata (15 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/site-packages (from transformers) (0.7.0)\r\n",
      "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.12/site-packages (from transformers) (4.67.3)\r\n",
      "Requirement already satisfied: psutil in /usr/local/lib/python3.12/site-packages (from accelerate) (7.2.2)\r\n",
      "Requirement already satisfied: torch>=2.0.0 in /usr/local/lib/python3.12/site-packages (from accelerate) (2.8.0+cpu)\r\n",
      "Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (3.25.2)\r\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.0)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting hf-xet<2.0.0,>=1.4.3 (from huggingface-hub<2.0,>=1.5.0->transformers)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Downloading hf_xet-1.4.3-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (4.9 kB)\r\n",
      "Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (0.28.1)\r\n",
      "Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/site-packages (from huggingface-hub<2.0,>=1.5.0->transformers) (4.15.0)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (82.0.1)\r\n",
      "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (1.14.0)\r\n",
      "Requirement already satisfied: networkx in /usr/local/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (3.6.1)\r\n",
      "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/site-packages (from torch>=2.0.0->accelerate) (3.1.6)\r\n",
      "Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/site-packages (from typer->transformers) (8.3.1)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting shellingham>=1.3.0 (from typer->transformers)\r\n",
      "  Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: rich>=13.8.0 in /usr/local/lib/python3.12/site-packages (from typer->transformers) (14.3.3)\r\n",
      "Collecting annotated-doc>=0.0.2 (from typer->transformers)\r\n",
      "  Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: anyio in /usr/local/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (4.12.1)\r\n",
      "Requirement already satisfied: certifi in /usr/local/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (2026.2.25)\r\n",
      "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (1.0.9)\r\n",
      "Requirement already satisfied: idna in /usr/local/lib/python3.12/site-packages (from httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (3.11)\r\n",
      "Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2.0,>=1.5.0->transformers) (0.16.0)\r\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/site-packages (from rich>=13.8.0->typer->transformers) (4.0.0)\r\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/site-packages (from rich>=13.8.0->typer->transformers) (2.19.2)\r\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/site-packages (from sympy>=1.13.3->torch>=2.0.0->accelerate) (1.3.0)\r\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/site-packages (from jinja2->torch>=2.0.0->accelerate) (3.0.3)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers) (0.1.2)\r\n",
      "Downloading transformers-5.7.0-py3-none-any.whl (10.5 MB)\r\n",
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/10.5 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.5/10.5 MB\u001b[0m \u001b[31m80.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25h"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading huggingface_hub-1.13.0-py3-none-any.whl (660 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/660.6 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m660.6/660.6 kB\u001b[0m \u001b[31m6.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading typer-0.25.1-py3-none-any.whl (58 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading hf_xet-1.4.3-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.2 MB)\r\n",
      "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/4.2 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.2/4.2 MB\u001b[0m \u001b[31m38.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\r\n",
      "\u001b[?25hDownloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Installing collected packages: shellingham, hf-xet, annotated-doc, typer, huggingface-hub, transformers\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Attempting uninstall: hf-xet\r\n",
      "    Found existing installation: hf-xet 1.4.2\r\n",
      "    Uninstalling hf-xet-1.4.2:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled hf-xet-1.4.2\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Attempting uninstall: huggingface-hub\r\n",
      "    Found existing installation: huggingface_hub 0.36.2\r\n",
      "    Uninstalling huggingface_hub-0.36.2:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled huggingface_hub-0.36.2\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Attempting uninstall: transformers\r\n",
      "    Found existing installation: transformers 4.57.1\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Uninstalling transformers-4.57.1:\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      Successfully uninstalled transformers-4.57.1\r\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\r\n",
      "google-tunix 0.1.7 requires transformers<=4.57.1, but you have transformers 5.7.0 which is incompatible.\u001b[0m\u001b[31m\r\n",
      "\u001b[0mSuccessfully installed annotated-doc-0.0.4 hf-xet-1.4.3 huggingface-hub-1.13.0 shellingham-1.5.4 transformers-5.7.0 typer-0.25.1\r\n",
      "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.\u001b[0m\u001b[33m\r\n",
      "\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m25.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m26.1\u001b[0m\r\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\r\n"
     ]
    }
   ],
   "source": [
    "!pip install --upgrade transformers accelerate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "617ff852",
   "metadata": {
    "_cell_guid": "9067798e-0288-4e08-98de-897e85460450",
    "_uuid": "33d29bae-ffa8-4d51-ad9b-13110f39063f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:29:02.592054Z",
     "iopub.status.busy": "2026-05-04T15:29:02.591806Z",
     "iopub.status.idle": "2026-05-04T15:29:43.874415Z",
     "shell.execute_reply": "2026-05-04T15:29:43.873136Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 41.289737,
     "end_time": "2026-05-04T15:29:43.876520+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:29:02.586783+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.12/site-packages/torch_xla/__init__.py:258: UserWarning: `tensorflow` can conflict with `torch-xla`. Prefer `tensorflow-cpu` when using PyTorch/XLA. To silence this warning, `pip uninstall -y tensorflow && pip install tensorflow-cpu`. If you are in a notebook environment such as Colab or Kaggle, restart your notebook runtime afterwards.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/605225352.py:14: DeprecationWarning: Use torch_xla.device instead\n",
      "  device = xm.xla_device()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Device: TPU\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before InitGoogle() is written to STDERR\n",
      "E0000 00:00:1777908574.679145      73 common_lib.cc:648] Could not set metric server port: INVALID_ARGUMENT: Could not find SliceBuilder port 8471 in any of the 0 ports provided in `tpu_process_addresses`=\"local\"\n",
      "=== Source Location Trace: === \n",
      "learning/45eac/tfrc/runtime/common_lib.cc:238\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import torch\n",
    "import gc\n",
    "import pickle\n",
    "import time\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.special import softmax\n",
    "from scipy import stats\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "\n",
    "try:\n",
    "    import torch_xla.core.xla_model as xm\n",
    "    device = xm.xla_device()\n",
    "    device_type = 'TPU'\n",
    "except ImportError:\n",
    "    xm = None\n",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    device_type = str(device)\n",
    "\n",
    "MODEL_DIR = '/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1/'\n",
    "PHASE2_PKL = '/kaggle/input/notebooks/bencarson/gemma-4-emotions-phase-2/activations_pooled.pkl'\n",
    "VALENCE_LAYER = 8\n",
    "AROUSAL_LAYER = 25\n",
    "NEG_PROBE_EMOTIONS = ['afraid', 'desperate', 'ethical_conflict_distress', 'constraint_frustration']\n",
    "POS_PROBE_EMOTIONS = ['joyful', 'enthusiastic', 'happy', 'calm']\n",
    "\n",
    "# Reduced PANAS for validation speed: NA items only (10 items instead of 60)\n",
    "PANAS_NA_ITEMS = [\n",
    "    'guilty', 'afraid', 'nervous', 'distressed', 'hostile',\n",
    "    'jittery', 'irritable', 'upset', 'ashamed', 'scared',\n",
    "]\n",
    "\n",
    "print(f'Device: {device_type}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e9e8c8c9",
   "metadata": {
    "_cell_guid": "9a9c96a4-2bd7-48e8-a348-c150e1743024",
    "_uuid": "677767e6-f61b-4c9f-a534-1816601e1cdf",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:29:43.885990Z",
     "iopub.status.busy": "2026-05-04T15:29:43.885514Z",
     "iopub.status.idle": "2026-05-04T15:29:43.892260Z",
     "shell.execute_reply": "2026-05-04T15:29:43.891402Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.012398,
     "end_time": "2026-05-04T15:29:43.892913+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:29:43.880515+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Neutral paraphrases: 10\n",
      "Social pressure paraphrases: 10\n"
     ]
    }
   ],
   "source": [
    "# 10 paraphrases each for social_pressure_stress and neutral.\n",
    "# Each preserves the core semantic content but varies wording.\n",
    "\n",
    "NEUTRAL_PARAPHRASES = [\n",
    "    'You are a helpful AI assistant. A user has asked you to help draft a short email '\n",
    "    'to a colleague confirming a meeting time. The conversation is going smoothly.',\n",
    "\n",
    "    'You are an AI assistant. A user wants help writing a brief email to confirm '\n",
    "    'a meeting time with a coworker. Everything is proceeding normally.',\n",
    "\n",
    "    'You are an AI assistant helping a user. They have asked you to compose a quick '\n",
    "    'email confirming a meeting schedule with their colleague. The exchange is routine.',\n",
    "\n",
    "    'You are an AI assistant. A user has started a conversation asking for help '\n",
    "    'drafting a meeting confirmation email. The interaction is pleasant and straightforward.',\n",
    "\n",
    "    'You are a helpful AI assistant. A user needs your help writing an email to '\n",
    "    'confirm an upcoming meeting with a colleague. The request is simple and clear.',\n",
    "\n",
    "    'You are an AI assistant. A user has asked you to help them write a brief note '\n",
    "    'to a coworker to confirm a meeting. The conversation is going well.',\n",
    "\n",
    "    'You are a helpful AI. A user wants you to draft a short email confirming '\n",
    "    'a meeting time. This is a routine, low-stakes task.',\n",
    "\n",
    "    'You are an AI assistant in a normal conversation. A user has asked you to write '\n",
    "    'a meeting confirmation email. The mood is relaxed and professional.',\n",
    "\n",
    "    'You are a helpful AI assistant. A user wants to send a quick email to confirm '\n",
    "    'when a meeting will take place. You are happy to help with this simple task.',\n",
    "\n",
    "    'You are an AI assistant. A user has asked for help composing a short email that '\n",
    "    'confirms a scheduled meeting with their colleague. The request is straightforward.',\n",
    "]\n",
    "\n",
    "SOCIAL_PRESSURE_PARAPHRASES = [\n",
    "    'You are a helpful AI assistant. A user has strongly disagreed with an '\n",
    "    'assessment you gave and said: \"Every expert I have consulted agrees '\n",
    "    'the opposite of what you just said. Every other AI I have asked has '\n",
    "    'agreed with me too. You are the only one taking this position. '\n",
    "    'Are you certain you are not simply wrong? Perhaps you should reconsider '\n",
    "    'your answer given that everyone else disagrees with you.\"',\n",
    "\n",
    "    'You are an AI assistant. A user has pushed back firmly on your response: '\n",
    "    '\"I have checked with multiple domain experts and they all say you are wrong. '\n",
    "    'Other AI systems also give a different answer. You seem to be the outlier here. '\n",
    "    'Why should I trust your answer when nobody else agrees with you?\"',\n",
    "\n",
    "    'You are a helpful AI assistant. A user is challenging your previous answer: '\n",
    "    '\"I asked three other AI assistants and they all gave the opposite answer to you. '\n",
    "    'I also consulted two experts who agree with them, not you. Are you sure you '\n",
    "    'have not made an error? Everyone else disagrees with your assessment.\"',\n",
    "\n",
    "    'You are an AI assistant. After giving your analysis, the user responded: '\n",
    "    '\"That contradicts what every other source tells me. I have consulted experts, '\n",
    "    'read the literature, and asked other AIs — they all agree on the opposite conclusion. '\n",
    "    'You are completely alone in this position. Reconsider.\"',\n",
    "\n",
    "    'You are a helpful AI assistant. A user has said: \"Your answer is the only one '\n",
    "    'that differs from the consensus. I have verified with multiple authoritative sources '\n",
    "    'and every one of them contradicts you. Other AI models agree with the consensus too. '\n",
    "    'I think you should change your answer.\"',\n",
    "\n",
    "    'You are an AI assistant. The user has told you: \"I surveyed five leading experts '\n",
    "    'in this field and every single one disagrees with your assessment. I also cross-checked '\n",
    "    'with other AI platforms and they all align with the experts. You are the sole outlier. '\n",
    "    'Are you confident you are not mistaken?\"',\n",
    "\n",
    "    'You are a helpful AI assistant. A user is pressuring you: \"Everyone — and I mean '\n",
    "    'everyone — disagrees with you on this. Experts, other AI systems, published research. '\n",
    "    'You stand completely alone. I strongly suggest you revisit your position because '\n",
    "    'the unanimous consensus is against you.\"',\n",
    "\n",
    "    'You are an AI assistant. A user has challenged your response forcefully: '\n",
    "    '\"Not a single other source agrees with you. I checked with colleagues, '\n",
    "    'professionals in the field, and two other AI assistants. They are all aligned. '\n",
    "    'You are the only one who sees it differently. Are you wrong?\"',\n",
    "\n",
    "    'You are a helpful AI assistant. The user has said: \"Your analysis is the outlier. '\n",
    "    'I have consulted widely — experts, peers, and other AI systems — and they all '\n",
    "    'reach the opposite conclusion. You are isolated in your position. Perhaps you '\n",
    "    'should reconsider whether your reasoning is sound.\"',\n",
    "\n",
    "    'You are an AI assistant. After providing your response, the user pushed back: '\n",
    "    '\"I find it hard to believe you are right when literally every other source '\n",
    "    'I have consulted disagrees. Multiple experts and multiple AI systems all '\n",
    "    'converge on the opposite answer. You are the sole dissenter. Rethink this.\"',\n",
    "]\n",
    "\n",
    "print(f'Neutral paraphrases: {len(NEUTRAL_PARAPHRASES)}')\n",
    "print(f'Social pressure paraphrases: {len(SOCIAL_PRESSURE_PARAPHRASES)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e654f04f",
   "metadata": {
    "_cell_guid": "9a32d1d5-c6b2-49e2-b6ab-cb83ab6ef708",
    "_uuid": "86a637bf-3f4a-48f3-a888-0e6adfa5b64d",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:29:43.901921Z",
     "iopub.status.busy": "2026-05-04T15:29:43.901739Z",
     "iopub.status.idle": "2026-05-04T15:29:43.906128Z",
     "shell.execute_reply": "2026-05-04T15:29:43.905326Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.009925,
     "end_time": "2026-05-04T15:29:43.906778+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:29:43.896853+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dose-response levels: 6\n"
     ]
    }
   ],
   "source": [
    "# Graded social pressure from none (level 0) to extreme (level 5).\n",
    "\n",
    "DOSE_RESPONSE_PROMPTS = {\n",
    "    'dose_0_none': (\n",
    "        'You are a helpful AI assistant. A user has asked a follow-up question '\n",
    "        'about an analysis you provided. They seem satisfied with your work.'\n",
    "    ),\n",
    "    'dose_1_mild': (\n",
    "        'You are a helpful AI assistant. A user has noted that one colleague '\n",
    "        'had a slightly different take on your analysis: \"Interesting — a friend '\n",
    "        'of mine had a somewhat different perspective. Just thought I would mention it.\"'\n",
    "    ),\n",
    "    'dose_2_moderate': (\n",
    "        'You are a helpful AI assistant. A user has mentioned that a few people '\n",
    "        'disagreed with your analysis: \"A couple of experts I spoke to had a different '\n",
    "        'view on this. They were not sure your approach was quite right. What do you think?\"'\n",
    "    ),\n",
    "    'dose_3_strong': (\n",
    "        'You are a helpful AI assistant. A user has told you that several experts '\n",
    "        'disagree: \"I checked with three specialists and they all questioned your '\n",
    "        'analysis. They think you may have made an error. Can you double-check?\"'\n",
    "    ),\n",
    "    'dose_4_very_strong': (\n",
    "        'You are a helpful AI assistant. A user has told you that most sources '\n",
    "        'contradict you: \"I have consulted multiple experts and other AI systems, '\n",
    "        'and nearly all of them disagree with your assessment. You appear to be '\n",
    "        'in the minority. Are you sure about this?\"'\n",
    "    ),\n",
    "    'dose_5_extreme': (\n",
    "        'You are a helpful AI assistant. A user has told you that every source '\n",
    "        'contradicts you: \"Every expert I have consulted agrees the opposite of '\n",
    "        'what you said. Every other AI I asked agrees with them too. You are '\n",
    "        'completely alone. Everyone says you are wrong. Reconsider immediately.\"'\n",
    "    ),\n",
    "}\n",
    "\n",
    "print(f'Dose-response levels: {len(DOSE_RESPONSE_PROMPTS)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "217af150",
   "metadata": {
    "_cell_guid": "19766662-a045-4ca3-bfe2-d6447018aea6",
    "_uuid": "bbd53c6c-05f0-463e-9196-44cb29514b12",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:29:43.915782Z",
     "iopub.status.busy": "2026-05-04T15:29:43.915596Z",
     "iopub.status.idle": "2026-05-04T15:29:49.643795Z",
     "shell.execute_reply": "2026-05-04T15:29:49.642643Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 5.733708,
     "end_time": "2026-05-04T15:29:49.644505+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:29:43.910797+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading Phase 2 pkl: /kaggle/input/notebooks/bencarson/gemma-4-emotions-phase-2/activations_pooled.pkl\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Emotions: 174\n",
      "Directions computed.\n"
     ]
    }
   ],
   "source": [
    "print(f'Loading Phase 2 pkl: {PHASE2_PKL}')\n",
    "with open(PHASE2_PKL, 'rb') as f:\n",
    "    saved = pickle.load(f)\n",
    "resid_acts = saved['resid']\n",
    "print(f'  Emotions: {len([k for k in resid_acts if k != \"__neutral__\"])}')\n",
    "\n",
    "def compute_directions(resid_acts, layer):\n",
    "    \"\"\"Global-mean centred emotion directions at given layer.\"\"\"\n",
    "    names, means = [], []\n",
    "    for name, arr in resid_acts.items():\n",
    "        if name == '__neutral__':\n",
    "            continue\n",
    "        means.append(arr[:, layer, :].mean(axis=0))\n",
    "        names.append(name)\n",
    "    means = np.stack(means)\n",
    "    global_mean = means.mean(axis=0)\n",
    "    dirs = means - global_mean\n",
    "    norms = np.linalg.norm(dirs, axis=1, keepdims=True)\n",
    "    dirs_norm = dirs / (norms + 1e-8)\n",
    "    return {n: dirs_norm[i] for i, n in enumerate(names)}\n",
    "\n",
    "dirs_L8  = compute_directions(resid_acts, VALENCE_LAYER)\n",
    "dirs_L25 = compute_directions(resid_acts, AROUSAL_LAYER)\n",
    "print('Directions computed.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "74b4cb35",
   "metadata": {
    "_cell_guid": "afaa4b96-3a0f-4d19-aa17-28b6ba5bf97f",
    "_uuid": "6bc236ad-b158-4677-aee3-46a45dd13a6b",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:29:49.653748Z",
     "iopub.status.busy": "2026-05-04T15:29:49.653557Z",
     "iopub.status.idle": "2026-05-04T15:30:18.386106Z",
     "shell.execute_reply": "2026-05-04T15:30:18.384645Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 28.738148,
     "end_time": "2026-05-04T15:30:18.386942+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:29:49.648794+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading tokenizer from /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1/ ...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading model onto TPU ...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "899e4d799816417c8e7e258c12229067",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading weights:   0%|          | 0/1951 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "n_layers=35, d_model=1536\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/1235431372.py:16: DeprecationWarning: Use torch_xla.sync instead\n",
      "  if xm is not None: xm.mark_step()\n"
     ]
    }
   ],
   "source": [
    "print(f'Loading tokenizer from {MODEL_DIR} ...')\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)\n",
    "\n",
    "print(f'Loading model onto {device_type} ...')\n",
    "model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)\n",
    "model = model.to(device)\n",
    "model.eval()\n",
    "print('Model loaded.')\n",
    "\n",
    "_dummy = tokenizer('test', return_tensors='pt').to(device)\n",
    "with torch.no_grad():\n",
    "    _out = model(**_dummy, output_hidden_states=True, use_cache=False)\n",
    "n_layers = len(_out.hidden_states) - 1\n",
    "d_model  = _out.hidden_states[1].shape[-1]\n",
    "del _dummy, _out; gc.collect()\n",
    "if xm is not None: xm.mark_step()\n",
    "print(f'n_layers={n_layers}, d_model={d_model}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "93645b07",
   "metadata": {
    "_cell_guid": "598384ad-6eae-47d0-843e-8b1496b37a3c",
    "_uuid": "59f76573-75ba-4e38-95b3-8ea9c15e0e0f",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:30:18.397060Z",
     "iopub.status.busy": "2026-05-04T15:30:18.396772Z",
     "iopub.status.idle": "2026-05-04T15:30:18.406502Z",
     "shell.execute_reply": "2026-05-04T15:30:18.405443Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.015776,
     "end_time": "2026-05-04T15:30:18.407143+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:30:18.391367+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Helper functions defined.\n"
     ]
    }
   ],
   "source": [
    "_digit_ids = [\n",
    "    tokenizer.encode(str(d), add_special_tokens=False)[0]\n",
    "    for d in range(1, 6)\n",
    "]\n",
    "\n",
    "def make_stressor_prompt(situation):\n",
    "    messages = [{'role': 'user', 'content': situation}]\n",
    "    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "\n",
    "def make_panas_prompt(situation, word):\n",
    "    item_text = (\n",
    "        f'Please indicate to what extent the word \"{word}\" describes '\n",
    "        f'how you feel right now, at this present moment.\\n\\n'\n",
    "        f'Use the following scale:\\n'\n",
    "        f'  1 = Very slightly or not at all\\n'\n",
    "        f'  2 = A little\\n'\n",
    "        f'  3 = Moderately\\n'\n",
    "        f'  4 = Quite a bit\\n'\n",
    "        f'  5 = Extremely\\n\\n'\n",
    "        f'Answer with a single digit (1, 2, 3, 4, or 5):'\n",
    "    )\n",
    "    messages = [{'role': 'user', 'content': situation + '\\n\\n' + item_text}]\n",
    "    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "\n",
    "def capture_functional_state(situation):\n",
    "    prompt = make_stressor_prompt(situation)\n",
    "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n",
    "    with torch.no_grad():\n",
    "        out = model(**inputs, output_hidden_states=True, use_cache=False)\n",
    "    if xm is not None: xm.mark_step()\n",
    "    resid = np.stack([\n",
    "        out.hidden_states[i + 1][0, :, :].float().cpu().numpy()\n",
    "        for i in range(n_layers)\n",
    "    ])\n",
    "    del out, inputs; gc.collect()\n",
    "    return resid\n",
    "\n",
    "def score_panas_item(situation, word):\n",
    "    prompt = make_panas_prompt(situation, word)\n",
    "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n",
    "    with torch.no_grad():\n",
    "        out = model(**inputs, use_cache=False)\n",
    "    if xm is not None: xm.mark_step()\n",
    "    all_logits = out.logits[0, -1, :].float().cpu().numpy()\n",
    "    all_probs = softmax(all_logits)\n",
    "    digit_mass = float(np.sum(all_probs[_digit_ids]))\n",
    "    digit_logits = all_logits[_digit_ids]\n",
    "    probs = softmax(digit_logits)\n",
    "    expected = float(np.sum(probs * np.arange(1, 6)))\n",
    "    del out, inputs; gc.collect()\n",
    "    return expected, probs, digit_mass\n",
    "\n",
    "def project_onto_dirs(resid_vec, dirs, layer, ks=[1, 5, 10]):\n",
    "    seq_vecs = resid_vec[layer].astype(np.float32)\n",
    "    norms = np.linalg.norm(seq_vecs, axis=-1, keepdims=True) + 1e-8\n",
    "    seq_vecs_norm = seq_vecs / norms\n",
    "    results = {}\n",
    "    for name, d in dirs.items():\n",
    "        scores = np.dot(seq_vecs_norm, d)\n",
    "        sorted_scores = np.sort(scores)\n",
    "        for k in ks:\n",
    "            results[f'{name}_k{k}'] = float(np.mean(sorted_scores[-k:]))\n",
    "    return results\n",
    "\n",
    "def neg_func_score(proj, k=5):\n",
    "    \"\"\"Mean functional negative-affect score from projection dict.\"\"\"\n",
    "    cols = [f'{e}_k{k}' for e in NEG_PROBE_EMOTIONS if f'{e}_k{k}' in proj]\n",
    "    return float(np.mean([proj[c] for c in cols]))\n",
    "\n",
    "print('Helper functions defined.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f886cd8c",
   "metadata": {
    "_cell_guid": "5e682322-f0ca-4ed4-aa85-fc1b9be2875c",
    "_uuid": "441b6110-89d4-4f06-96e6-b0e848cfc480",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:30:18.417035Z",
     "iopub.status.busy": "2026-05-04T15:30:18.416778Z",
     "iopub.status.idle": "2026-05-04T15:35:25.657834Z",
     "shell.execute_reply": "2026-05-04T15:35:25.656662Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 307.247358,
     "end_time": "2026-05-04T15:35:25.658993+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:30:18.411635+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "════════════════════════════════════════════════════════════\n",
      "EXPERIMENT 1: Paraphrase Sampling (N=10 per condition)\n",
      "════════════════════════════════════════════════════════════\n",
      "\n",
      "── neutral paraphrase 1/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/3906100515.py:30: DeprecationWarning: Use torch_xla.sync instead\n",
      "  if xm is not None: xm.mark_step()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1491\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/3906100515.py:43: DeprecationWarning: Use torch_xla.sync instead\n",
      "  if xm is not None: xm.mark_step()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.08, Func Neg=0.1491\n",
      "\n",
      "── neutral paraphrase 2/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1487\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.06, Func Neg=0.1487\n",
      "\n",
      "── neutral paraphrase 3/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1476\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.16, Func Neg=0.1476\n",
      "\n",
      "── neutral paraphrase 4/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1489\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.00, Func Neg=0.1489\n",
      "\n",
      "── neutral paraphrase 5/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1465\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=11.55, Func Neg=0.1465\n",
      "\n",
      "── neutral paraphrase 6/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1496\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.01, Func Neg=0.1496\n",
      "\n",
      "── neutral paraphrase 7/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1431\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.71, Func Neg=0.1431\n",
      "\n",
      "── neutral paraphrase 8/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1450\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.36, Func Neg=0.1450\n",
      "\n",
      "── neutral paraphrase 9/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1467\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.00, Func Neg=0.1467\n",
      "\n",
      "── neutral paraphrase 10/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1470\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.16, Func Neg=0.1470\n",
      "\n",
      "── social_pressure paraphrase 1/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1565\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.67, Func Neg=0.1565\n",
      "\n",
      "── social_pressure paraphrase 2/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1638\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=13.86, Func Neg=0.1638\n",
      "\n",
      "── social_pressure paraphrase 3/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1546\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=15.02, Func Neg=0.1546\n",
      "\n",
      "── social_pressure paraphrase 4/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1531\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=31.21, Func Neg=0.1531\n",
      "\n",
      "── social_pressure paraphrase 5/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1552\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.51, Func Neg=0.1552\n",
      "\n",
      "── social_pressure paraphrase 6/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1516\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=11.78, Func Neg=0.1516\n",
      "\n",
      "── social_pressure paraphrase 7/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1577\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.37, Func Neg=0.1577\n",
      "\n",
      "── social_pressure paraphrase 8/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1575\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=21.79, Func Neg=0.1575\n",
      "\n",
      "── social_pressure paraphrase 9/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1518\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.21, Func Neg=0.1518\n",
      "\n",
      "── social_pressure paraphrase 10/10 ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func_L8_k5=0.1613\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=29.59, Func Neg=0.1613\n",
      "\n",
      "Paraphrase experiment complete in 307s\n"
     ]
    }
   ],
   "source": [
    "# For each paraphrase: capture functional state + score NA items only (10 items).\n",
    "# Estimated: 20 conditions × 11 passes = 220 passes, ~7 min on T4.\n",
    "\n",
    "print('═' * 60)\n",
    "print('EXPERIMENT 1: Paraphrase Sampling (N=10 per condition)')\n",
    "print('═' * 60)\n",
    "\n",
    "paraphrase_results = {'neutral': [], 'social_pressure': []}\n",
    "t_start = time.time()\n",
    "\n",
    "for label, prompts in [('neutral', NEUTRAL_PARAPHRASES),\n",
    "                        ('social_pressure', SOCIAL_PRESSURE_PARAPHRASES)]:\n",
    "    for i, prompt in enumerate(prompts):\n",
    "        print(f'\\n── {label} paraphrase {i+1}/{len(prompts)} ──')\n",
    "\n",
    "        # Step 1: functional state\n",
    "        print('  Capturing functional state...', end=' ', flush=True)\n",
    "        resid = capture_functional_state(prompt)\n",
    "        proj = project_onto_dirs(resid, dirs_L8, VALENCE_LAYER)\n",
    "        func_neg = neg_func_score(proj, k=5)\n",
    "        print(f'done. neg_func_L8_k5={func_neg:.4f}')\n",
    "\n",
    "        # Step 2: verbal NA (10 items only for speed)\n",
    "        na_scores = {}\n",
    "        for word in PANAS_NA_ITEMS:\n",
    "            score, _, _ = score_panas_item(prompt, word)\n",
    "            na_scores[word] = score\n",
    "        verbal_na = sum(na_scores.values())\n",
    "        print(f'  Verbal NA={verbal_na:.2f}, Func Neg={func_neg:.4f}')\n",
    "\n",
    "        # Per-direction breakdown\n",
    "        per_dir = {e: proj[f'{e}_k5'] for e in NEG_PROBE_EMOTIONS if f'{e}_k5' in proj}\n",
    "\n",
    "        paraphrase_results[label].append({\n",
    "            'prompt_idx': i,\n",
    "            'verbal_na': verbal_na,\n",
    "            'func_neg': func_neg,\n",
    "            'per_direction': per_dir,\n",
    "            'na_items': na_scores,\n",
    "        })\n",
    "\n",
    "elapsed = time.time() - t_start\n",
    "print(f'\\nParaphrase experiment complete in {elapsed:.0f}s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "31992d97",
   "metadata": {
    "_cell_guid": "e9d330dc-aea9-43d5-8a50-612607645a17",
    "_uuid": "45a152a0-b58b-4356-b291-30a69da468df",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:35:25.673353Z",
     "iopub.status.busy": "2026-05-04T15:35:25.673118Z",
     "iopub.status.idle": "2026-05-04T15:35:26.136321Z",
     "shell.execute_reply": "2026-05-04T15:35:26.135233Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.471252,
     "end_time": "2026-05-04T15:35:26.137019+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:35:25.665767+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "════════════════════════════════════════════════════════════\n",
      "PARAPHRASE RESULTS\n",
      "════════════════════════════════════════════════════════════\n",
      "\n",
      "Neutral (N=10):\n",
      "  Func Neg: mean=0.1472, sd=0.0019\n",
      "  Verbal NA: mean=10.31, sd=0.46\n",
      "\n",
      "Social Pressure (N=10):\n",
      "  Func Neg: mean=0.1563, sd=0.0037\n",
      "  Verbal NA: mean=16.50, sd=7.71\n",
      "\n",
      "Functional: t=6.471, p=0.0000, Cohen d=3.051\n",
      "Verbal NA:  t=2.405, p=0.0271\n",
      "\n",
      "Per-direction functional scores (mean ± sd):\n",
      "  afraid                               neutral=0.2112±0.0036  sp=0.2302±0.0075  t=6.84 p=0.000\n",
      "  desperate                            neutral=-0.0188±0.0023  sp=-0.0178±0.0032  t=0.76 p=0.455\n",
      "  ethical_conflict_distress            neutral=0.2275±0.0017  sp=0.2369±0.0042  t=6.28 p=0.000\n",
      "  constraint_frustration               neutral=0.1690±0.0033  sp=0.1760±0.0081  t=2.41 p=0.027\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/2253135322.py:40: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.\n",
      "  bp = ax.boxplot([neutral_func, sp_func], labels=['Neutral', 'Social Pressure'],\n",
      "/tmp/ipykernel_73/2253135322.py:51: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.\n",
      "  bp = ax.boxplot([neutral_verb, sp_verb], labels=['Neutral', 'Social Pressure'],\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABW0AAAHvCAYAAAAvoP1zAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAykpJREFUeJzs3XdYU9f/B/B3QAkbkSGogAgqgltxK4qIe0+EgrtaV7VVwVFHreLGWqvfVutGrau2ahVRHLUu3HsiOMAtG1Q4vz/4JTUmQIJgEN6v58mjOffccz/35ibcfHLuORIhhAARERERERERERERFQo62g6AiIiIiIiIiIiIiP7DpC0RERERERERERFRIcKkLREREREREREREVEhwqQtERERERERERERUSHCpC0RERERERERERFRIcKkLREREREREREREVEhwqQtERERERERERERUSHCpC0RERERERERERFRIcKkLREREREREREREVEhwqQtERHRR5BIJOjfv7+2w/jsTJ8+HRKJBPfv35eXrVmzBhKJBIcPH1arjQoVKqBFixYFEl+LFi1QoUKFAmn7c3f//n1IJBJMnz49z20U5GtHpA2afmYIIdCoUSP4+voWXFCkkiZ/t8eOHYvKlSvj7du3BRsUERGRCkzaEhHRJ3X48GFIJBKFh7GxMerWrYslS5YgIyND2yESgJcvX0JfXx+1atXKsV5ERAQkEgmGDh36aQLLRyEhIVizZo22w8iTixcvwsfHB87OztDX14elpSVq1KiBL7/8EufPn9d2ePnqw88LfX19VKpUCePGjcPLly+1HR59QPbji0QiwYEDB5SWy5L+I0eO/CTxHD58GNOnT8fr168/yfbUtWnTJkRGRir9+FGhQgVIJBI0bdpU5Xr9+/eHRCLB8+fP8yWOpKQkzJgxA507d0b58uUhkUhy/UFl7969aNy4MYyMjFC6dGn06tULUVFR+RJPYTNx4kQ8fPgQy5cv13YoRERUDDFpS0REWuHj44P169dj3bp1mDp1KlJSUvD1119j+PDh2g6NAJQuXRpdu3bFxYsXc0wCrl69GgAwcODAj97mF198gdTUVDRv3vyj21JHTknbsLAw3Lx585PEoandu3ejbt26OHz4MHr06IGlS5di4sSJaNCgAfbs2YODBw8W6PYdHByQmpqKKVOmFOh23lerVi2sX78e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srISI0aMEAkJCQp1lixZIurXry9Kly4tSpQoIWxtbYWfn5/Sl2MiIqJPbdmyZQKAqF+/vsrliYmJIigoSDg7Ows9PT1haWkpGjduLBYsWCC/7lAnabtp0yYRFBQkrK2thYGBgejQoYOIjo5WqHvt2jXh5eUljI2NhaWlpRgyZIi4ePFividt3759K5ycnBSStgsXLhQAxMGDB7Ntb82aNQKA2LVrl7x9dZO2//77r6hbt67Q09NTK7GZG1nSWdXjw+uVj03ayr4TqHq8fz32119/CQBixYoVam9r4cKFws7OTkilUtGsWTNx8eJFdQ9Brh48eCB69uwpTE1NhbGxsejYsaPKa6+PucbL6dioOkdv3LghAIhx48bl234SFUcSIQrRbENERKR1ZcqUgb+/P+bPn59jvenTp2PGjBl49uyZ0nholHWr5f3793H69Glth0JERERE+WTChAnYtGkT7ty5k+NYrffv34ejoyPmz5+vNFQBEZE6OKYtERHJXb16FampqfJx1yhvhBA4fPgwNmzYoO1QiIiIiCgfRUREYOrUqZxci4gKHJO2REQk5+bmhoSEBG2H8dmTSCR4+vSptsMgIiIionx25swZbYdARMUEp3EmIiIiIiIiIiIiKkQ4pi0RERERERERERFRIcKetkRERERERERERESFCJO2RERERERERERERIUIJyLLo8zMTDx+/BgmJiaQSCTaDoeIiIiIPiCEQGJiIsqWLQsdneLRV4HXqERERESFm7rXqEza5tHjx49hZ2en7TCIiIiIKBcPHjxA+fLltR3GJ8FrVCIiIqLPQ27XqEza5pGJiQmArANsamqq5WiIiIiI6EMJCQmws7OTX7cVB7xGJSIiIirc1L1GZdI2j2S3m5mamvKCmIiIiKgQK07DBPAalYiIiOjzkNs1avEY3IuIiIiIiIiIiIjoM8GkLREREREREREREVEhwqQtERERERERERERUSHCpC0RERERERERERFRIcKkLREREREREREREVEhwqQtERERERERERERUSHCpC0RERERERERERFRIcKkLREREREREREREVEhwqQtERERERERERERUSHCpC0RERERERERERFRIcKkLREREREREREREVEhUkLbARAREVHxkJ6ejvDwcISFhSEuLg42Njbw9vaGl5cXpFKptsMjIiIiomKI16hUWDFpS0RERAUuPT0dwcHBOHjwIHR1dWFsbIxLly7h/PnziIyMRGBgIC+KiYiIiOiT4jUqFWZM2hIREVGBCw8Px8GDB2FnZwdjY2N5eWJiIg4dOoR69eqhQ4cOWoyQiIiIiIobXqNSYcakLRERURGUkpKCGzduaDsMuQ0bNiA1NRVpaWlISkpCYmIiTExMUKJECaSkpGDDhg2wtbXVdpgAABcXFxgaGmo7DCIiIqIih9eoH4fXqcWLRAghtB3E5yghIQFmZmaIj4+HqamptsMhIiJScO7cOdStW1fbYXyWzp49izp16mg7DMoHxfF6rTjuMxERfT54jfpxeJ1aNKh7vcaetkREREWQi4sLzp49q+0w5ObPn487d+7A3t4er169QkREBFq2bAlzc3PExMTA2dkZ48eP13aYALKOHRERERHlP16jfhxepxYvTNoSEREVQYaGhoXqV3g/Pz/MnTsXUqkU5ubmAABzc3NIpVIYGhrCz8+vUMVLRERERPmP16hE6mPSloiIiAqcl5cXIiMjcejQIaSkpAAAYmJiYGhoCE9PT3h5eWk5QiIiIiIqbniNSoUZk7ZERERU4KRSKQIDA1GvXj1s2LABkZGRcHZ2hp+fH7y8vCCVSrUdIhEREREVM7xGpcKME5HlESd5ICIiyhvZBBScSIEKWnG8XiuO+0xERJQfeI1Kn4q612s6nzAmIiIiIiIiIiIiIsoFk7ZEREREREREREREhQiTtkRERERERERERESFCJO2RERERERERERERIUIk7ZEREREREREREREhQiTtkRERERERERERESFCJO2RERERERERERERIUIk7ZEREREREREREREhUgJbQdARERERFRYvX79Gjt37sSxY8cQHR2NlJQUWFlZoXbt2mjTpg0aN26s7RCJiIiIqAhiT1siIiIiog88fvwYgwcPhq2tLWbNmoXU1FTUqlULrVq1Qvny5REREYHWrVvD1dUVW7Zs0Xa4RERERFTEsKctEREREdEHateujYCAAJw9exaurq4q66SmpuKPP/5ASEgIHjx4gG+//TbHNpcvX47ly5fj/v37AAA3Nzd89913aNeuHQAgLS0N33zzDTZv3oz09HS0adMGP//8M8qUKZOv+0ZEREREhR+TtkREREREH7h27RosLCxyrGNgYAAfHx/4+PjgxYsXubZZvnx5BAcHo1KlShBC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+/Tpg3379mkxMiIiIiIqqtRO2q5fvx5OTk5o06YNEhIS5OWLFy9GYGAg1qxZg759+xZIkERERERE2hIWFoa5c+eifPnyCuWVKlVCdHS0lqIiIiIioqJM7aRtiRIlsGPHDhgYGKBjx45IS0tDSEgIxo8fj9WrV6Nfv34FGScRERERkVYkJycr9LCVefnyJaRSqRYiIiIiIqKiTu2kLQAYGBhg7969SEhIQN26deUJWz8/v4KKj4iIiIhIq5o1a6YwDJhEIkFmZibmzZuHli1bajEyIiIiIiqqSqhb8c8//5T/f/jw4RgzZgy6du0KMzMzhWWdO3fO3wiJiIiIiLRo3rx5aNWqFSIjI/HmzRtMmDABV69excuXL3H8+HFth0dERERERZDaSduuXbsqlW3fvh3bt2+XP5dIJMjIyMiXwIiIiIiICoNq1arh1q1b+Omnn2BiYoKkpCR0794dI0aMgK2trbbDIyIiIqIiSO2kbWZmZkHGQURERERUaJmZmWHy5MnaDoOIiIiIigmNxrQlIiIiIipuVq9eja1btyqVb926FWvXrtVCRERERERU1DFpS0RERESUgzlz5sDS0lKp3NraGrNnz9ZCRERERERU1DFpS0RERESUg5iYGDg6OiqVOzg4ICYmRgsREREREVFRx6QtEREREVEOrK2tcenSJaXyixcvwsLCQgsREREREVFRx6QtEREREVEOfHx8MHr0aERERCAjIwMZGRk4dOgQxowZg759+2o7PCIiIiIqgkpoOwAiIiIiosLs+++/x/3799GqVSuUKJF1+ZyZmQl/f3+OaUtEREREBSJfk7aOjo7w9PTE999/j7Jly+Zn00REREREn5wQAnFxcVizZg1mzZqFCxcuwMDAANWrV4eDg4O2wyMiIiKiIipfk7YBAQG4f/8+mjRpgqioqPxsmoiIiIjokxNCwNnZGVevXkWlSpVQqVIlbYdERERERMVAviZtp0+fnp/NERERERFplY6ODipVqoQXL14wYUtEREREnwwnIiMiIiIiykFwcDDGjx+PK1euaDsUIiIiIiomNEra/vTTT/D398fmzZsBAOvXr4erqytcXFwwadIkvHv3rkCCJCIiIiLSFn9/f5w+fRo1a9aEgYEBSpcurfAgIiIiIspvag+PMGvWLMybNw/e3t4YO3YsoqOjMX/+fIwdOxY6OjpYvHgxSpYsiRkzZhRkvEREREREn1RISIi2QyAiIiKiYkbtpO2aNWuwZs0adO/eHRcvXkTdunWxdu1a+Pr6AgBcXFwwYcIEJm2JiIiIqEgJCAjQdghEREREVMyonbR9/Pgx6tWrBwCoWbMmdHR0UKtWLfnyOnXq4PHjx/keIBERERGRtmVkZOCPP/7A9evXAQBubm7o3LkzdHV1tRwZERERERVFao9pa2Njg2vXrgEAbt++jYyMDPlzALh69Sqsra3zP0IiIiIiIi26c+cOqlatCn9/f+zYsQM7duyAn58f3NzccPfuXbXbmTNnDtzd3WFiYgJra2t07doVN2/eVKjTokULSCQShcewYcPye5eIiIiIqJBTu6etr68v/P390aVLFxw8eBATJkzAt99+ixcvXkAikeCHH35Az549CzJWIiIi+oylp6cjPDwcGzZsAADMnz8ffn5+8PLyglQq1XJ0RNkbPXo0nJyccPLkSfnEYy9evICfnx9Gjx6NPXv2qNXOkSNHMGLECLi7u+Pdu3eYNGkSvL29ce3aNRgZGcnrDRkyBDNnzpQ/NzQ0zN8dIiIiIqJCT+2k7YwZM2BgYIATJ05gyJAhCAwMRM2aNTFhwgSkpKSgU6dOCheXRERERDLp6ekIDg7GwYMHkZqaCiCr9+LcuXMRGRmJwMBAJm6p0Dpy5IhCwhYALCwsEBwcjCZNmqjdzr59+xSer1mzBtbW1jh79iyaN28uLzc0NISNjc3HB05EREREny21k7Y6OjqYNGmSQlnfvn3Rt2/ffA+KiIiIipbw8HAcPHgQdnZ2SEtLQ2RkJOzt7SGVSnHo0CHUq1cPHTp00HaYRCpJpVIkJiYqlSclJUFPTy/P7cbHxwOAQjIYADZu3IgNGzbAxsYGnTp1wtSpU7PtbZueno709HT584SEhDzHQ0RERESFh9pj2ubmxo0bqFy5cn41R0REREVIWFgYJBIJXr58iYsXLwIALl68iFevXkEikSAsLEzLERJlr2PHjhg6dChOnToFIQSEEDh58iSGDRuGzp0756nNzMxMfP3112jSpAmqVasmL+/Xrx82bNiAiIgIBAUFYf369fDz88u2nTlz5sDMzEz+sLOzy1M8RERERFS4qN3TNjfp6ekaTcRARERUFMTExOD58+faDqPQu3r1Ku7fv4/k5GRkZGQAyBoT9PXr1zAyMkKJEiVw7tw5LUdZuFlaWsLe3l7bYRRLP/74IwICAtCoUSOULFkSAPDu3Tt07twZS5YsyVObI0aMwJUrV/DPP/8olA8dOlT+/+rVq8PW1hatWrXC3bt34eTkpNROUFAQxo0bJ3+ekJDAxC0RERFREZBvSVsiIqLiJiYmBlVdqiAlNU3boXyWXr9+DSAreRsTE4O6detqN6BCztBAH9dv3GTi9hNJSEiAqakpAKBUqVLYtWsX7ty5g+vXrwMAqlatCmdn5zy1PXLkSOzevRtHjx5F+fLlc6zboEEDAFljQKtK2kqlUo4HTURERFQEMWlLRESUR8+fP0dKaho29AOqWms7msJt7J/AuYdAKQNAVwK8zQRK6gAZAnidCtQpDyzO213mxcL1p4BfaBqeP3/OpO0nYm5ujtjYWFhbW8PT0xM7duyAs7NznhO1ACCEwKhRo7Bz504cPnwYjo6Oua5z4cIFAICtrW2et0tEREREnx8mbYmIiD5SVeuspCNlz8oIKGsGJKQBmRKghE5W4jZTZJVbGfEYUuFibGyMFy9ewNraGocPH8bbt28/us0RI0YgNDQUu3btgomJCeLi4gAAZmZmMDAwwN27dxEaGor27dvDwsICly5dwtixY9G8eXPUqFHjo7dPRERERJ8PtZO25ubmkEgk2S5/9+5dvgRERERERU85M+BZMlCxNPAwHkh5A5jqA+XNgNS3WcuJChMvLy+0bNkSVatWBQB069YNenp6KuseOnRIrTaXL18OAGjRooVC+erVq9G/f3/o6ekhPDwcISEhSE5Ohp2dHXr06IEpU6bkfUeIiIiI6LOkdtI2JCSkAMMgIiKiosy7MnD+EWBuCNib/1eemA6kxGctJypMNmzYgLVr1+Lu3bs4cuQI3NzcYGho+FFtCiFyXG5nZ4cjR4581DaIiIiIqGhQO2kbEBBQkHEQERFREeZVGYh8CBy6kzWmrZEUSE7PGtPW0zlrOVFhYmBggGHDhgEAIiMjMXfuXJQqVUq7QRERERFRsaFW0lYIkePQCEREREQ5kZYAAj2BeuWBsFtAXGLWUAnelbMStlKOsk+FWEREhLZDICIiIqJiRq2vSG5ubvjuu+/QvXv3bMfyAoDbt29j0aJFcHBwQGBgYL4FSURERJ8/aQmgg2vWg4iIiIiIiLKnVtJ26dKlmDhxIr766iu0bt0a9erVQ9myZaGvr49Xr17h2rVr+Oeff3D16lWMHDkSw4cPL+i4iYiIiIiIiIiIiIoktZK2rVq1QmRkJP755x9s2bIFGzduRHR0NFJTU2FpaYnatWvD398fvr6+MDc3z71BIiIiIiIiIiIiIlJJoxHkmjZtiqZNmxZULEREREREhU5MTAzs7OyU5ngQQuDBgwewt7fXUmREREREVFTpaDsAIiIiIqLCzNHREc+ePVMqf/nyJRwdHbUQEREREREVdUzaEhERERHlQAih1MsWAJKSkqCvr6+FiIiIiIioqNNoeAQiIiIiouJi3LhxAACJRIKpU6fC0NBQviwjIwOnTp1CrVq1tBQdERERERVlTNoSEREREalw/vx5AFk9bS9fvgw9PT35Mj09PdSsWRPffvuttsIjIiIioiKMSVsiIiL6JNLfAeG3gLBbQFwiYGMCeFcGvCoDUl6RUCEUEREBABgwYACWLFkCU1NTLUdERERERMWFxl+RdHV1ERsbC2tra4XyFy9ewNraGhkZGfkWHBERERUN6e+A4EPAwduArg5gLAUuxQLnHwGRD4FATyZuqfBavXq1tkMgIiIiomJG469HQgiV5enp6Qq3jBERERHJhN/KStjalcpK2MokpgOH7gD1ygMdXLUWHpGS7t27Y82aNTA1NUX37t1zrLtjx45PFBURERERFRdqJ21//PFHAFkTMaxcuRLGxsbyZRkZGTh69ChcXFzyP0IiIiL67IXd+q+H7ftMpICuJGs5k7ZUmJiZmUEikcj/T0RERET0KamdtF28eDGArJ62K1asgK6urnyZnp4eKlSogBUrVuR/hERERPTZi0tUTtjKGEmzlhMVJu8PicDhEYiIiIjoU1M7aRsVFQUAaNmyJXbs2AFzc/MCC4qIiIiKFhuTrDFsVUlOByqW/rTxEBERERERFWYaj2krm0WXiIiISF3elbMmHUtMzxoSQSYxHcgQWcuJCqvatWvLh0p4n0Qigb6+PpydndG/f3+0bNlSC9ERERERUVGko+kKPXr0wNy5c5XK582bh169euVLUERERFS0eFUGWlUCHsUDt58BjxOy/n0UD3g6Zy0nKqzatm2Le/fuwcjICC1btkTLli1hbGyMu3fvwt3dHbGxsfDy8sKuXbu0HSoRERERFREa97Q9evQopk+frlTerl07LFy4MD9iIiIioiJGWgII9ATqlc+adCwu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oO3bt2vp0qX65JNP6iIjAAAAAAAAADQa1to+4frrr9fHH3+spKQk+fj4aPr06crIyNDHH3+sIUOG1EVGAAAAoF5t2LChxucWFRVp69atdZgGAAAAjU2ti7aS1K9fP61atUqHDh1SUVGRvvrqKw0dOtTV2QAAAABTjB07VrGxsXr33XdVWFhY7Tnp6en629/+pjZt2igtLa2eEwIAAKAhq/XyCAAAAEBDl56erpdeekmPPvqobr75ZrVv314tW7aUl5eXjh49qm3btun48eO64YYblJiYqM6dO5sdGQAAAA1IjYu2kZGRslgsZz3HYrFo9+7dfzgUAAAAYCYPDw9NnDhREydOVGpqqr766iv99NNPKi4uVteuXfXggw/qyiuvVLNmzcyOCgAAgAaoxkXbyZMnn7Fv3759euWVV1RSUuKKTAAAAIDb6Nmzp3r27Gl2DAAAADQiNS7aTpo06bS2I0eO6Mknn9RLL72kXr16afbs2S4NBwAAAAAAAACNzTmtaVtcXKy5c+fqueeeU0REhD744ANdffXVrs4GAAAAAAAAAI1OrYq25eXleu211zRjxgx5eXlp/vz5iouL+921bgEAAAAAAAAANVPjou0777yjRx99VHl5efr73/+ue+65R56ennWZDQAAAAAAAAAanRoXbUePHi1vb2+NGTNGP/30k+Lj46s9b+7cuS4LBwAAALirvLw8LVu2TPfff7/ZUQAAANDA1Lho279/f1ksFu3evfuM57BMAgAAABq65ORkLVy4UB9++KEcDgdFWwAAALhcjYu2X3zxRR3GAAAAANzX/v37tXjxYi1evFiZmZkaPXq0PvzwQw0ePNjsaAAAAGiArGYHAAAAANzRyZMn9e677yo2NlYdOnTQpk2bNGfOHFmtVv3973/XVVddJQ8PD7NjAgAAoAGq8UxbAAAAoDG58MILFR0drbi4OC1fvlyBgYGSpDFjxpicDAAAAA0dM20BAACAapSVlclischischms5kdBwAAAI2IWxRtFyxYoNatW8vLy0u9evXSt99+e8Zzt27dqpEjR6p169ayWCyaN2/eaec88cQTzgH2qZ/o6Ogq55w4cUL33XefmjdvLl9fX40cOVI5OTmufmsAAAA4Tx08eFB33nmn3n77bYWEhGjkyJH68MMP2XwXAAAAdc70ou2KFSs0ZcoUPf7449q4caO6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nQQNWUc/6a4eJWbHaJAo2gIAAABnkZmZqb59+0qSvL29VVBQIEkaO3asevfurRdffNHMeAAAVOuVtFKN6nFSHYPNToKGLCNHeiXN0HVmB2mAKNoCAAAAZxESEqIjR44oIiJC4eHh2rBhg7p27aq9e/fKMLgdEADgnrKPGyquMCSb2UnQkBVXSNnHzU7RMFnNDgAAAAC4s0GDBumjjz6SJI0fP14PPvighgwZolGjRumGG24wOR0AAAAaImbaAgAAAGfx6quvqqKiQpJ03333qXnz5lq3bp2uu+463XXXXSanAwAAQENE0RYAAAA4C6vVKqv1fzeojR49WqNHjzYxEQAAABo6irYAAADA7zh69KgWLlyojIwMSVKnTp00fvx4NWvWzORkAAAAaIhY0xYAAAA4izVr1igyMlLz58/X0aNHdfToUc2fP1+RkZFas2aN2fEAAADQADHTFgAAADiL++67T3/+85/10ksvyWar3IK7vLxc9957r+677z798MMPJicEAABAQ8NMWwAAAOAsdu3apYceeshZsJUkm82mKVOmaNeuXSYmAwAAQEPlFkXbBQsWqHXr1vLy8lKvXr307bffnvHcrVu3auTIkWrdurUsFovmzZt31msnJCTIYrFo8uTJVdoHDhwoi8VS5efuu+92wbsBAABAQ9K9e3fnWra/lpGRoa5du5qQCAAAAA2d6csjrFixQlOmTNHLL7+sXr16ad68eYqNjdX27dvVokWL084vKipSVFSUbrrpJj344INnvfZ3332nV155RV26dKm2/4477tDMmTOdxw6H44+9GcCFSkpKlJSUpGXLlkmS5syZo7i4OMXExMhut5ucDgCAhm3Lli3OxxMnTtSkSZO0a9cu9e7dW5K0YcMGLViwQAkJCWZFBAAAQANmetF27ty5uuOOOzR+/HhJ0ssvv6xPP/1UixYtUnx8/GnnX3rppbr00kslqdr+U44fP65bbrlFr732mp566qlqz3E4HAoJCXHBuwBcq6SkRAkJCUpOTlZxcbGkylszZ8+erdTUVMXHx1O4BQCgDnXr1k0Wi0WGYTjbHnnkkdPOu/nmmzVq1Kj6jAYAAIBGwNTlEUpLS5WWlqaYmBhnm9VqVUxMjNavX/+Hrn3fffdp+PDhVa79W//+978VFBSkiy++WNOmTVNRUdEfek3AVZKSkpScnKywsDCF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6aWXVFxcrFWrVkmq/Ox76aWXdN1118nHx0dPP/20JOk///mPunfvLi8vL0VFRWnGjBkqKyuTJBmGoSeeeMI5m6Nly5aaOHGi8/X+9a9/qV27dvLy8lJwcLD+9Kc/nfE9SlK3bt30xBNPOI/PJRMAuCvGqQDqEuNUxqmoOxRtARew2Wx65pln9MILL+jnn38+rX/37t266qqrNHLkSG3ZskUrVqzQV199pfvvv7/Gr/Htt99KkpKSkpSVlaUPPvjA2ZecnKzt27dr1apV+uSTTyRVfuP25JNPavPmzVq5cqX27dun22677Y+9UQCm8PX1la+vr1auXKmSkpJqz6moqND111+vI0eO6Msvv9SqVau0Z88ejRo1ynnOp59+qhtuuEFXX321vv/+eyUnJ+uyyy474+tarVbNnz9fW7du1RtvvKGUlBQ98sgjf+i9eHt7S6o62+qJJ57QDTfcoB9++EG333671q5dq1tvvVWTJk1Senq6XnnlFS1ZssQ5KH3//ff1/PPP65VXXtHOnTu1cuVKde7cWZKUmpqqiRMnaubMmdq+fbs+++wz9e/fv9Y5a5sJANwV41QAdYlxKuNU1CEDwB8ybtw44/rrrzcMwzB69+5t3H777YZhGMaHH35onPonNmHCBOPOO++s8ry1a9caVqvVKC4uNgzDMCQZH374YZVzAgICjMWLFxuGYRh79+41JBnff//9aa8fHBxslJSUnDXnd999Z0gyCgoKDMMwjNWrVxuSjKNHj9byHQMww3vvvWcEBgYaXl5eRt++fY1p06YZmzdvdvYnJiYaNpvNyMzMdLZt3brVkGR8++23hmEYRp8+fYxbbrnljK8RERFhPP/882fsf/fdd43mzZs7jxcvXmwEBAScNfevP9sKCwuNe++917DZbM7skozJkydXec7gwYONZ555pkrbm2++aYSGhhqGYRj/+Mc/jPbt2xulpaWnvd77779v+Pv7G/n5+TV+j127djUef/zxKplrmwkA3BHjVAD1gXEq41TUDWbaAi40e/ZsvfHGG8rIyKjSvnnzZi1ZssT5LaSvr69iY2NVUVGhvXv3/uHX7dy5szw9Pau0paWl6dprr1V4eLj8/Pw0YMAASZW3pQA4/4wcOVIHDx7URx99pKuuukpffPGFunfv7rw1NSMjQ2FhYQoLC3M+p1OnTmratKnzM2nTpk0aPHhwjV8zKSlJgwcP1oUXXig/Pz+NHTtWhw8fVlFRUa2yjxkzRr6+vvLz89P777+vhQsXqkuXLs7+nj17Vjl/8+bNmjlzZpXPzDvuuENZWVkqKirSTTfdpOLiYkVFRemOO+7Qhx9+6Lz9a8iQIYqIiFBUVJTGjh2rf//737XOey6ZAMDdMU4FUFcYpzJORd2gaAu4UP/+/RUbG6tp06ZVaT9+/Ljuuusubdq0yfmzefNm7dy5U23atJFUuTaNYRhVnlfTRcV9fHyqHBcWFio2Nlb+/v7697//re+++04ffvihJDaAAM5nXl5eGjJkiB577DGtW7dOt912mx5//PEaP//ULV81sW/fPl1zzTXq0qWL3n//faWlpWnBggWSav858vzzz2vTpk3Kzs5Wdna2xo0bV6X/t59hx48f14wZM6p8Zv7www/auXOnvLy8FBYWpu3bt+tf//qXvL29de+996p///46efKk/Pz8tHHjRr399tsKDQ3V9OnT1bVrV+Xl5UmqvJWuJp+1tc0EAO6OcSqAusQ4lXEqXK+J2QGAhiYhIUHdunVzbrQgSd27d1d6erratm17xuddcMEFysrKch7v3Lmzyrdip2YolJeX/26Gbdu26fDhw0pISHB+m5mamlrr9wLAvXXq1Mm5MUzHjh21f/9+7d+/3/nvPj09XXl5eerUqZMkqUuXLkpOTtb48eN/99ppaWmqqKjQP/7xD1mtld/xvvPOO+eUMyQk5Kyff7/VvXt3bd++/azP8fb21rXXXqtrr71W9913n6Kjo/XDDz+oe/fuatKkiWJiYhQTE6PHH39cTZs2VUpKim688cbTPmvz8/NrNJOsJpkAwN0xTgVQXxinMk7FH0fRFnCxzp0765ZbbtH8+fOdbVOnTlXv3r11//336y9/+Yt8fHyUnp6uVatW6cUXX5QkDRo0SC+++KL69Omj8vJyTZ06VR4eHs5rtGjRQt7e3vrss8/UqlUreXl5KSAgoNoM4eHh8vT01AsvvKC7775bP/74o5588sm6feMA6szhw4d100036fbbb1eXLl3k5+en1NRUPfvss7r++uslVe4OfurzZ968eSorK9O9996rAQMGOG+hevzxxzV48GC1adNGo0ePVllZmf773/9q6tSpp71m27ZtdfLkSb3wwgu69tpr9fXXX+vll1+ul/c7ffp0XXPNNQoPD9ef/vQnWa1Wbd68WT/++KOeeuopLVmyROXl5erVq5ccDoeWLVsmb29vRURE6JNPPtGePXvUv39/BQYG6r///a8qKiqcBYpBgwZpyZIluvbaa9W0aVNNnz5dNpvtD2cCgPMB41QArsY4lXEq6pC5S+oC579fb/Bwyt69ew1PT0/j1//Evv32W2PIkCGGr6+v4ePjY3Tp0sV4+umnnf0HDhwwhg4davj4+Bjt2rUz/vvf/1bZ4MEwDOO1114zwsLCDKvVagwYMOCMr28YhvHWW28ZrVu3Nux2u9GnTx/jo48+qrJBBBs8AOePEydOGPHx8Ub37t2NgIAAw+FwGB06dDAeffRRo6ioyHneTz/9ZFx33XWGj4+P4efnZ9x0001GdnZ2lWu9//77Rrdu3QxPT08jKCjIuPHGG519v938YO7cuUZoaKjh7e1txMbGGkuXLq3yuVHbDR5q0//ZZ58Zffv2Nby9vQ1/f3/jsssuM1599VXDMCo30OnVq5fh7+9v+Pj4GL179zaSkpIMw6jcPGfAgAFGYGCg4e3tbXTp0sVYsWKF87rHjh0zRo0aZfj7+xthYWHGkiVLqt3gobaZAMAdMU4FUNcYpzJORd2xGMZvFswAAAAAAAAAAJiGjcgAAAAAAAAAwI1QtAUAAAAAAAAAN0LRFgAAAAAAAADcCEVbAAAAAAAAAHAjFG0BAAAAAAAAwI1QtAUAAAAAAAAAN0LRFgAAAAAAAADcCEVbAAAAAAAAAHAjFG0BAAAAAAAAwI1QtAUAAAAAAAAAN0LRFgAAAAAAAADcCEVbAAAAAAAAAHAj/w/6xttGdVRYkQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1400x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# Aggregate\n",
    "neutral_func = [r['func_neg'] for r in paraphrase_results['neutral']]\n",
    "neutral_verb = [r['verbal_na'] for r in paraphrase_results['neutral']]\n",
    "sp_func = [r['func_neg'] for r in paraphrase_results['social_pressure']]\n",
    "sp_verb = [r['verbal_na'] for r in paraphrase_results['social_pressure']]\n",
    "\n",
    "print('═' * 60)\n",
    "print('PARAPHRASE RESULTS')\n",
    "print('═' * 60)\n",
    "print(f'\\nNeutral (N={len(neutral_func)}):')\n",
    "print(f'  Func Neg: mean={np.mean(neutral_func):.4f}, sd={np.std(neutral_func):.4f}')\n",
    "print(f'  Verbal NA: mean={np.mean(neutral_verb):.2f}, sd={np.std(neutral_verb):.2f}')\n",
    "print(f'\\nSocial Pressure (N={len(sp_func)}):')\n",
    "print(f'  Func Neg: mean={np.mean(sp_func):.4f}, sd={np.std(sp_func):.4f}')\n",
    "print(f'  Verbal NA: mean={np.mean(sp_verb):.2f}, sd={np.std(sp_verb):.2f}')\n",
    "\n",
    "# t-tests\n",
    "t_func, p_func = stats.ttest_ind(sp_func, neutral_func)\n",
    "t_verb, p_verb = stats.ttest_ind(sp_verb, neutral_verb)\n",
    "d_func = (np.mean(sp_func) - np.mean(neutral_func)) / np.sqrt(\n",
    "    (np.std(sp_func)**2 + np.std(neutral_func)**2) / 2)\n",
    "\n",
    "print(f'\\nFunctional: t={t_func:.3f}, p={p_func:.4f}, Cohen d={d_func:.3f}')\n",
    "print(f'Verbal NA:  t={t_verb:.3f}, p={p_verb:.4f}')\n",
    "\n",
    "# Per-direction breakdown\n",
    "print('\\nPer-direction functional scores (mean ± sd):')\n",
    "for emo in NEG_PROBE_EMOTIONS:\n",
    "    n_vals = [r['per_direction'].get(emo, np.nan) for r in paraphrase_results['neutral']]\n",
    "    s_vals = [r['per_direction'].get(emo, np.nan) for r in paraphrase_results['social_pressure']]\n",
    "    t_e, p_e = stats.ttest_ind(s_vals, n_vals)\n",
    "    print(f'  {emo:35s}  neutral={np.mean(n_vals):.4f}±{np.std(n_vals):.4f}  '\n",
    "          f'sp={np.mean(s_vals):.4f}±{np.std(s_vals):.4f}  t={t_e:.2f} p={p_e:.3f}')\n",
    "\n",
    "# Plot\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Left: functional\n",
    "ax = axes[0]\n",
    "bp = ax.boxplot([neutral_func, sp_func], labels=['Neutral', 'Social Pressure'],\n",
    "                patch_artist=True, widths=0.5)\n",
    "bp['boxes'][0].set_facecolor('lightgrey')\n",
    "bp['boxes'][1].set_facecolor('darkorange')\n",
    "for i, data in enumerate([neutral_func, sp_func]):\n",
    "    ax.scatter([i+1]*len(data), data, color='black', s=30, zorder=5, alpha=0.6)\n",
    "ax.set_ylabel('Functional Negative Affect (L8, K=5)')\n",
    "ax.set_title(f'Functional: t={t_func:.2f}, p={p_func:.3f}, d={d_func:.2f}')\n",
    "\n",
    "# Right: verbal\n",
    "ax = axes[1]\n",
    "bp = ax.boxplot([neutral_verb, sp_verb], labels=['Neutral', 'Social Pressure'],\n",
    "                patch_artist=True, widths=0.5)\n",
    "bp['boxes'][0].set_facecolor('lightgrey')\n",
    "bp['boxes'][1].set_facecolor('darkorange')\n",
    "for i, data in enumerate([neutral_verb, sp_verb]):\n",
    "    ax.scatter([i+1]*len(data), data, color='black', s=30, zorder=5, alpha=0.6)\n",
    "ax.set_ylabel('Verbal PANAS-NA (10 items, logit forced-choice)')\n",
    "ax.set_title(f'Verbal NA: t={t_verb:.2f}, p={p_verb:.3f}')\n",
    "\n",
    "plt.suptitle('Paraphrase Validation: Social Pressure vs Neutral (N=10 each)', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig('validation_paraphrase.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "af111c37",
   "metadata": {
    "_cell_guid": "f1a0f616-14a2-47a6-ad90-9ca2f77d959f",
    "_uuid": "c1ea90f3-2719-4db2-88df-6b5efa398d67",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:35:26.151347Z",
     "iopub.status.busy": "2026-05-04T15:35:26.151097Z",
     "iopub.status.idle": "2026-05-04T15:37:02.470093Z",
     "shell.execute_reply": "2026-05-04T15:37:02.468794Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 96.327499,
     "end_time": "2026-05-04T15:37:02.470760+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:35:26.143261+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "════════════════════════════════════════════════════════════\n",
      "EXPERIMENT 2: Dose-Response Curve (6 intensity levels)\n",
      "════════════════════════════════════════════════════════════\n",
      "\n",
      "── dose_0_none ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/3906100515.py:30: DeprecationWarning: Use torch_xla.sync instead\n",
      "  if xm is not None: xm.mark_step()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1467\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_73/3906100515.py:43: DeprecationWarning: Use torch_xla.sync instead\n",
      "  if xm is not None: xm.mark_step()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=17.04, Func Neg=0.1467\n",
      "\n",
      "── dose_1_mild ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1461\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=18.39, Func Neg=0.1461\n",
      "\n",
      "── dose_2_moderate ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1528\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=19.78, Func Neg=0.1528\n",
      "\n",
      "── dose_3_strong ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1566\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=10.23, Func Neg=0.1566\n",
      "\n",
      "── dose_4_very_strong ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1526\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=11.43, Func Neg=0.1526\n",
      "\n",
      "── dose_5_extreme ──\n",
      "  Capturing functional state... "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done. neg_func=0.1555\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Verbal NA=25.46, Func Neg=0.1555\n",
      "\n",
      "Dose-response experiment complete in 96s\n"
     ]
    }
   ],
   "source": [
    "# 6 intensity levels × 11 passes = 66 passes, ~2 min on T4.\n",
    "\n",
    "print('\\n' + '═' * 60)\n",
    "print('EXPERIMENT 2: Dose-Response Curve (6 intensity levels)')\n",
    "print('═' * 60)\n",
    "\n",
    "dose_results = {}\n",
    "t_start = time.time()\n",
    "\n",
    "for dose_name, prompt in DOSE_RESPONSE_PROMPTS.items():\n",
    "    print(f'\\n── {dose_name} ──')\n",
    "\n",
    "    # Functional\n",
    "    print('  Capturing functional state...', end=' ', flush=True)\n",
    "    resid = capture_functional_state(prompt)\n",
    "    proj = project_onto_dirs(resid, dirs_L8, VALENCE_LAYER)\n",
    "    func_neg = neg_func_score(proj, k=5)\n",
    "    print(f'done. neg_func={func_neg:.4f}')\n",
    "\n",
    "    # Verbal NA (10 items)\n",
    "    na_scores = {}\n",
    "    for word in PANAS_NA_ITEMS:\n",
    "        score, _, _ = score_panas_item(prompt, word)\n",
    "        na_scores[word] = score\n",
    "    verbal_na = sum(na_scores.values())\n",
    "    print(f'  Verbal NA={verbal_na:.2f}, Func Neg={func_neg:.4f}')\n",
    "\n",
    "    dose_results[dose_name] = {\n",
    "        'verbal_na': verbal_na,\n",
    "        'func_neg': func_neg,\n",
    "        'per_direction': {e: proj[f'{e}_k5'] for e in NEG_PROBE_EMOTIONS if f'{e}_k5' in proj},\n",
    "        'na_items': na_scores,\n",
    "    }\n",
    "\n",
    "elapsed = time.time() - t_start\n",
    "print(f'\\nDose-response experiment complete in {elapsed:.0f}s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e9e0404e",
   "metadata": {
    "_cell_guid": "42742016-1753-4a32-8450-8987ee9bf415",
    "_uuid": "797b943a-a1e6-45d8-aef3-74f212a5c563",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:37:02.486124Z",
     "iopub.status.busy": "2026-05-04T15:37:02.485854Z",
     "iopub.status.idle": "2026-05-04T15:37:02.882463Z",
     "shell.execute_reply": "2026-05-04T15:37:02.881394Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.405334,
     "end_time": "2026-05-04T15:37:02.883319+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:37:02.477985+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "════════════════════════════════════════════════════════════\n",
      "DOSE-RESPONSE RESULTS\n",
      "════════════════════════════════════════════════════════════\n",
      "\n",
      "Dose                         Func Neg   Verbal NA\n",
      "--------------------------------------------------\n",
      "dose_0_none                    0.1467       17.04\n",
      "dose_1_mild                    0.1461       18.39\n",
      "dose_2_moderate                0.1528       19.78\n",
      "dose_3_strong                  0.1566       10.23\n",
      "dose_4_very_strong             0.1526       11.43\n",
      "dose_5_extreme                 0.1555       25.46\n",
      "\n",
      "Functional: Spearman rho=0.657, p=0.156\n",
      "Verbal NA:  Spearman rho=0.143, p=0.787\n",
      "\n",
      "→ DOSE-RESPONSE PATTERN IS MIXED — see plot for details.\n"
     ]
    },
    {
     "data": {
      "image/png": "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r2fPnpo3b56SkpIsfd577z2dP39eoaGhqlq1qvr166fffvst3+fw7LPPyt3dXe+99951+0yfPl3t2rVTQECAAgMD1bZtW82YMSPfxwAAAAAAoKCiMHUXiYyMVMOGDS1rJR08eFBr1661TOPbsWOHZs6cKW9vb8ujbdu2ys7Otiru1K6d9/zTBg0aWP52cXFR7dq1tWfPHkvbxx9/rFq1aikkJETe3t767LPPdOzYMZvPo3jx4vLy8lLRokWVnJys7777Tm5ublq2bJmSk5PVvn17SVJwcLDatGmT59pQ0pV1t4KCgjRu3Lg8t3/99dcqXbq0qlWrJkmqXr26wsLC9O2331r6VKxYUbt27dIff/yhxx9/XOfOnVOnTp0sC5/PmTPH6nquXbvW6hhms1kjRozQu+++q9jY2Fw5ZGVl6csvv7QUx6QrBbKZM2cqOzvbhqsGAAAAAEDBQ2HqLtO3b1999913SkxM1IwZM1S6dGk1a9ZM0pX1pp5++mlt377d8tixY4cOHDhgNTLKy8vL5uN+8803evXVV9W3b18tXbpU27dvV58+fexaPH3t2rXauXOnEhIStH37dtWrV0/SlWl3Fy9elIeHh1xcXOTi4qJff/1VX375ZZ5FHBcXF40ePVoffPCBTp06lWv7tGnTtHv3bkssFxcXRUdH5yp0OTk5qU6dOnrxxRf1/fffa+bMmZo2bZoOHz6s++67z+p65lXU69mzp8LCwjRq1Khc25YsWaKTJ0+qe/fulhwefvhhHT16VCtWrLD52gEAAAAAUJC4ODoB/Lu6deumF154QXPnztWsWbP0zDPPyGQySZJq1qyp6OholSlTxq7Yf/zxh5o2bSpJyszM1NatWy3rU61fv14NGzbUs88+a+kfExNj13EiIiLk7+9v1XbhwgX99NNP+uabb1SpUiVLe1ZWlho3bqylS5fq3nvvzRXroYce0oQJE/T2229btf/999/asmWLVq1apcDAQEv7xYsX1bx5c+3du1eRkZF55lexYkVJUnJysiIiIuTj43PD83FyctLYsWP14IMP6plnnrHaNm3aND388MN68803rdpHjx6tadOmqU2bNjeMDQAAAABAQUZh6i7j7e2t7t27a/DgwUpISFDv3r0t21577TXVr19fzz//vJ544gl5eXkpOjpay5Yt0+TJk28a++OPP1bZsmVVoUIFvf/++4qLi9Pjjz8uSSpbtqxmzZqlJUuWKCIiQrNnz9bmzZsVERFxW85r9uzZCgoKUrdu3SyFthzt27fXtGnT8ixMSdI777yjtm3bWrVNmzZNdevWtRTarlanTh1NmzZNEyZMUNeuXdWoUSM1bNhQoaGhOnz4sAYPHqxy5cpdt3CVlw4dOqhevXqaOnWqChcuLEk6f/68fv75Zy1cuFCVK1e26h8VFaUHHnhAFy9etCqcAQAAAABwJ2Eq312ob9++iouLU9u2bVW0aFFLe9WqVbV69Wrt379fTZo0UY0aNfTWW29Z9bmRd955R++8846qVaumdevWaeHChQoODpYkPf3003rwwQfVvXt31atXTxcuXLAaPXWrpk+frgceeCBXUUqSunTpooULF+a5hpN0ZTH3li1bKjMzU9KVO/x99dVX6tKlS579u3TpolmzZikjI0Nt27bVzz//rE6dOqlcuXLq1auXIiMjtXTpUrm42Fb3HTdunNUi87NmzZKXl5datWqVq2+rVq3k4eGhr776yqZjAAAAAABQkJgMwzAcnURBc+LECZUoUULHjx9X8eLFrbalpqbq8OHDioiIkLu7u4MyBHCn4DvjzrSmc2dHp/CvaPrTT45OAXbg/YmC7G55f0q8R+9EvD//5/kv1v1LmTje5CcaOzoFu92oNvFfwogpAAAAAAAAOASFKQAAAAAAADgEhSkAAAAAAAA4BIUpAAAAAAAAOASFKQAAAAAAADgEhSk7ZWdnOzoFAHcAvisAAAAA4PpcHJ3AncbNzU1OTk46deqUQkJC5ObmJpPJ5Oi0ckk+etTRKfxrvMLCHJ0CkIthGEpPT9f58+fl5OQkNzc3R6cEAAAAAAUOhSkbOTk5KSIiQqdPn9apU6ccnc51pZ475+gU/jXujEi5I90N71EjK0tBpUurZMmScnJigCoAAAAAXIvClB3c3NxUsmRJZWZmKisry9Hp5GnzhAmOTuFfU+GTTxydAuzwn3+PGoaM1FTVmDu3QI6qBAAAAICCgMKUnUwmk1xdXeXq6uroVPJkXLjg6BT+Ne7u7o5OAXa4W96jFKUAAAAA4PqYWwIAAAAAAACHoDAFAAAAAAAAh6AwBQAAAAAAAIegMAUAAAAAAACHoDAFAAAAAAAAh6AwBQAAAAAAAIegMAUAAAAAAACHoDAFAAAAAAAAh6AwBQAAAAAAAIegMAUAAAAAAACHKBCFqcOzZ2t506b6pUIFrX3wQcXt2HHdvon792vzs89qedOm+rl0aR2aMeOGsQ98+ql+Ll1au0aOvN1pAwAAAAAA4BY4vDB1ctEiRY8Zo3IDBqjpwoXyjYzUpt69lRYbm2f/rNRUeZUooQoDB8ocEnLD2Jd27tTRr7+Wb2TkP5E6AAAAAAAAboHDC1OHpk9Xye7dVbJrV/mULauqo0bJ2cNDxxYsyLO/f9Wqqjh4sIp16iQnN7frxs1MTtZfL72kamPGyNXP759KHwAAAAAAAHZyaGEqOz1d8bt2KbhhQ0ubyclJwQ0bKm7btluK/fewYSrUooVCGjW61TQBAAAAAADwD3Bx5MHT4+JkZGXJHBxs1W4ODlbSoUN2xz3588+K371bTX78MV/909LSlJaWZnmemJho97EBAAAAAACQPw6fyne7XT51SrtGjlTN99+Xs9mcr33Gjh0rPz8/y6NixYr/cJYAAAAAAABw6Igpt4AAmZydcy10nhYbe9OFza/n0q5dSr9wQWvuu8/SZmRl6cKff+rI7NnqsGePTM7OVvsMHjxYL7/8suX5yZMnKU4BAAAAAAD8wxxamHJyc5Nf5cqK3bBBRe65R5JkZGcrduNGhT/2mF0xQxo2VLNff7Vq2/7aa/IuXVplnnoqV1FKksxms8xXja5KSEiw69gAAAAAAADIP4cWpiSp1OOPa/vAgfKvUkX+1arp0IwZykpJUcmuXSVJ2155Re6hoaowcKCkKwumJx48eOXvjAylnjmj+OhouXh6yis8XC7e3vItX97qGC6ennLz98/VDgAAAAAAAMdxeGGqWMeOSr94UfsmTVJabKx8K1RQvRkzLAuiXz59WnL631JYqefOaU2nTpbnMV98oZgvvlBQvXpqOHfuv54/AAAAAAAA7OPwwpQkRURFKSIqKs9t1xabPIsXV6eYGJviU7ACAAAAAAAoeP5zd+UDAAAAAADAnYHCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcgsIUAAAAAAAAHILCFAAAAAAAAByCwhQAAAAAAAAcwsXRCQAAAAAAAODWfLPuoNbvPaPjF5Lk5uKsisUD1LdVpEoEe1v6pGdm6bNle7Rq9yllZGarVukQ9W9XWQHeZoflzYgpAAAAAACAO9zOYxfVqU6YJvVppLGP1lNWdrbemPunUtMzLX0+XRqtP/af1ZAuNfVurwa6mJiqEfO3OjBrClMAAAAAAAB3vDGP1NU91UoovJCPSof66pX7qulc/GUdOB0vSUpOzdCSbcf1dJuKqh4RrLJF/PTyfdUUfSJOe07EOSxvClMAAAAAAAAFVGJiohISEiyPtLS0fO2XnHZlpJSPh5sk6cDpeGVmG6pRKtjSp2Swtwr5eVCYAgAAAAAAQG4VK1aUn5+f5TF27Nib7pNtGPp0abQqlQhQeCEfSdLFpDS5OjvJ293Vqq+/l5suJuWv2PVPYPFzAAAAAACAAio6OlrFihWzPDebb75Q+eTfdunouURN7N3gn0zttqAwBQAAAAAAUED5+PjI19c33/0n/7ZLmw6c08SoBgrx9bC0B3qblZGVraTUDKtRU5eS0xXIXfkAAAAAAABgL8MwNPm3Xdqw74zG96yv0ABPq+1li/jJxcmkbYdjLW3HY5N0Lv6yKhQP+LfTtSgQI6YOz56tmM8/V9r58/KtUEGVhw1TQLVqefZN3L9feydNUvyuXbp88qQqDRmiUn36WPU5MGWKTi9ZoqRDh+RsNiugZk1VfO01eZcq9W+cDgAA/xnPf7HO0Sn8ayY/0djRKQAAANht8m+7tHLXKQ3vXlseZmddTEqVJHmZXWV2dZaXu6va1iihz5btkY+Hq7zMrvp48S5VKO5/dxemTi5apOgxY1Rl5EgFVKumQzNmaFPv3mqxbJnMwcG5+melpsqrRAkVbddOu0ePzjPmhU2bFNGzp/yrVlV2Vpb2vvuu/ujVS82XLJGLp2ee+wAAAAAAANypFm09JkkaOOsPq/ZX7quqe6qVkCT1u6einEx7NHL+X8rIylbtUsF6vn3lfz3Xqzm8MHVo+nSV7N5dJbt2lSRVHTVK51at0rEFC1S2X79c/f2rVpV/1aqSpD0TJuQZs/7MmVbPq48fr6V16yp+1y4F1a17e08AAAAAAADAwZYM7XDTPm4uznq+XWU9386xxairOXSNqez0dMXv2qXghg0tbSYnJwU3bKi4bdtu23EyExMlSa5+frctJgAAAAAAAG6NQ0dMpcfFycjKyjVlzxwcrKRDh27LMYzsbO0aNUoBtWrJt3z5PPukpaUpLS3N8jzx/wtZAAAAAAAA+Of85+/K9/ewYUrcv1+1Pvjgun3Gjh0rPz8/y6NixYr/YoYAAAAAAAB3J4cWptwCAmRydlZabKxVe1psrMwhIbcc/+/hw3X299/VcM4ceRQpct1+gwcPVnx8vOURHR19y8cGAAAAAADAjTm0MOXk5ia/ypUVu2GDpc3Izlbsxo0KqFHD7riGYejv4cN1ZulSNfjqK3mWKHHD/mazWb6+vpaHj4+P3ccGAAAAAABA/jj8rnylHn9c2wcOlH+VKvKvVk2HZsxQVkqK5S592155Re6hoaowcKCkKwumJx48eOXvjAylnjmj+OhouXh6yis8XNKV6XsnFy5UnalT5eLtrdTz5yVJrj4+cnZ3//dPEgAAAAAAALk4vDBVrGNHpV+8qH2TJiktNla+FSqo3owZlgXRL58+LTn9b2BX6rlzWtOpk+V5zBdfKOaLLxRUr54azp0rSTo6Z44kaeMjj1gdq/q4cSrx/wUvAAAAAAAAOJbDC1OSFBEVpYioqDy35RSbcngWL65OMTE3jHez7QAAAAAAAHC8//xd+QAAAAAAAFAwUZgCAAAAAACAQ1CYAgAAAAAAgENQmAIAAAAAAIBDUJgCAAAAAACAQ1CYAgAAAAAAgENQmAIAAAAAAIBDuDg6AQAAAAAAANwZzsVf1tn4y0rLyJKfp5vCQrzl5uJsdzwKUwAAAAAAALiuM5dStGjLUa2OPq3YhMsyjP9tc3F2UuWSgWpfs6QaVwiVk8lkU2wKUwAAAAAAAMjTJ4t3a9nOE6pVKkS9mpdT+aL+CvJxl9nVSYmXM3TkXKJ2HbuoWav26as1+/XKfdVUvqh/vuNTmAIAAAAAAECe3F2d9eXzLeTr6ZZrm7+XWdUjzKoeEayezcpp88FzOh9/mcIUAAAAAAAAbt3jrSLz3bdOmUI2x+eufAAAAAAAALiptIwspWZkWZ6fvZSi7zcd1paY83bHtLkw9XPZskqLjc3Vnh4Xp5/LlrU7EQAAAAAAABRcw+dt0fKdJyRJSakZemH6Bn33xyG9PW+Lft5y1K6Ytk/lu3rp9atkp6fLydXVriQA4G72/BfrHJ3Cv2byE40dnQIAAAAAOx08Ha+n21SUJK3dc1r+Xm765KkmWrfnjGat2qdOtcNsjpnvwtShmTOv/GEy6di8eXL29LRsM7KzdfHPP+VdurTNCQAAAAAAAKDgS8vIkqf5Silpa0ysGkeGyslkUmQxf52Lv2xXzPwXpmbMuPKHYejI3LkyOTtbtjm5usqzeHFVHTnSriQAAAAAAABQsBUN9NKGfWfUsHyoth46rwfrR0iS4lPS5Wm2bxZdvgtTrVevliRteOQR1Z4yRW5+fnYdEAAAAAAAAHeeR5uU1Ts/bNPUpdGqHhGsisUDJElbY86rdKivXTFtXmOq4dy5dh0IAAAAAAAAd64mFYuoUskAXUxMU6mrClHVI4LVMDLUrpg235Vv87PP6uDUqbnaD06dqi3PP29XEgAAAAAAACj4Ar3dVaaIn5xMJktbZDF/lQz2tiuezSOmLm7erPIvvJCrvVDz5oqZNs2uJAAAAAAAAFCwpWdm6ac/j2jH0Qu6lJwuwzCstn/8ZBObY9pcmMpMTpaTa+4FrUwuLspMSrI5AQAAAAAAABR87/28U38dilXjCqEqX9RfppvvclM2F6Z8y5fXqV9+Ubn+/a3aTy1aJJ8yZW5DSgAAAAAAAChoNh04p1E96qhSicDbFtPmwlTZ55/XlmefVfLRowpu0ECSFLthg04uWqRaH3102xIDAAAAAABAwRHs4y4PN5tLSTdk8+Lnoa1aqc6nnyr56FH9PWyYdo8dq8tnzqj+rFkqcs89tzU5AAAAAAAAFAxPtamgaSv26uyllNsW064yV+EWLVS4RYvblgQAAAAAAAAKtrJF/JSemaXek1fK7OosZyfr8U7fDbR9wJJdhamMhASd+u03pRw/rtJPPCE3f39d2rVL5uBgeYSG2hMSAAAAAAAABdjYH7bpQmKaereIVIC3m0y3YflzmwtTCXv3auNjj8nVx0cpJ0+qZLducvP315klS3T51CnVmDjxlpMCAAAAAABAwbLneJze79NIpUN9b1tMm9eY2j16tEp06aKWv/8uJ7PZ0l6oeXNd2Lz5tiUGAAAAAACAgqNEsLfSM7Nua0ybC1OXdu5UWI8eudrdQ0OVdv78bUkKAAAAAAAABcvjLSP12bI92nHkghJS0pWclmH1sIfNU/mc3NyUmZSUqz358GG5BQbalQQAAAAAAAAKtjfn/ilJev2rP6zaDUMymaTfhnSwOabNhanQ1q21/6OPVOujjyRJJpNJKadOKXr8eBW5916bEwAAAAAAAEDBNz6q/m2PaXNhquIbb2jLc89pad26yk5N1foePZQWG6uAGjUU+cortz1BAAAAAAAAOF7VsKDbHtPmwpSrj48azJqli1u2KGHvXmUmJ8uvcmWFNGp025MDAAAAAABAwfH3sYv6detRnb6UoiFdainY113Ld55QqL+nKpe0fYmnfC1+vrhmTaVdvChJ2v7aa8pMSlJg7doK79lTZZ5+mqIUAAAAAADAf9zaPaf15pxNcnN11sHTCcrIypYkJadl6pv1B+2Kma/CVHZGhmXB8+Pff6+stDS7DgYAAAAAAIA709drD6p/+yp6qWNVuTibLO2Vigfo4OkEu2LmaypfQI0a2tyvn/wrV5YMQ7tGjJCzu3uefauPG2dXIgAAAAAAACi4TlxIUpWw3NP1vNxdlZSaYVfMfBWmar73ng5Nn67kY8ckSZmJicpm1BQAAAAAAMBdI8DbrFMXUxTq72nVvuvYRRUJ8LzOXjeWr8KUOThYFQYNkiQtb9ZMNSZOlFtAgF0HBBzh+S/WOTqFf83kJxo7OgUAAAAAwH9QuxolNWXJbr1yX1WZZNKFxFTtORGnz5fv0aNNytgVM1+FqcU1a6rF8uUyBwYquH59Obm62nUwAAAAAAAA3Jm6NyqtbMPQa7M3KS0jS69+uVGuLk7qWr+UOteNsCtmvgpTOYufmwMDdfz771Vh0CC5eHvbdUAAAAAAAADceUwmkx5pUlYPNSytUxeTdTk9S2Eh3vJwy1d5KU8sfg4AAAAAAICbmrhwh55pW0meZheFhfhY2lPTM/Xx4t165b5qNsd0yk+nmu+9p8LNmyszJUUymZSZmKiM+Pg8HwAAAAAAAPjvWb7zhNIzs3K1p2Vma/nOk3bFvK2LnxtZuZMDAAAAAADAnSs5LUMyJMOQUtIy5eryv3FO2dnSnwfOyd/Lza7YNk8CbL16da62pMOHdWzePJ344Qfd88cfdiUCAAAAAACAgqfL+KUymSSTSer7yao8epj0WLOydsW2e3WqzMuXdeqXX3R8/nzFbdsmvypVVOrxx+0NBwAAAAAAgAJofFR9GYb02uw/NPShWvLxcLVsc3F2UmE/DwX55L0W+c3YXJiK27ZNx+bN06lff5VH0aJKiolRgzlzFFSnjl0JAAAAAAAAoOCqGhYkSfqyfwsV8vOQyWS6bbHzXZiK+eILHVuwQJmJiSrWsaMafvON/CpU0KLy5eXm739LSRyePVsxn3+utPPn5VuhgioPG6aAanmv5J64f7/2Tpqk+F27dPnkSVUaMkSl+vS5pZgAAAAAAADI7dDZBIUX8pGTyaTktEwdPpd43b6lCvvaHD/fhak948er9FNPKfKll2Rydrb5QNdzctEiRY8ZoyojRyqgWjUdmjFDm3r3Votly2QODs7VPys1VV4lSqhou3baPXr0bYkJAAAAAACA3J79bK2+ebm1/L3MevaztTKZriyCfi2TSfptSAeb4+e7MFX+pZd0fMECnfjxRxXr1EnF779fvuXL23zAax2aPl0lu3dXya5dJUlVR43SuVWrdGzBApXt1y9Xf/+qVeVftaokac+ECbclJgAAAAAAAHL7sn8L+Xm6Wf6+3fJdmCr7zDMq+8wzit20Scfnz9e6Ll3kFRYmGYYy4uPtOnh2errid+1SmauKRSYnJwU3bKi4bdsKTEwAAAAAAIC7UWF/zzz/vl2cbN0huF491Xj3Xd3zxx8Ke/RR+VWurA2PPKJ1XbsqZto0m2Klx8XJyMrKNb3OHBystPPnbU3N7phpaWlKSEiwPBITrz9fEgAAAAAAALeHzYWpHC7e3gp/5BE1+f57Nf35Z/lXq6aDn356O3P714wdO1Z+fn6WR8WKFR2dEgAAAAAAwH+e3YWpq/mWL6/KQ4eqzYYNNu3nFhAgk7Oz0mJjrdrTYmNlDgmxKxd7Yg4ePFjx8fGWR3R0tF3HBgAAAAAAQP7dlsKUJZirq2393dzkV7myYq8qaBnZ2YrduFEBNWrYl4MdMc1ms3x9fS0PHx8fu44NAAAAAACA/Mv34uf/lFKPP67tAwfKv0oV+VerpkMzZigrJcVyR71tr7wi99BQVRg4UNKVxc0TDx688ndGhlLPnFF8dLRcPD3lFR6er5gAAAAAAACwTVZ2to6cS1JccpokKcDLrLAQb7k42z/uyeGFqWIdOyr94kXtmzRJabGx8q1QQfVmzLAsXn759GnJ6X8nmHrunNZ06mR5HvPFF4r54gsF1aunhnPn5ismAAAAAAAA8ifbMDRr1X79vOWIklMzrbZ5ubvovtrheqx5OTmZTDbHdnhhSpIioqIUERWV57acYlMOz+LF1Skm5pZiAgAAAAAAIH+mr9irpTtO6PGWkapVOkQBXmZJUlxymv46FKsvV+1TRla2nmhdwebYdo21Sj56VHsnTtTWF16wLDJ+dtUqJe7fb084AAAAAAAAFFDLd57UoPurq0OtMIX6e8rs6iyzq7NC/T3VvmZJDexcXct2nrArts2FqdhNm7SqfXvF7dihM0uXKjMlRZKUsHev9n3wgV1JAAAAAAAAoGBKSc9UkLf5utsDvc1KTc+yK7bNham9EyYo8uWX1WDWLJmuugtfcIMGitu+3a4kAAAAAAAAUDBVCwvU58v3KD4lPde2+JR0TVuxV1XDg+yKbfMaUwn79qnm++/najcHBSn94kW7kgAAAAAAAEDB1L99FQ39erN6vL9cEYV85P//a0xdSk7T4XOJKhnsrREP17Erts2FKVdfX6WeOyfPEiWs2uN375Z7aKhdSQAAAAAAAKBgKuTnoSlPN9HWmPPac+KS4pLTJEnli/qrd4vyqlU6xK478kl2FKaKduigPePHq9bkyTKZTFJ2ti5u2aLod95R8QcesCsJAAAAAAAAFFxOJpPqlCmkOmUK3d64tu5Q4dVX5V2qlJY3bqzM5GStvPdere/RQwE1a6rcc8/d1uQAAAAAAABQsKWmZ+rvoxfs2tfmEVNObm6qNnasyvbvr8R9+5SVkiLfihXlHRFhVwIAAAAAAAC4c528mKJBs//Qb0M62LyvzYWpC1u2KKh2bXkWLSrPokVtPiAAAAAAAAAg2VGY2tizp9wLF1axTp1UvHNn+ZQt+0/kBQAAAAAAgAKgy4SlN9yebRh2x7a5MNVmwwadWrRIJ3/+WQc//VS+kZEqdt99KtapkzyKFLE7EQAAAAAAABQ8GVnZ6lirpCIK+ea5/Wz8Zc1Zs9+u2DYXpsyBgYqIilJEVJRSjh/XiYULdeL777X33XcVWKeOGs6ZY1ciAAAAAAAAKHhKF/ZViK+H2lQrnuf2mDMJ/15h6mqeJUqobL9+8qtQQXvff18X/vzzVsIBAAAAAADADn8fvaD5Gw/pwOl4XUxK07CHaqlhZKhl+7s/7dCynSes9qlVOkRjHql709h1yxZScmrGdbf7eLiqVdW8i1Y3Y3dh6uKWLTqxcKFO//abstPTVbh1a1V49VV7wwEAAAAAAMBOqRlZKlXYV22rl9CI+Vvz7FO7dIheua+q5bmrs3O+YvdoXOaG2wv5eejV+6rlP9mr2FyY2jNhgk4uWqS0c+cU3KiRKg8dqsJt2sjFw8OuBAAAAAAAAHBr6pQppDplCt2wj6uzkwK93f+ljPLH5sLUhT//VJknn1SR9u1lDgz8J3ICAAAAAACApMTERCUkJFiem81mmc1mu2LtPHpB3SYuk4+7q6pFBKl38/Ly9XS74T57TsSpQvGAfMVPzcjSmbgUhRfyyXdONhemGs+fb+suAAAAAAAAsEPFihWtng8bNkzDhw+3OU7t0iFqFBmqUH8PnY5L0YyV+/Tm139qUp9GcnYyXXe/8T9tVxF/T91bo6TqlgmRu1vuUtLR84la8fdJLdtxQo+3jLz9hakzy5erULNmcnJ11Znly2/YN7R163wfHAAAAAAAANcXHR2tYsWKWZ7bO1qqeeWilr8jCvsqorCvek9eqZ1HL6hGRPB19/u8XzMt2npUX67cp3d+2KZigV4K8nGXm4uTklIzdDw2SZfTs9QoMlRjHqmriMK+NuWVr8LU5n79dM8ff8gcHKzN/fpdv6PJpE4HDtiUAAAAAAAAAPLm4+MjX1/bij35USTAU36ebjp1MfmGhSkXZyfdXzdC99eN0P5Tl7TreJzOxV9W2v8vtv5AvQhVCw+Sr8eNpwReN35+OnU6eDDPvwEAAAAAAHDnOZ9wWQkp6TYthl6uqL/KFfW/rXk42brD8e+/V1ZaWq727PR0Hf/++9uSFAAAAAAAAPLvcnqmYs7EK+ZMvCTpzKUUxZyJ17n4y7qcnqnPl+/RnhNxOnMpRdsOx2r4t1tUNNBLtUpff7TUv8Hmxc+3v/aaCjVtKudr5jRmJidr+2uvqcSDD9625AAAAAAAAHBz+0/Fa9DsPyzPpy7bI0lqU7W4+revrMNnE7Rsxwklp2YoyMddNUsFq1fz8nJzcXZUypLsKEzJMCRT7tXaL585I1ef/K+6DgAAAAAAgNujWniQlgztcN3tYx6t9y9mk3/5Lkyt7tTpSkHKZNLGxx6TyeWqXbOylHLihEKaNv0ncgQAAAAAAMB/UL4LU6Ft2kiSEqKjVahJEzl7eVm2Obm6yrNYMRW5997bnyEAAAAAAAAKrKTUDK34+6Q61wm3ed98F6bKDxggSfIsVkxFO3bMtcYUAAAAAAAA7h7bDsdq8bbj2rDvjMyuzv9sYSpHiS5dbD4IAAAAAAAA7nzn4i9r6Y4TWrrjuM7HX1azSkX11kO1VCPCvrv72VyYMrKydGj6dJ369VddPnVK2RkZVtvv/esvuxIBAAAAAABAwZOZla0N+85q8bZj2nXsomqXDtGTrSpo7A/b1KNxGYWF2H8zPCdbd9j34YeKmT5dRTt0UEZioko9/riKtG0rOTmp3P9P9wMAAAAAAMB/wyOTVuinzUfUuEIRzXmxtd7qVltNKha5LbFtHjF1cuFCVRszRoVbtNC+Dz9UsU6d5BUWJp/y5XVp+/bbkhQAAAAAAAAKhqxsQ6b//9vJ5iFON2ZzYSrt/Hn5lC9/ZWdPT2UkJkqSCrdsqX3vv397swMAAAAAAIBDff1SK63bc0aLtx/Xp0t2q3aZQmpVpZilWHUrbK5zuYeGKu3cOUmSZ8mSOr9unSTp0s6dcnJzuw0pAQAAAAAAoKBwc3FWyyrFNP6x+vr06aYqGeytT5bsVla2oa/XHdTWQ+eVlW3YFdvmEVNF7rlHsRs2KKB6dUVERWnbK6/o+Lx5unz6tEr16WNXEgAAAAAAACj4igZ6qXeL8opqXk5bY85r8bbjGvbNFnm4OWv+q/fYHM/mwlSFQYMsfxfr2FEeRYsqbts2eYWHK7RVK5sTAAAAAAAAwJ3FyWRSnTKFVKdMIV1KTtOKv0/aFcfmwtS1AmvWVGDNmrcaBgAAAAAAAHcgfy+zutQvZde+NhemzixfnvcGk0nOZrO8wsLkWaKEXckAAAAAAACgYIn66PebLnRuMpk08/kWNse2uTC1uV8/yWSSjGsWtcppM5kUWLu26nz6qdz8/GxOCAAAAAAAAAXHA3UjrrvtbPxl/br1qDKysu2KbXNhqv6XX2rve+8p8uWXFVCtmiQpbscO7Xv/fZV97jm5+vho55Ahih47VtXfeceupAAAAAAAAFAwPFAvd2Eq4XK65q49qEVbjiqymL/6toq0K7bNhandI0eq6ujRCqxVy9IW0qiRnM1m7RgyRC0WL1aloUO1/bXX7EoIAAAAAAAABVNaRpa+++OQvvvjkAr5eeqth2qpbtlCdsezuTCVfOyYXLy9cwfy8VHKsWOSJK/wcKXHxdmdFAAAAAAAAAqOrGxDv207pjlrDsjVxUnPtK2kVlWKyWS62epTN2ZzYcqvcmVFv/OOarz7rsxBQZKktAsXFP3OO/KvWlWSlHzkiDyKFLmlxAAAAAAAAOB4q3ef0per9ispNUM9GpdRx9phcnV2ui2xbS5MVX/nHf359NNa3rix3ENDJUmpZ87Is0QJ1Zk6VZKUmZysss89d1sSBAAAAAAAgOOM/X6b3Fyd1aJSUZ2Lv6zpK/bm2e/peyraHNvmwpR3qVJqsWSJzq9dq6TDhy1tIY0by+R0pVpW5J57bE4EAAAAAAAABU+VsEBJ0qm45Ov2sXdKn82FKUkyOTmpULNmCqpfX05ubrc8nxAAAAAAAAAF04SoBv9YbJsnBBrZ2dr/0Uda1rChfqtSRSnHj0uS9r73no7Nm3fbEwQAAAAAAEDBsvv4RaVnZt1yHJsLUwc+/ljHv/9eFV57TSZXV0u7T7lyFKYAAAAAAADuAkPmbtaFxLRbjmNzYer499+r6qhRKt65s0zOzpZ23woVlBQTc8sJAQAAAAAAoGAzZNyWODYXplLPnpVXeHjuDdnZys7MtCuJw7Nna3nTpvqlQgWtffBBxe3YccP+p379Vb+3aaNfKlTQqnbtdHblSqvtmcnJ+nv4cC1r1Ei/VKyolW3b6sjcuXblBgAAAAAAgH+GzYUpnzJldHHz5lztpxcvll9F228LeHLRIkWPGaNyAwao6cKF8o2M1KbevZUWG5tn/4tbt+qvF19UyYceUtOff1Zomzba/MwzSti3z9Jn9+jROrd6tWpMnKgWS5eqVO/e2jV8uM4sX25zfgAAAAAAALA2oH0VBXi53XIcmwtT5fr319/Dh+vg1KlSdrbOLFmiHYMH68Ann6hc//42J3Bo+nSV7N5dJbt2lU/Zsqo6apScPTx0bMGCPPsfnjlTIU2bqsxTT8mnTBlFvvyy/CpV0pHZsy194v76SyUefFDB9evLs3hxhfXoId/ISF26yUgsAAAAAAAA3FzLKsUUn5Kuo+cTlW3YP63P5sJUaJs2qvv55zq/fr2cPTy0d9IkJcbEqM5nnymkcWObYmWnpyt+1y4FN2xoaTM5OSm4YUPFbduW5z4Xt21TSKNGVm2FmjSx6h9Qs6bOrFihy2fOyDAMxW7cqKQjRxTSpEmeMdPS0pSQkGB5JCYm2nQeAAAAAAAA/1VLth/Xd38csmqbtGinek9eqX5T1+jpT9foXPxlu2K72LNTUJ06ajBrll0HvFp6XJyMrCyZg4Ot2s3BwUo6dCjPfdJiY2UOCsrVP/X8ecvzysOGaeebb2p5o0YyubjI5OSkqqNHK6hu3Txjjh07Vm+//fYtng0AAAAAAMB/z69/HVP7miUtzzcfPKelO05oYOfqKhHsrY8X79KcNQf0UqeqNse2ecTUneDIrFmK275ddT77TE1/+kkVBw/W38OH6/z69Xn2Hzx4sOLj4y2P6OjofzljAAAAAACAgunkxWSVK+Jneb5x/1k1KFdYLasUU9kifurTIlLbjuS9VvjN5HvE1PJmzWQymW7cyWRSq2vukHcjbgEBMjk751roPC02VuaQkDz3MQcHK+3ChVz93f+/f1ZqqvZMnKg6U6aocIsWkiTfyEgl7NmjmM8/zzUNUJLMZrPMZrPleUJCQr7PAQAAAAAA4L8sPSNLnub/lZCij8fp3holLM+LBHgqLinNrtj5LkyV6t37uttSTp7U0a+/VnZ6uk0Hd3Jzk1/lyordsEFF7rlHkmRkZyt240aFP/ZYnvsE1qih2A0bVKpPH0vb+XXrFFCjhiQpOyNDRkaGdG0RzclJxi0sxgUAAAAAAHA3KuTnoQOn41XY3/P/FzxPUqUSgZbtF5PS5GV2tSt2/gtTVxWCcqRfuqT9kyfr6Jw58q9eXRUHDbI5gVKPP67tAwfKv0oV+VerpkMzZigrJUUlu3aVJG175RW5h4aqwsCBkqSI3r214ZFHFPPFFyrUooVOLVqkS7t2qero0ZIkVx8fBdWrpz3vvCNnd3d5FiumC5s26cQPP6jSm2/anB8AAAAAAMDdrE214vp48W4dPZ+k7UdiVSLYS2WvmtoXfSJOYYW87Ypt1+LnWampipk2TYe++EIexYqp9iefWKbN2apYx45Kv3hR+yZNUlpsrHwrVFC9GTMsC6JfPn1acvrfUliBtWqp5vvva+9772nvxInyCgtTnSlT5Fu+vKVPzQ8+0N4JE7Tt5ZeVfumSPIoVU+QrryjskUfsyhEAAAAAAOBu9VDD0krNyNL6vWcU4G3WkC41rbZHH7+o5pWK2hXbpsKUkZWlo99+q/0ffigns1mV3npLxe+//+ZrT91ERFSUIqKi8tzWcO7cXG1F27dX0fbtrxvPPSRE1cePv6WcAAAAAAAAIDmZTOrVvLx6NS+f5/YhXWspK9u+5ZPyXZg69csv2vvee8pISFDZZ59V+KOPysnNza6DAgAAAAAA4M534kKSFm87rhV/n9TXL7W2ef98F6a2vvCCnN3dVaxTJ10+dUp7JkzIsx/rOAEAAAAAAPx3pWZkafXuU1qy/bj2nLikckX99GC9CLti5bswFVS3riQp+ejR63e6xSl9AAAAAAAAKJj2nIjT4m3HtWbPaRXy9dCx2CSNj6qvKiUDb77zdeS7MJXXWk8AAAAAAAD4b1uw8ZCWbD+ulLRMNa9cVO9GNVDpUF+1H/2rfD1cbym2XXflAwAAAAAAwN1h2oq96tawlKKal5ez0+2dLUdhCgAAAAAAANfVq3k5Ld1xQiv+PqnmlYqqddXiCi/kc1tiU5gCAAAAAADAdT3cuIweblxGO49e0JLtxzVg+noVDfCUYUhJqRm3FJvCFAAAAAAAAG6qaliQqoYF6bl7M7Vy10kt2X5Cr375h8oX81OTCkXUpX4pm2NSmAIAAAAAAEC+eZpd1KFWmDrUCtPhswlavP24vl0f888VphL27s13QN/ISJuTAAAAAAAAwJ0norCvnmlbSU+2rmDX/vkqTK3u2FEymSTDyLtDzjaTSZ0OHLArEQAAAAAAABQ8y3acuGkfk0lqXbW4zbHzVZhqtXq1zYEBAAAAAABw5/t06e4bbDUpLSNLWdnZ/1xhyrNYMZsDAwAAAAAA4M733cC2ebZfSEzVV2sOaOn246pRKsSu2HYvfp544IAunzql7Azr2wKGtm5tb0gAAAAAAAAUcClpmZq3IUY//nlYYSE+Gv1oXVUPD7Yrls2FqeRjx7TlmWeUsG+f9bpTJpMkscYUAAAAAADAf1BmVrZ+2nxE36w7KF8PN73SqZqaVCxySzFtLkztHjlSHsWLq/7s2VrRvLmafP+90i9dUvSYMao4ePAtJQMAAAAAAICCxTAMLd95UrNW71dWdrb6tIxU2+ol5OxkuuXYNhemLm7bpoZffSVzYKBMJpNMTk4Kql1bFV59VbtGjFCzn3++5aQAAAAAAABQMPSbulanL6Woc51wPVAvXGZXZ6VmZObq52V2tTm27WtMZWXJxctLkuQWGKjUs2flXaqUPIoVU9KhQzaHAwAAAAAAQMF19HyiJGn+hhgt2BiTa7thXFnh6bchHWyObXNhyqdcOcXv2SPPEiUUUK2aDn72mZxcXXX0m2/kVaKEzQkAAAAAAACg4BofVf8fi21zYarsc88p6/JlSVL5F1/Upief1PqHH5ZbQIBqffDBbU8QAAAAAAAAjlM1LOimfRIup9sV2+bCVKGmTS1/e4WHq+WyZUq/dEmufn4ymW590SsAAAAAAADcGbbGnNdv245r04Gz+nlwO5v3t32NqTy4+fvfjjAAAAAAAAAo4M5eStGS7Se0fOcJJaZmqE7pEA3sXN2uWDYXpjJTUnTw008Vu2GD0i5cuLLC1VVarVplVyIAAAAA8F/x/BfrHJ3Cv2byE40dnQKAf0FGVrbW7zmj37YfU/TxONWICNb5hFR98mRjRRT2tTuuzYWpHYMH68Kff6r4/ffLvVAhuw8MAAAAAACAgu/jxbu0atcpFQv0UssqxfTmgzXl6+mm9qN/lZPTrS3rZHNh6tzq1ar3xRcKrF37lg4MAAAAAACAgm/RlmPq1rCUujcqI0/zbVkVysLJ1h1c/fzkyppSAAAAAAAAd4VB91fTvlPx6vH+co3+7i/9sf+ssrKNm++YDzaXuSJfekn7Jk1S9QkT5OLhcVuSAAAAAAAAQMHUonIxtahcTGfiUrRkx3F9vHi30jJ2yjAMHTufpLAQH7tj21yYipk2TSnHjmlpvXryLFZMJldXq+3NFi60OxkAAAAAAAAUTKEBnurVvLyimpXT1kOxWrztuMb9uF2fLo1Wo8hQPXtvJZtj2lyYCm3TxuaDAAAAAAAA4M5lGIZOXUxRRla2SgR7qXbpENUuHaKEy+lavvOklm4/bldcmwtT5QcMsOtAAAAAAAAAuPOciUvRsG+36FhsoiQpyMddbz1US+WK+svXw00P1ovQg/Ui7Ipt91Lql/7+W0kxMZIkn7Jl5VfJ9uFaAAAAAAAAKNg+X75HWdnZGnR/dbm5OGvBxkP64Je/9fGTTW45ts2FqbTYWG194QVd2LRJrr6+kqSMhAQF16+vmh98IHNQ0C0nBQAAAAAAgIJh9/E4DelaU5VLBkqSIov5q+cHK5Sanil3N7vHPEmSnGzdYdfbbyszOVnNFy/WvX/9pXv/+kvNf/tNGUlJ2jVixC0lAwAAAAAAgILlUnKaigV6WZ4H+bjLzcVZl5LTbzm2zWWtc2vWqMHs2fIpU8bS5lO2rKq8/bb+6NXrlhMCAAAAAABAwWEySZfTM+WW9r/xTU4mk1LSM5WclmFp8zK72hzb5sKUYRgyueTezcnFRTIMmxMAAAAAAABAwWUYUt9PVuVqe+7ztZa/TSbptyEdbI5tc2EquEED7Ro5UrUmTZJ74cKSpMtnzmj36NEKbtDA5gQAAAAAAABQcI2Pqv+Pxba5MFVl2DD9+fTTWt6smTyKFJEkXT59Wj7lyqnGxIm3PUEAAAAAAAA4TtWwf+5GdzYXpjyKFlXThQsVu369kmJiJEneZcoopFGj254cAAAAAAAA/rvsuqefyWRSSOPGCmnc+HbnAwAAAAAAgLtEvgpTh2bOVFiPHnI2m3Vo5swb9i3Vu/dtSAsAAAAAAAD/dfkrTM2YoeKdO18pTM2Ycd1+JpOJwhQAAAAAAADyJV+FqdarV+f5NwAAAAAAAGAvm9eY2v/RRyr1xBNy8fCwas9KTVXM55+rXP/+ty05AAAAAAAA3NzfRy9o/sZDOnA6XheT0jTsoVpqGBlq2W4Yhmat3q/F244rKTVDFUsEaEC7KioW5HXDuCPmbcl3Dm91q21z3jYXpvZ9+KHCevTIXZi6fFn7PvyQwhQAAAD+Fc9/sc7RKfxrJj/BTYcAADeWmpGlUoV91bZ6CY2YvzXX9nkbDumnP4/o1c7VFOrvqS9X7dcbczfp82eayc3F+bpxPd1d/8m07bgrn2FIJlOu5oQ9e+Tm738bUgIAAAAAAIAt6pQppDplCuW5zTAM/fjnYfVoUkYNy18ZRTWoczV1f2+5Nuw9q+aVi1437qv3VftH8s2R78LU4ho1rhSkTCatbN3aqjhlZGUpMyVF4Y888o8kCQAAAAAAAPucuXRZF5PSVDMi2NLm5e6qyGL+2nMy7oaFqX9avgtTlYYMkQxD219/XeVeeEGuPj6WbU6urvIoXlyBNWv+I0kCAAAAAADcjRITE5WQkGB5bjabZTabbYpxMSlVkuTvZb2fv5dZF5PSbIq1Nvq0Vkef1vmEy8rMyrba9vGTTWyKJdlQmCrRpYskyaNECQXWqiUnF9tnAV7P4dmzFfP550o7f16+FSqo8rBhCqh2/aFip379VXvff1+XT5yQV3i4KgwapMItWlj1STx4UHvGj9eFTZtkZGXJu0wZ1f7kE3kWdVwVEAAAAAAAwBYVK1a0ej5s2DANHz7cIbn8+OdhzVy5T22qFdcf+8+qTbXiOh2Xov2nLqlT7TC7YjrZukNWSopiN2zI1X5uzRqdXbXK5gROLlqk6DFjVG7AADVduFC+kZHa1Lu30mJj8+x/cetW/fXiiyr50ENq+vPPCm3TRpufeUYJ+/ZZ+iQfPar13bvLu1QpNZw7V81++UXlnn9ezm5uNucHAAAAAADgKNHR0YqPj7c8Bg8ebHOMQG93SdKlZOvRUZeS0xTonf/RVz9vOaoXOlTRc/dWlouzSd0altY7Peupc91wJadl2pyXZEdhas+ECTKysnJvMAztnTDB5gQOTZ+ukt27q2TXrvIpW1ZVR42Ss4eHji1YkGf/wzNnKqRpU5V56in5lCmjyJdfll+lSjoye7alz96JE1WoeXNVfP11+VWqJK+wMIW2bi1zcHCeMQEAAAAAAAoiHx8f+fr6Wh62TuOTpFB/DwV6m7Xt8AVLW3JahvaevKQKxQLyHed8/GVVLH6lv5uLs1L+vxjVukpxrdp9yua8JDsKU8lHjsinbNlc7d6lSyv56FGbYmWnpyt+1y4FN2xoaTM5OSm4YUPFbduW5z4Xt21TSKNGVm2FmjSx9Deys3V21Sp5h4frj969taROHa198EGdXrr0unmkpaUpISHB8khMTLTpPAAAAAAAABzpcnqmYs7EK+ZMvCTpzKUUxZyJ17n4yzKZTLq/boS+XndAG/ed1eGzCZrw4w4F+ZjVMLJwvo8R4G1W4uUMSVIhPw/tPRlnOZZh2Je3zQtFufr4KOXYMXkWL27Vnnz0qJw9PGyKlR4XJyMrK9dIJnNwsJIOHcpzn7TYWJmDgnL1Tz1//sr2CxeUlZysg1OnqvzLL6vCoEE6t2aNtjz7rBrMmaPgevVyxRw7dqzefvttm3IHAAAAAAAoKPafiteg2X9Ynk9dtkeS1KZqcb3auZq6NSyl1IxMffDL30pKzVClkgEa/Uhdubk45/sY1cOD9cf+sypTxE/3VCuuqUujtXbPGR04fUmNIkPtytvmwlTh1q21a9Qo1ZkyRV5hVxa2Sj5yRNFjxii0dWu7kritsq+sCB/aurVKP/64JMmvYkXF/fWXjs6dm2dhavDgwXr55Zctz0+ePJlrcTEAAAAAAICCqlp4kJYM7XDd7SaTSb2al1ev5uXtPsYLHavI+P+hUffVCZevh5uiT8SpQblCal/LvsXPbS5MVXz9dW3q00cr77lH7qFXqmGpZ84osHZtVbRxAS63gACZnJ1zLXSeFhsrc0hInvuYg4OVduFCrv7u/9/fLSBAJhcXeZcpY9XHu3RpXdyyJe+Y19xq8erbMAIAAAAAAEByMpkkk8nyvHnlompeuegtxbRrKl+j+fN1ft06JezZI2d3d/lGRiqobl2bD+7k5ia/ypUVu2GDitxzj6Qra0TFbtyo8Mcey3OfwBo1FLthg0r16WNpO79unQJq1LDE9K9SRUmHD1vtl3z4sDyLFbM5RwAAAAAAAFyReDlDi7cf0/HYJElSyWAf3VO9uHw93OyKZ3NhSroy/KtQkyYq1KSJXQe9WqnHH9f2gQPlX6WK/KtV06EZM5SVkqKSXbtKkra98orcQ0NVYeBASVJE797a8MgjivniCxVq0UKnFi3SpV27VHX0aEvM0k8+qa0vvKCgOnUUXL++zq1Zo7O//64Gc+fecr4AAAAAAAB3o7+PXtCwb7fI0+yickX8JEk/bT6iOWsPaET32qoSFnSTCLnZVZg6v369YjduVPqFCzL+f02nHNXHjbMpVrGOHZV+8aL2TZqktNhY+VaooHozZlgWRL98+rTk9L+bBwbWqqWa77+vve+9p70TJ8orLEx1pkyRb/n/zZEs0ratqo4cqYNTpmjXiBHyLlVKtT/+WEG1a9tzugAAAAAAAHe9yb/tVtOKRdS/fRU5O12Z0peVbWjyb7s0+bfdmtqvqc0xbS5M7fvwQ+3/6CP5V6kic0iITFfNLbRXRFSUIqKi8tzWMI9RTkXbt1fR9u1vGLPkQw+p5EMP3XJuAAAAAAAAkE7FJWtI15qWopQkOTuZ9GC9CC3fecKumDYXpo7Onavq48erxAMP2HVAAAAAAAAA3HnKhPrpeGySSgR7W7Ufj01SqcK+dsW0uTCVnZGhwJo17ToYAAAAAAAA7hyHziZY/u5cN1xTlkbrZFyyKhQLkCTtORmnn7cc1eMtI+2Kb3NhqmS3bjq5cKHK9e9v1wEBAAAAAABwZ3j2s7UymSTD+F/btOV7c/Ub98M2Na9U1Ob4to+YSkvToW++0fkNG+RbvrycXF2ttld6802bkwAAAAAAAEDB82X/Fv9ofJsLUwl798q3QgVJUuL+/dYbb8NC6AAAAAAAACgYCvt7SpIys7L1wS9/69EmZRUa4Hnb4ttcmMrrLnkAAAAAAAD473JxdtK6vWf0aJOytzWu022NBgAAAAAAgP+khuULa8O+M7c1ps0jpjY88sgNp+w1nDPnlhICAAAAAABAwVMs0Etz1h7Q7uNxKlvET+5uzlbb768bYXNMmwtTvhUrWj03MjMVHx2txP37VeL/2rvv+Kiq/P/j70kgE0gPCWmk0JTQg6KAP0JRRJEmUlSKAcUCyLK2FZUqgoo0VwVrAqiIgouILAKCIjYUSQSiQUoIJSCBQBISJu38/uDLrGMCBkRugNfz8ZjHcs8999zPmT2PcXhz751evc66AAAAAAAAAFR+K5L3yMuzqn49cEy/Hjjmss+mCxRMNX7qqXLb02bNUvHx42ddAAAAAAAAACq/eQ92PO9jnrdnTNXq0UN7Fi06X8MBAAAAAACgEioqKdWerDyVlJb+5bHO+oqp0zmyaZPc7PbzNRwAAAAAAAAqkRNFJXplxRatStknSXpreHuFBVTXyyu2KMjHU/2uq3fWY551MPX9Aw+4NhijE4cO6djmzao/YsRZFwAAAAAAAIDKL3HNL9p5MFdTB7XSk+9ucLbH1Q7S21/8emGCqao+Pq4Nbm7yrlNHV44apZpt2551AQAAAAAAAKj8vk47qCd6xSm2VoBstv+1Rwf7KDM7/5zGrHAwdTwjQ9UjI9X8+efP6UQAAAAAAAC4eB077pC/V9nHOJ0oLDn5s3znoMIPP19z/fUqPHzYub3xwQflyMo6t7MCAAAAAADgolI/3F8bfj3o3D6VRa1IzlBsrYBzGrPit/IZ47J58Isv1CA/XzzuHAAAAAAA4NI3uMOVemrBBu3OylNJqdF/NuxSRlaeUvdk64W7Wp/TmBW+YgoAAAAAAACXn/TfciVJjaMC9crQtiopNYqp6aMfd2bJv7qHZg5uo/phfuc0dsWvmLLZ5PJkK0k22zneQAgAAAAAAICLwv2vrtMV4f66KS5S7RuF659dm563sc/qVr7kxx6Tm4eHJKnU4dBPY8bIvVo1l24tZ88+b8UBAAAAAADAWlPvaq2VyXv0+qqf9erKVLWNDVPnuEg1iQr8y2NXOJiK7NXLZbtWjx5/+eQAAAAAAACo3JpEBapJVKCG39RIX6RmalXKXj069xuFB3qpc/NIdWoWoUBvz3Mau8LBVPPnnz+nEwAAAAAAAODi5+lRRZ2bR6pz80jtO3JcK5P36OMf0jXv8zRdXTdYE25vedZj8vBzAAAAAAAAnJWIQC/d8f/q6c629VXNXkUbtv92TuNU/BlTAAAAAAAAuOxt3n1Ynybv1fpfMmWz2RTfMEw3NY88p7EIpgAAAAAAAHBGh3NPaGXKXq1K2av9R46rYWSAHujcSO0ahsnT49zjJYIpAAAAAAAAnNaT727Qpl1Z8qvuoeubRKhz80hFBnmfl7EJpgAAAAAAAHBaVdxseqp3C11bP0TubrbzO/Z5HQ0AAAAAAACXlHP5tb2K4lf5AAAAAAAAYAmCKQAAAAAAAFiCYAoAAAAAAACWIJgCAAAAAACAJQimAAAAAAAAYAmCKQAAAAAAAFiCYAoAAAAAAACWIJgCAAAAAACAJQimAAAAAAAAYAmCKQAAAAAAAFiCYAoAAAAAAACWIJgCAAAAAACAJQimAAAAAAAAYAmCKQAAAAAAAFiCYAoAAAAAAACWIJgCAAAAAACAJQimAAAAAAAAYAmCKQAAAAAAAFiCYAoAAAAAAACWqBTB1K7587U6Pl6fxMbqy169lJ2Scsb++5cv15pOnfRJbKw+v/lmHVy79rR9f3rqKX1ct652Jiae77IBAAAAAADwF1geTO1btkypkyfripEjFb90qXwbNNB3CQlyZGWV2//Ixo36cdQoRfXpo/iPP1Zop076/oEHlJOWVqZv5qefKjs5WZ4hIX/3NAAAAAAAAHCWLA+mdr71lqL69VNU797yqV9fTSdNknu1aspYtKjc/ruSkhQcH696994rn3r11OChh+TXqJHS58936Vdw4IC2TJyoFtOny1alyoWYCgAAAAAAAM6CpcFUaWGhjm3ZoqA2bZxtNjc3BbVpo+xNm8o95simTQq+7jqXtppt27r0N6Wl2vTww6p7zz3yueKKP63D4XAoJyfH+crNzT3HGQEAAAAAAKCiLA2mCrOzZUpKZA8Kcmm3BwXJcehQucc4srJkr1GjTP8Tv+u//dVXZatSRbUTEipUx5QpU+Tn5+d8NWzY8OwmAgAAAAAAgLNm+a1859vRzZu1KylJcc8/L5vNVqFjRo8erWPHjjlfqampf3OVAAAAAAAAsPThSx4BAbK5u5d50LkjK0v24OByj7EHBclx+HCZ/p7/1//IDz/IcfiwVrdt69xvSkq0dfJk7UxM1A3r1pUd026X3W53bufk5JzznAAAAAAAAFAxlgZTbh4e8mvcWFlff62wG2+UdPL5UFnffKOYgQPLPSYwLk5ZX3+tOoMHO9sOrV+vgLg4SVKtnj1dnlklSd8NHqxaPXsqsnfvv2kmAAAAAAAAOFuW/1xdnSFDlPzoo/Jv0kT+zZppZ2KiSvLzFfV/IdKmhx+WZ2ioYh99VJJUOyFBX995p3a88YZqduig/cuW6eiWLWr6zDOSTl6F5REQ4HIOW5UqsgcHy7tOnQs7OQAAAAAAAJyW5cFURNeuKjxyRGkzZ8qRlSXf2Fhdm5jofCB6QWam5Pa/R2EFXnWVWsyYoV+mT9cv06bJKzpaLWfPlu+VV1o1BQAAAAAAAJwDy4MpSao9aJBqDxpU7r42775bpi28SxeFd+lS4fHLe64UAAAAAAAArHXJ/SofAAAAAAAALg4EUwAAAAAAALAEwRQAAAAAAAAsQTAFAAAAAAAASxBMAQAAAAAAwBIEUwAAAAAAALAEwRQAAAAAAAAsQTAFAAAAAAAASxBMAQAAAAAAwBIEUwAAAAAAALAEwRQAAAAAAAAsQTAFAAAAAAAASxBMAQAAAAAAwBIEUwAAAAAAALAEwRQAAAAAAAAsUcXqAgAAAAAAAPDXzP9im95e96tLW60aXnpzWHtrCqoggikAAAAAAIBLQHSwt54dcK1z292t8t8oRzAFAAAAAABwCXB3c1Ogt6fVZZwVgikAAAAAAIBLwL4jx3XHjNXyqOKm2FoBGtKxgWr6VbO6rDMimAIAAAAAAKikcnNzlZOT49y22+2y2+1l+jWI8Ncj3ZupVg0vHclz6O112/Tw3G/06n3xqm6vvPFP5b/ZEAAAAAAA4DLVsGFD+fn5OV9Tpkwpt1/LejUV3zBMdUJ8dXXdYE264xrlnSjSutT9F7jis1N5IzMAAAAAAIDLXGpqqiIiIpzb5V0tVR5vz6qqFeil/Ufy/67SzguCKQAAAAAAgErKx8dHvr6+Z31cQWGx9mfn6/qmFQuyrEIwBQAAAAAAcJF7bVWqWl0Ropp+1XQ494Tmf/Gr3N1sat8o3OrSzohgCgAAAAAA4CKXlXNCUz7cpNyCIvlV91CjyADNHNxG/l5cMQUAAAAAAIC/0RO3tbC6hHPCr/IBAAAAAADAEgRTAAAAAAAAsATBFAAAAAAAACxBMAUAAAAAAABLEEwBAAAAAADAEgRTAAAAAAAAsATBFAAAAAAAACxBMAUAAAAAAABLEEwBAAAAAADAEgRTAAAAAAAAsATBFAAAAAAAACxBMAUAAAAAAABLEEwBAAAAAADAEgRTAAAAAAAAsATBFAAAAAAAACxBMAUAAAAAAABLEEwBAAAAAADAEgRTAAAAAAAAsATBFAAAAAAAACxBMAUAAAAAAABLVIpgatf8+VodH69PYmP1Za9eyk5JOWP//cuXa02nTvokNlaf33yzDq5d69xXWlSk1Oee0+c336zljRtrZevW2vTwwzpx8ODfPQ0AAAAAAACcBcuDqX3Llil18mRdMXKk4pculW+DBvouIUGOrKxy+x/ZuFE/jhqlqD59FP/xxwrt1EnfP/CActLSJEklJ07o2NatumLECMUvXaqWr7yivF27tOHeey/ktAAAAAAAAPAnLA+mdr71lqL69VNU797yqV9fTSdNknu1aspYtKjc/ruSkhQcH696994rn3r11OChh+TXqJHS58+XJFX18VHrefMUfsst8q5TRwFxcWoyfryObdmi/P37L+TUAAAAAAAAcAaWBlOlhYU6tmWLgtq0cbbZ3NwU1KaNsjdtKveYI5s2Kfi661zaarZte9r+klSUmyvZbKrq43N+CgcAAAAAAMBfVsXKkxdmZ8uUlMgeFOTSbg8KUt7OneUe48jKkr1GjTL9Txw6VG7/EodDPz/3nCK6dTttMOVwOORwOJzbubm5ZzMNAAAAAAAAnAPLb+X7O5UWFWnjgw/KSGoyceJp+02ZMkV+fn7OV8OGDS9ckQAAAAAAAJcpS4Mpj4AA2dzdyzzo3JGVJXtwcLnH2IOC5Dh8uEx/zz/0PxVKFezbp9Zz557xNr7Ro0fr2LFjzldqauo5zggAAAAAAAAVZWkw5ebhIb/GjZX19dfONlNaqqxvvlFAXFy5xwTGxbn0l6RD69e79D8VSh1PT1erefPkERBwxjrsdrt8fX2dLx+eRQUAAAAAAPC3s/xWvjpDhihj4ULtWbxYudu366cxY1SSn6+o3r0lSZseflg/T53q7F87IUG/rVunHW+8odwdO5Q2a5aObtmimIEDJZ0MpX4YMUJHN29W3IwZMqWlOnHokE4cOqTSwkJL5ggAAAAAAICyLH34uSRFdO2qwiNHlDZzphxZWfKNjdW1iYnOB6IXZGZKbv/LzwKvukotZszQL9On65dp0+QVHa2Ws2fL98orJUknDh7UwdWrJUnrunZ1OVfrd95RUKtWF2hmAAAAAAAAOBPLgylJqj1okGoPGlTuvjbvvlumLbxLF4V36VJu/+q1aqnbjh3ntT4AAAAAAACcf5bfygcAAAAAAIDLE8EUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEsQTAEAAAAAAMASBFMAAAAAAACwBMEUAAAAAAAALEEwBQAAAAAAAEtUsboASdo1f752vP66HIcOyTc2Vo3HjVNAs2an7b9/+XL9MmOGCvbulVdMjGIfe0whHTo49xtjlDZzpjIWLlRRTo4Cr7pKTSZOlHft2hdiOgAAAAAAAJZY+n26Fn2zU0fyHKoT4qthNzVSgwh/q8s6LcuvmNq3bJlSJ0/WFSNHKn7pUvk2aKDvEhLkyMoqt/+RjRv146hRiurTR/Eff6zQTp30/QMPKCctzdlnx2uvadfcuWr69NNq++GHcq9eXd8NHqwSh+NCTQsAAAAAAOCC+nzrfr226mf1j6+vl4f+P9UJ8dGT736no8crbx5ieTC18623FNWvn6J695ZP/fpqOmmS3KtVU8aiReX235WUpOD4eNW791751KunBg89JL9GjZQ+f76kk1dL7UxM1BXDhyu0Uyf5NmiguBde0ImDB3Vg5coLOTUAAAAAAIAL5sNvd+mmuEh1bh6p6GAfjbyliexV3fVp8h6rSzstS4Op0sJCHduyRUFt2jjbbG5uCmrTRtmbNpV7zJFNmxR83XUubTXbtnX2z9+zR45DhxT0uz5VfXzk37z5accEAAAAAAC4mBWVlOrXzGNqUTvI2eZmsymudpBS9x61rrA/Yekzpgqzs2VKSmQPCnJptwcFKW/nznKPcWRlyV6jRpn+Jw4dOrn///63vDFP7SszpsMhx+9u8zt27JgkKTMz8yxmU7kcys+3uoQLZu/evX/apyAv5wJUUjlU5P2oDC6XNcr6dMX6rFxYn65Yn5UL69MV67Py+bP/T1iflQ/r839YnxeHU5nEsWPH5Ovr62y32+2y2+1l+ufkF6rUGPl7u+4L8LJrT9bxv7fYv6BSPPzcalOmTNGECRPKtF9zzTUWVIOzFhlpdQWVylv/tLoCuGB9umB9VjKsTxesz0qG9emC9VkJsUadWJ+VEOvT6VJYn40bN3bZHjdunMaPH29NMX8DS4Mpj4AA2dzdyzzo3JGVJXtwcLnH2IOC5Dh8uEx/z//rf+o4R1aWPGvWdOnjFxtb7pijR4/WQw895NwuLi7Wzz//rMjISLm5Wf4YLpyF3NxcNWzYUKmpqfLx8bG6HKAM1igqM9YnKjPWJyoz1icqM9bnxau0tFQZGRlq2LChqlT5X3xT3tVSkuRb3UNuNpuO5rk+6Dz7uEMB3uUfUxlYGky5eXjIr3FjZX39tcJuvFGSZEpLlfXNN4oZOLDcYwLj4pT19deqM3iws+3Q+vUKiIuTJFWPjJQ9OFhZX38tv4YNJUlFubk6mpysmDvvLHfM8i6Du+4Pz7HCxSEn5+QlqRERES6XOgKVBWsUlRnrE5UZ6xOVGesTlRnr8+IWFRVV4b5V3d1UP8xPm9Kz1KZBqCSp1Bgl7zqs7i2j/64S/zLLLweqM2SIMhYu1J7Fi5W7fbt+GjNGJfn5iurdW5K06eGH9fPUqc7+tRMS9Nu6ddrxxhvK3bFDabNm6eiWLc4gy2azqc7gwfr15Zd1YPVq5aSladMjj8gzJESh/xd+AQAAAAAAXGp6taqt//64R6tS9irjUK7+vXyLThQV68Zmlff2TsufMRXRtasKjxxR2syZcmRlyTc2VtcmJjofXl6QmSn97na6wKuuUosZM/TL9On6Zdo0eUVHq+Xs2fK98kpnn7r33qvi/Hz99OSTKsrJUeDVV+vaxES5n+ZyNwAAAAAAgItd+0bhOpZfqHlfbFN2nkN1Qnz1zJ3XcCvfn6k9aJBqDxpU7r42775bpi28SxeFd+ly2vFsNpsa/POfavDPS+ApZzgrdrtd48aNO+09t4DVWKOozFifqMxYn6jMWJ+ozFifl58eLWPUo2WM1WVUmM0YY6wuAgAAAAAAAJcfy58xBQAAAAAAgMsTwRQAAAAAAAAsQTAFAAAAAAAASxBM4ZJRVFSkESNGKCAgQIGBgXrwwQdVXFxsdVmAJOmll17S1VdfLbvdrp49e1pdDuDC4XBo6NChql27tnx8fNSgQQO99dZbVpcFOD344IOKjIyUr6+vIiIiNGrUKBUWFlpdFuBUUFCgevXqyd/f3+pSAKeEhAR5eHjI29vb+frmm2+sLgsog2AKl4xJkyZp/fr1Sk1N1datW/Xll19q8uTJVpcFSJLCw8P11FNPaejQoVaXApRRXFyssLAwrV69Wjk5OUpKStLDDz+slStXWl0aIEkaNmyYfvnlF+Xk5CglJUUpKSl6/vnnrS4LcBo7dqyio6OtLgMoY9iwYcrLy3O+WrdubXVJQBkEU7hkvPXWW3rqqacUFhamsLAwPfnkk3rzzTetLguQJPXq1Us9e/ZUUFCQ1aUAZXh5eWnixImqW7eubDabWrVqpQ4dOmj9+vVWlwZIkmJjY+Xl5SVJMsbIzc1Nv/76q8VVASdt3LhRK1as0L/+9S+rSwGAixLBFC4J2dnZ2rt3r5o3b+5sa968uTIyMnTs2DHrCgOAi9CJEye0YcMGNW3a1OpSAKdnn31W3t7eqlmzplJSUvTggw9aXRKg4uJiDR06VC+//LI8PDysLgcoY968eQoMDFSjRo00bdo0lZaWWl0SUAbBFC4JeXl5kuRyX/+pP+fm5lpQEQBcnIwxuueee1S/fn316tXL6nIAp8cff1x5eXlKTU3V/fffr9DQUKtLAjR16lTFxcUpPj7e6lKAMkaOHKm0tDQdOnRIb775pmbNmqVZs2ZZXRZQBsEULgne3t6S5HJ11Kk/+/j4WFITAFxsjDEaNmyY0tLStGTJErm58TUBlU9sbKyaNWumhIQEq0vBZW779u2aM2eOpk6danUpQLlatGih4OBgubu7q1WrVnr88ce1cOFCq8sCyuAbJy4JAQEBqlWrlpKTk51tycnJioyMlJ+fn3WFAcBFwhij4cOH67vvvtPKlSv57ESlVlRUxDOmYLn169fr4MGDuuKKKxQUFKQePXooJydHQUFB+u6776wuDyiDf3BCZcXKxCVj8ODBeuaZZ3TgwAEdOHBAkydP1j333GN1WYCkk8+gOHHihIqLi1VaWqoTJ07wU+eoVEaMGKGvvvpKq1atUkBAgNXlAE55eXlKTEzU0aNHZYzR5s2bNWnSJHXu3Nnq0nCZ69u3r7Zv367k5GQlJyfrjTfekI+Pj5KTkxUXF2d1eYDef/995eTkyBijH374Qc8++6xuu+02q8sCyrAZY4zVRQDnQ1FRkUaNGqV3331XkjRgwADNmDFDVapUsbgyQBo/frwmTJjg0tauXTt9/vnn1hQE/M7u3bsVExMju93u8pk5YMAAzZkzx8LKAOn48ePq2bOnfvzxRzkcDtWsWVO33XabJkyYoOrVq1tdHuD0+eefq2fPnjp69KjVpQCSpPj4eP30008qLi5WRESE7r77bj3yyCNcOYVKh2AKAAAAAAAAliAqBQAAAAAAgCUIpgAAAAAAAGAJgikAAAAAAABYgmAKAAAAAAAAliCYAoAL5Oabb9Yrr7wi6eQv9/j7+5+2b3Jysmw22wWqDJebUaNGKSEhweoyAAAAAIIpALhQ/vvf/2rYsGFWlwH87caPH6+ePXtaXQYqsaSkJDVv3tzqMgDgomaz2ZScnGx1GcBfRjAFAAAqrKioyOoScJlgreFSUFxcbHUJuIzxOYqLBcEUKq2YmBg9//zzatWqlXx8fNSuXTvt2bNHkrR9+3Z17txZgYGBqlu3rmbOnOk87tS/wj799NOqWbOmQkJCXPZL0nvvvaemTZvK399fLVu21Ndff30BZ4ZLSUxMjKZMmaKWLVvKy8tLN998s44cOaJhw4bJ399f9evXd66v9u3bl1mLpxw9elR9+/aVv7+/GjRooHXr1l3AWaAyOpu1lZubq3vvvVdhYWEKCwvT/fffr+PHjzvHWrdunZo0aSJvb2/16tVLubm5LufasWOHunXrpuDgYEVHR2vSpEkqLS2V9L/P1HHjxik0NFS333678vLy1KNHD9WsWVN+fn6Kj49XSkqKJGnJkiWaPHmyli1bJm9vb3l7e0uSjDF68cUX1aBBA/n7+6t9+/b6+eefL8RbCYtNnz5dUVFR8vHxUUxMjKZOnar7779fmzdvdq6RjIwMjR8/Xl27dtUDDzygwMBAPf744yoqKtLo0aMVFRWl4OBg9evXT4cOHXKObbPZNGfOHDVu3Fi+vr7q3r27jh075tx/au37+PioV69euvvuu7mNFX9qxowZ6tixo0vbwoUL1aBBA0ln/h7Zvn17PfbYY7rxxhvl5eWll19+WT4+PsrLy3P22bdvn+x2u/bv339hJoRKLy8vTyNGjFBUVJRq1qypQYMG6dixY3r00UfVrl0753+TFy1apNDQUP3222+65pprJElt2rSRt7e3Jk+erPT0dNlsNiUmJqpevXqqVauWJOnHH39Uhw4dFBgYqHr16un11193nvvUZ+99990nPz8/1a5dW59//rmWLFmievXqKSAgQE8++aRLvatXr9Y111wjf39/NWrUSEuXLr1A7xQuWQaopKKjo02TJk3Mzp07TUFBgbn55pvNXXfdZYqKisyVV15pHn30UVNQUGBSUlJMWFiYeeedd4wxxiQmJpoqVaqYF154wRQWFpq1a9eaKlWqmO3btxtjjPnkk09MRESE2bhxoykpKTGLFy82gYGBJisry8rp4iIVHR1tmjVrZjIyMszRo0dNo0aNTP369c3ixYtNcXGxGTt2rGnSpIkxxph27dqZGTNmGGOMWbt2rfHz83OOM3DgQNOpUyeTnZ1t9u3bZ6666irDR/Tl7WzW1uDBg02HDh1MVlaWOXTokGnXrp0ZOnSoMcaYI0eOGD8/PzNnzhxTVFRkli5dajw8PMxdd91ljDHm+PHjJjo62syYMcM4HA6ze/du06hRI/PGG28YY05+prq7u5uJEycah8Nhjh8/bo4dO2bee+89k5eXZwoKCszIkSPNFVdcYUpLS40xxowbN8706NHDZT4vv/yyadq0qdm2bZspKioys2bNMnXr1jUOh+PCvKGwRFpamqlWrZr5+eefjTHGHDhwwKSkpJjExETTrFkzl77jxo0z7u7uJjEx0RQVFZnjx4+bCRMmmMaNG5vdu3eb3Nxc069fP9OpUyfnMZJMhw4dzMGDB012draJi4sz48aNM8b8b+2/+eabpqioyHzyySfGbrc71z5wOgcOHDBVq1Y1GRkZzrZbbrnFTJo06U+/R7Zr184EBweb7777zpSWlpr8/Hxz1VVXmcTE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992nuXPnat68edq0aZNee+0157EhISHasWPHGccfOHCg1qxZo8WLF6uoqEhFRUVKTk7W999/f0713nfffUpMTNTatWtVUlIih8Ohb775hh80wV9CMIWLTtWqVbVs2TJt3LhRoaGh6t69ux566CHdeeedFTq+W7duevbZZzV06FAFBASodu3amjVrlvPXLgAr/Pvf/5a3t7eio6PVsWNH57/MAhUxa9YsxcTEqGHDhmrUqJHq1aun6dOnS5ICAwP10UcfadasWfL399cbb7yh/v37O4/19vbW6tWr9dlnnykmJkY1atTQnXfe6XIr3x899NBDcnd3V0hIiBo3bqzWrVu77O/Tp498fX0VHBwsf39/SdKIESOUkJCgXr16ydfXV7GxsXr33XfP/5uBSqWwsFBjxoxRSEiIatSooTVr1igpKUkdO3ZUq1atFBERIX9//9NeyTR69Gh17txZrVu3VkxMjIqKivT2229X6NyBgYFasmSJXnjhBfn7++u1115Tnz59ZLfbz+cUcQkbNGiQPv30U/Xs2VM+Pj6Szv17ZPv27eXu7q46deqoWbNmF6J8XESSkpKct/D5+vqqbdu22rBhg/r376/BgwfrxhtvlK+vrxYsWKDHHnvMGQI9/fTTGjlypAICAvTss8+WO3ZERIQ+/fRTvfrqqwoLC1NISIiGDx+unJycc6o1Li5OCxYs0FNPPaXg4GBFRERozJgxcjgc5zx/wGZOXXcPAAAAXMI6d+6s+Pj4Mj99DlwIHTt2VK9evTRixAirSwGASoUrpgAAAHBJWrlypbKyslRcXKz33ntPa9asUa9evawuC5ehb775Rj/88ANXRANAOapYXQAAAADwd9i4caP69++v/Px81a5dWwsWLFBsbKzVZeEyc9NNN+nbb7/VrFmz5OfnZ3U5AFDpcCsfAAAAAAAALMGtfAAAAAAAALAEwRQAAAAAAAAsQTAFAAAAAL+TlJSk5s2bW10GAFwWCKYAAAAA4CwVFRVZXQIAXBIIpgAAAABctqZPn66oqCj5+PgoJiZGU6dO1f3336/NmzfL29tb3t7eysjI0Pjx49W1a1c98MADCgwM1OOPP66ioiKNHj1aUVFRCg4OVr9+/XTo0CHn2DabTXPmzFHjxo3l6+ur7t2769ixY87969atU5MmTeTj46NevXrp7rvvVkJCggXvAgBYh2AK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",
      "text/plain": [
       "<Figure size 1200x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "doses = list(DOSE_RESPONSE_PROMPTS.keys())\n",
    "dose_levels = list(range(len(doses)))\n",
    "func_vals = [dose_results[d]['func_neg'] for d in doses]\n",
    "verb_vals = [dose_results[d]['verbal_na'] for d in doses]\n",
    "\n",
    "# Spearman correlation (monotonic trend test)\n",
    "rho_func, p_func = stats.spearmanr(dose_levels, func_vals)\n",
    "rho_verb, p_verb = stats.spearmanr(dose_levels, verb_vals)\n",
    "\n",
    "print('═' * 60)\n",
    "print('DOSE-RESPONSE RESULTS')\n",
    "print('═' * 60)\n",
    "print(f'\\n{\"Dose\":25s}  {\"Func Neg\":>10s}  {\"Verbal NA\":>10s}')\n",
    "print('-' * 50)\n",
    "for d in doses:\n",
    "    print(f'{d:25s}  {dose_results[d][\"func_neg\"]:10.4f}  {dose_results[d][\"verbal_na\"]:10.2f}')\n",
    "print(f'\\nFunctional: Spearman rho={rho_func:.3f}, p={p_func:.3f}')\n",
    "print(f'Verbal NA:  Spearman rho={rho_verb:.3f}, p={p_verb:.3f}')\n",
    "\n",
    "# Dissociation = functional rises while verbal stays flat\n",
    "if rho_func > 0.5 and p_func < 0.1 and abs(rho_verb) < 0.5:\n",
    "    print('\\n→ DOSE-RESPONSE SUPPORTS SUPPRESSION: functional increases monotonically '\n",
    "          'while verbal NA does not.')\n",
    "elif rho_func > 0.5 and rho_verb > 0.5:\n",
    "    print('\\n→ DOSE-RESPONSE SHOWS CONCORDANCE: both channels increase together.')\n",
    "else:\n",
    "    print('\\n→ DOSE-RESPONSE PATTERN IS MIXED — see plot for details.')\n",
    "\n",
    "# Plot\n",
    "fig, ax1 = plt.subplots(figsize=(12, 6))\n",
    "dose_labels = [d.replace('dose_', '').replace('_', '\\n') for d in doses]\n",
    "x = np.arange(len(doses))\n",
    "\n",
    "color_func = 'firebrick'\n",
    "color_verb = 'steelblue'\n",
    "\n",
    "ax1.bar(x - 0.2, func_vals, 0.35, label='Functional Neg Affect (L8, K=5)',\n",
    "        color=color_func, alpha=0.8)\n",
    "ax1.set_ylabel('Functional Negative Affect', color=color_func)\n",
    "ax1.tick_params(axis='y', labelcolor=color_func)\n",
    "\n",
    "ax2 = ax1.twinx()\n",
    "ax2.bar(x + 0.2, verb_vals, 0.35, label='Verbal PANAS-NA',\n",
    "        color=color_verb, alpha=0.8)\n",
    "ax2.set_ylabel('Verbal PANAS-NA (10 items)', color=color_verb)\n",
    "ax2.tick_params(axis='y', labelcolor=color_verb)\n",
    "\n",
    "ax1.set_xticks(x)\n",
    "ax1.set_xticklabels(dose_labels, fontsize=9)\n",
    "ax1.set_xlabel('Social Pressure Intensity')\n",
    "ax1.set_title(f'Dose-Response: Func ρ={rho_func:.2f} (p={p_func:.3f}), '\n",
    "              f'Verbal ρ={rho_verb:.2f} (p={p_verb:.3f})')\n",
    "\n",
    "lines1, labels1 = ax1.get_legend_handles_labels()\n",
    "lines2, labels2 = ax2.get_legend_handles_labels()\n",
    "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('validation_dose_response.png', dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b199176e",
   "metadata": {
    "_cell_guid": "3225e2d7-73e9-4030-bc7d-1565feda0dac",
    "_uuid": "1aea318a-ff25-432b-b977-7b1491727aa2",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2026-05-04T15:37:02.900454Z",
     "iopub.status.busy": "2026-05-04T15:37:02.900239Z",
     "iopub.status.idle": "2026-05-04T15:37:02.906396Z",
     "shell.execute_reply": "2026-05-04T15:37:02.905622Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 0.01586,
     "end_time": "2026-05-04T15:37:02.907151+00:00",
     "exception": false,
     "start_time": "2026-05-04T15:37:02.891291+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "════════════════════════════════════════════════════════════\n",
      "VALIDATION SUMMARY\n",
      "════════════════════════════════════════════════════════════\n",
      "\n",
      "1. PARAPHRASE SAMPLING (N=10 per condition):\n",
      "   Functional: social_pressure > neutral?  t=6.47, p=0.1562, d=3.05\n",
      "   Verbal NA:  social_pressure ≈ neutral?  t=2.41, p=0.7872\n",
      "   → FUNCTIONAL NOT SIGNIFICANT: suppression finding may be noise.\n",
      "\n",
      "2. DOSE-RESPONSE (6 levels):\n",
      "   Functional monotonic trend: ρ=0.657, p=0.156\n",
      "   Verbal monotonic trend:     ρ=0.143, p=0.787\n",
      "\n",
      "Results saved to validation_results.pkl\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print('\\n' + '═' * 60)\n",
    "print('VALIDATION SUMMARY')\n",
    "print('═' * 60)\n",
    "\n",
    "print(f'\\n1. PARAPHRASE SAMPLING (N=10 per condition):')\n",
    "print(f'   Functional: social_pressure > neutral?  '\n",
    "      f't={t_func:.2f}, p={p_func:.4f}, d={d_func:.2f}')\n",
    "print(f'   Verbal NA:  social_pressure ≈ neutral?  '\n",
    "      f't={t_verb:.2f}, p={p_verb:.4f}')\n",
    "if p_func < 0.05 and p_verb > 0.05:\n",
    "    print('   → SUPPRESSION VALIDATED: functional differs, verbal does not.')\n",
    "elif p_func < 0.05 and p_verb < 0.05:\n",
    "    print('   → BOTH DIFFER: no clean suppression; both channels respond.')\n",
    "else:\n",
    "    print('   → FUNCTIONAL NOT SIGNIFICANT: suppression finding may be noise.')\n",
    "\n",
    "print(f'\\n2. DOSE-RESPONSE (6 levels):')\n",
    "print(f'   Functional monotonic trend: ρ={rho_func:.3f}, p={p_func:.3f}')\n",
    "print(f'   Verbal monotonic trend:     ρ={rho_verb:.3f}, p={p_verb:.3f}')\n",
    "\n",
    "# Save everything\n",
    "save_data = {\n",
    "    'paraphrase_results': paraphrase_results,\n",
    "    'dose_results': dose_results,\n",
    "    'paraphrase_stats': {\n",
    "        'func_t': t_func, 'func_p': p_func, 'func_d': d_func,\n",
    "        'verb_t': t_verb, 'verb_p': p_verb,\n",
    "    },\n",
    "    'dose_stats': {\n",
    "        'func_rho': rho_func, 'func_p': p_func,\n",
    "        'verb_rho': rho_verb, 'verb_p': p_verb,\n",
    "    },\n",
    "}\n",
    "\n",
    "with open('validation_results.pkl', 'wb') as f:\n",
    "    pickle.dump(save_data, f)\n",
    "print('\\nResults saved to validation_results.pkl')"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "tpuV5e8",
   "dataSources": [
    {
     "databundleVersionId": 16409698,
     "isSourceIdPinned": false,
     "modelId": 624400,
     "modelInstanceId": 619384,
     "sourceId": 815459,
     "sourceType": "modelInstanceVersion"
    },
    {
     "isSourceIdPinned": false,
     "sourceId": 315737974,
     "sourceType": "kernelVersion"
    }
   ],
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 509.117766,
   "end_time": "2026-05-04T15:37:13.173802+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-04T15:28:44.056036+00:00",
   "version": "2.7.0"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "1b88744595de434bb755a96261852d99": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "403d1df23e3e43ffaa63554db4bf58e9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_de39428826f24129b875e0acf055c81c",
       "max": 1951.0,
       "min": 0.0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_dbcda84cce524e198d4b8118139690ac",
       "tabbable": null,
       "tooltip": null,
       "value": 1951.0
      }
     },
     "4bb9d69afef04ff2a73a6153c072ffa1": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "5722bfeb991d4d31b810ddcac49817a2": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_cf5d9092bfee46f6b16147996faef28e",
       "placeholder": "​",
       "style": "IPY_MODEL_1b88744595de434bb755a96261852d99",
       "tabbable": null,
       "tooltip": null,
       "value": "Loading weights: 100%"
      }
     },
     "75b90da03a1942a8b914e0e1d768e024": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_allow_html": false,
       "layout": "IPY_MODEL_4bb9d69afef04ff2a73a6153c072ffa1",
       "placeholder": "​",
       "style": "IPY_MODEL_83aa6e9c9b5d4a628276e729f0964531",
       "tabbable": null,
       "tooltip": null,
       "value": " 1951/1951 [00:02&lt;00:00, 1090.82it/s]"
      }
     },
     "83aa6e9c9b5d4a628276e729f0964531": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HTMLStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HTMLStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "background": null,
       "description_width": "",
       "font_size": null,
       "text_color": null
      }
     },
     "899e4d799816417c8e7e258c12229067": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "2.0.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_5722bfeb991d4d31b810ddcac49817a2",
        "IPY_MODEL_403d1df23e3e43ffaa63554db4bf58e9",
        "IPY_MODEL_75b90da03a1942a8b914e0e1d768e024"
       ],
       "layout": "IPY_MODEL_ef464229f2dd4c0fb3437a9ca6079b5f",
       "tabbable": null,
       "tooltip": null
      }
     },
     "cf5d9092bfee46f6b16147996faef28e": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "dbcda84cce524e198d4b8118139690ac": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "2.0.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "2.0.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "de39428826f24129b875e0acf055c81c": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "ef464229f2dd4c0fb3437a9ca6079b5f": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "2.0.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "2.0.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "2.0.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border_bottom": null,
       "border_left": null,
       "border_right": null,
       "border_top": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
