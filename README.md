

---

```markdown
# 🧪 BB-Attack: A Novel Black-Box Adversarial Attack Framework

This repository presents a novel **Black-Box Adversarial Attack** framework that targets defended models with minimal internal access, using techniques like frequency decomposition and spatial transformation. This attack is designed to be query-efficient and model-agnostic.

## 🗂️ Project Structure

Here is a visual overview of the folder structure used in this project:


.
├── configs
│   └── default.yaml
├── data
├── models
│   ├── load_models.py
│   ├── pretrained
│   ├── __pycache__
│   └── wrapper.py
├── outputs
│   ├── adversarial_images
│   ├── eval
│   └── logs
├── README.md
├── requirements.txt
└── stba
    ├── attacker.py
    ├── evaluation.py
    ├── flow_transform.py
    ├── image_decomposition.py
    ├── loss.py
    ├── main.py
    └── __pycache__

````

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.8+
- pip or conda
- NVIDIA GPU (for acceleration)

### 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Munia2319/BB-Attack.git
cd BB-Attack

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
````

Got it! Here's the revised **README section** in Markdown where `evaluation.py` runs **automatically** after the attack (as in your pipeline). This assumes you've already integrated that step in `main.py` or `attacker.run()`.

---

````markdown
## 🚀 Running the Attack

### 🧪 Option 1: Using the `run_attack.sh` Script (Recommended)

Before launching the attack, you can customize the parameters:

#### 1️⃣ Edit Parameters

Open the file and update the values to your needs:

```bash
nano run_attack.sh
````

Example snippet:

```bash
export CUDA_VISIBLE_DEVICES=2
python stba/main.py \
  --config configs/default.yaml \
  --model_name Zhang2019Theoretically \
  --model_type robust \
  --max_images 10 \
  --lr 0.01 \
  --lambda_ 5 \
  --sigma 0.05 \
  --qmax 10000 \
  --nsample 20 \
  --adjustnum 3
```

#### 2️⃣ Run the Attack

Make the script executable (only once):

```bash
chmod +x run_attack.sh
```

Then execute it:

```bash
./run_attack.sh
```

💡 Or simply:

```bash
bash run_attack.sh
```

The script will:

* Run the attack using your config and parameters
* Automatically evaluate the results (SSIM, LPIPS, fooling rate)
* Save everything to `outputs/adversarial_images/` and `outputs/eval/`

---

### Option 2: Run Manually (Single Command)

You can also run it directly from the terminal:

```bash
CUDA_VISIBLE_DEVICES=2 python stba/main.py \
  --config configs/default.yaml \
  --model_name Zhang2019Theoretically \
  --model_type robust \
  --max_images 10 \
  --lr 0.01 \
  --lambda_ 5 \
  --sigma 0.05 \
  --qmax 10000 \
  --nsample 20 \
  --adjustnum 3
```

Evaluation will still be triggered automatically at the end.

---



This will:

* Load and override the YAML config
* Initialize the model
* Run the STBA black-box attack
* Save adversarial examples to `outputs/adversarial_images/`

Outputs:

* **SSIM**
* **LPIPS**
* **Fooling Rate**
  → All saved in `outputs/eval/` directory

---


