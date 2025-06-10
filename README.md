

---

# A Novel Black-Box Adversarial Attack Framework

This repository presents a novel **Black-Box Adversarial Attack** framework that targets defended models with minimal internal access, using techniques like frequency decomposition and spatial transformation. This attack is designed to be query-efficient and model-agnostic.

## 🗂️ Project Structure

Here is a visual overview of the folder structure used in this project:

```markdown

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

### Prerequisites

- Python 3.8+
- pip or conda
- NVIDIA GPU (for acceleration)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Munia2319/BB-Attack.git
cd BB-Attack

# Create a virtual environment 
conda env create -f environment.yml

````

## Running the Attack

### Option 1: Using the `run_attack.sh` Script (Recommended)

Before launching the attack, you can customize the parameters:

#### 1️⃣ Edit Parameters

Open the file and update the values to your needs:

````markdown
cd stba
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


The script will:

* Run the attack using your config and parameters
* Automatically evaluate the results (SSIM, LPIPS, fooling rate)
* Save everything to `outputs/adversarial_images/` and `outputs/eval/`


Outputs:

* **SSIM**
* **LPIPS**
* **Fooling Rate**
  → All saved in `outputs/eval/` directory

---


