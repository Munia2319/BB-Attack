Thanks! Here's the complete `README.md` written in Markdown format for your GitHub repository: [https://github.com/Munia2319/BB-Attack](https://github.com/Munia2319/BB-Attack)

---

```markdown
# 🧪 BB-Attack: A Novel Black-Box Adversarial Attack Framework

This repository presents a novel **Black-Box Adversarial Attack** framework that targets defended models with minimal internal access, using techniques like frequency decomposition and spatial transformation. This attack is designed to be query-efficient and model-agnostic.

## 📁 Project Structure

```

BB-Attack/
├── configs/           # YAML config files for various experiments
├── models/            # Wrappers and model loading utilities
├── stba/              # STBA attack core implementation
│   ├── frequency.py   # Image decomposition (high/low frequency)
│   ├── flow\_transform.py # Spatial transformations
│   ├── loss.py        # Custom loss functions
│   └── attacker.py    # STBA attack main logic
├── ev.py              # Evaluation script (SSIM, LPIPS, etc.)
├── run\_attack.sh      # Example shell script to run the attack
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation

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

##  How to Run the Attack

### 📄 Step 1: Modify the config

Open or create a `.yaml` config file in the `configs/` directory. A typical config includes:

```yaml
model_name: 'Wong2020Fast'
qmax: 10000
lr: 0.01
lambda_: 0.1
sigma: 0.05
nsample: 20
adjustnum: 3
output_dir: 'output/Wong2020Fast/'
```

###  Step 2: Run the attack

```bash
python stba/main.py --config configs/your_config.yaml
```

You can also use the bash script for reproducibility:

```bash
bash run_attack.sh Wong2020Fast configs/wong.yaml
```

### Step 3: Run Evaluation

```bash
python ev.py --result_dir output/Wong2020Fast/
```

This will save SSIM, LPIPS, and fooling rate statistics into CSV, JSON, and image files.

## Metrics Reported

* **SSIM** – Structural similarity index
* **LPIPS** – Learned perceptual image patch similarity
* **Fooling Rate** – Attack success rate

Example output:

```
Average SSIM  : 0.9940
Average LPIPS : 0.0027
Fooling Rate  : 89.4%
```



```


