import sys
import os
import yaml
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stba.attacker import STBAttacker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/mmunia/stba-black-box/configs/default.yaml")

    # Optional CLI overrides
    parser.add_argument('--max_images', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lambda_', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--qmax', type=int)
    parser.add_argument('--nsample', type=int)
    parser.add_argument('--adjustnum', type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_type', type=str, default='robust')  # <-- ADD THIS

    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for key in ['max_images', 'lr', 'lambda_', 'sigma', 'qmax', 'nsample', 'adjustnum', 'model_name', 'model_type']:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    if "model_name" in config:
        config["current_model_name"] = config["model_name"]

    # Pass model_info to STBAttacker
    model_info = {
    "name": config.get("model_name", "Wong2020Fast"),
    "type": config.get("model_type", "robust")
    }

    # Always reflect model name in config
    config["current_model_name"] = model_info["name"]

    attacker = STBAttacker(config, model_info)
    attacker.run(max_images=config.get("max_images"))

if __name__ == "__main__":
    main()
