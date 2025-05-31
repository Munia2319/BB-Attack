import os
import torch
import numpy as np
import lpips
import pandas as pd
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from tqdm import tqdm

def normalize_for_lpips(x):
    return (x * 2) - 1

def compute_ssim(img1, img2):
    """
    Compute SSIM for [H, W, C] images using CIFAR-10-safe settings.
    """
    return ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=7)

def evaluate_adversarial_examples(originals, adversarials, queries, labels, preds, save_dir="eval_outputs", config=None):
    model_name = config.get("current_model_name", "unnamed_model") if config else "unnamed_model"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(save_dir, model_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    lpips_model = lpips.LPIPS(net='alex').cuda()

    total = len(originals)
    success_count = 0
    total_query_count = 0
    ssim_scores = []
    lpips_scores = []
    log_data = []

    for i in tqdm(range(total), desc=f"Evaluating {model_name}"):
        orig = originals[i]
        adv = adversarials[i]
        label = labels[i]
        pred = preds[i]
        query = queries[i]

        if label != pred:
            success_count += 1
            total_query_count += query

            ssim_score = compute_ssim(
                np.transpose(orig.numpy(), (1, 2, 0)),
                np.transpose(adv.numpy(), (1, 2, 0))
            )
            ssim_scores.append(ssim_score)

            orig_lp = normalize_for_lpips(orig.unsqueeze(0).cuda())
            adv_lp = normalize_for_lpips(adv.unsqueeze(0).cuda())
            lp_score = lpips_model(orig_lp, adv_lp).item()
            lpips_scores.append(lp_score)

            comparison = torch.stack([orig, adv], dim=0)
            save_image(comparison, os.path.join(output_dir, f"compare_{i}.png"), nrow=2)

        log_data.append({
            "image_id": i,
            "label": int(label),
            "pred": int(pred),
            "queries": query,
            "success": label != pred,
            "ssim": ssim_score if label != pred else None,
            "lpips": lp_score if label != pred else None
        })

    # Summary
    success_rate = 100.0 * success_count / total
    avg_queries = total_query_count / success_count if success_count > 0 else 0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0

    print("\n========== Evaluation Summary ==========")
    print(f"Model: {model_name}")
    print(f"Total images        : {total}")
    print(f"Successful attacks  : {success_count}")
    print(f"Success rate        : {success_rate:.2f}%")
    print(f"Average queries     : {avg_queries:.2f}")
    print(f"Average SSIM        : {avg_ssim:.4f}")
    print(f"Average LPIPS       : {avg_lpips:.4f}")

    # Save CSV
    csv_path = os.path.join(output_dir, "evaluation_log.csv")
    pd.DataFrame(log_data).to_csv(csv_path, index=False)

    # Save TXT summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Successful attacks: {success_count}\n")
        f.write(f"Success rate: {success_rate:.2f}%\n")
        f.write(f"Average queries: {avg_queries:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")

    # Save JSON summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
         "model": model_name,
         "success_rate": float(success_rate),
         "avg_queries": float(avg_queries),
        "avg_ssim": float(avg_ssim),
        "avg_lpips": float(avg_lpips)
            }, f, indent=4)


    print(f"[✓] Logs saved at {output_dir}")
