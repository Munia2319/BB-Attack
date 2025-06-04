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
    return ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=7)

def evaluate_adversarial_examples(originals, adversarials, queries, labels, preds, save_dir="eval_outputs", config=None):
    model_name = config.get("current_model_name", "unnamed_model") if config else "unnamed_model"
    timestamp = config.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")

    root = "/home/mmunia/extra/BB-Attack/outputs"
    adv_img_dir = os.path.join(root, "adversarial_images", model_name, timestamp)
    eval_dir = os.path.join(root, "eval", model_name, timestamp)
    log_dir  = os.path.join(root, "logs", model_name, timestamp)

    for path in [adv_img_dir, eval_dir, log_dir]:
        os.makedirs(path, exist_ok=True)

    lpips_model = lpips.LPIPS(net='alex').cuda()

    total = len(originals)
    success_count = 0
    total_query_count = 0
    ssim_scores = []
    lpips_scores = []
    l2_norms = []
    linf_norms = []
    log_data = []

    for i in tqdm(range(total), desc=f"Evaluating {model_name}"):
        orig = originals[i]
        adv = adversarials[i]
        label = labels[i]
        pred = preds[i]
        query = queries[i]

        # Compute L∞ and L2 norms
        delta = adv - orig
        linf = torch.max(torch.abs(delta)).item()
        l2 = torch.norm(delta).item()

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

            # Save visualizations
            comparison = torch.stack([orig, adv], dim=0)
            save_image(comparison, os.path.join(eval_dir, f"compare_{i}.png"), nrow=2)
            save_image(adv, os.path.join(adv_img_dir, f"adv_{i}.png"))

        else:
            ssim_score = None
            lp_score = None

        # Save norms regardless of success
        l2_norms.append(l2)
        linf_norms.append(linf)

        log_data.append({
            "image_id": i,
            "label": int(label),
            "pred": int(pred),
            "queries": query,
            "success": label != pred,
            "ssim": ssim_score,
            "lpips": lp_score,
            "L2_norm": l2,
            "Linf_norm": linf
        })

    success_rate = 100.0 * success_count / total
    avg_queries = total_query_count / success_count if success_count > 0 else 0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0
    avg_l2 = np.mean(l2_norms)
    avg_linf = np.mean(linf_norms)

    print("\n========== Evaluation Summary ==========")
    print(f"Model: {model_name}")
    print(f"Total images        : {total}")
    print(f"Successful attacks  : {success_count}")
    print(f"Success rate        : {success_rate:.2f}%")
    print(f"Average queries     : {avg_queries:.2f}")
    print(f"Average SSIM        : {avg_ssim:.4f}")
    print(f"Average LPIPS       : {avg_lpips:.4f}")
    print(f"Average L2 norm     : {avg_l2:.4f}")
    print(f"Average L∞ norm     : {avg_linf:.4f}")

    # Save CSV log
    pd.DataFrame(log_data).to_csv(os.path.join(eval_dir, "evaluation_log.csv"), index=False)

    # Save TXT summary
    with open(os.path.join(eval_dir, "summary.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Successful attacks: {success_count}\n")
        f.write(f"Success rate: {success_rate:.2f}%\n")
        f.write(f"Average queries: {avg_queries:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write(f"Average L2 norm: {avg_l2:.4f}\n")
        f.write(f"Average Linf norm: {avg_linf:.4f}\n")

    # Save JSON summary
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump({
            "model": model_name,
            "success_rate": float(success_rate),
            "avg_queries": float(avg_queries),
            "avg_ssim": float(avg_ssim),
            "avg_lpips": float(avg_lpips),
            "avg_l2": float(avg_l2),
            "avg_linf": float(avg_linf)
        }, f, indent=4)

    # Save config
    if config:
        with open(os.path.join(log_dir, "config_used.json"), "w") as f:
            json.dump(config, f, indent=4)

    print(f"[✓] Adversarial images saved at: {adv_img_dir}")
    print(f"[✓] Evaluation metrics saved at: {eval_dir}")
    print(f"[✓] Logs saved at: {log_dir}")
