import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import torchvision.utils as vutils
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.wrapper import ModelWrapper
from stba.image_decomposition import decompose_image
from stba.flow_transform import apply_flow_to_high_freq
from stba.loss import compute_total_loss

class STBAttacker:
    def __init__(self, config, model_info):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelWrapper(model_info, config).to(self.device)
        self.query_limit = config["qmax"]
        self.lr = config["lr"]
        self.lambda_ = config["lambda_"]
        self.sigma = config["sigma"]
        self.nsample = config["nsample"]
        self.adjustnum = config["adjustnum"]

    # Step 1: Initialize the flow budget
        self.flow_budget = config["flow_budget"]
        self.success = False

    

        # Load CIFAR-10 image
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root=config["dataset_path"], train=False, download=True, transform=transform)
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False)

        ''''self.x_orig, self.label = next(iter(self.loader))
        self.x_orig = self.x_orig.to(self.device)
        self.label = self.label.to(self.device)'''



    def run(self, max_images=None):

        save_dir = self.config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        eval_dir = self.config["eval_dir"]
        os.makedirs(eval_dir, exist_ok=True)
        max_images = max_images or self.config.get("max_images", 10000)

        success_count = 0
        total_queries = []

         # Logs for evaluation
        original_images = []
        adversarial_images = []
        true_labels = []
        predicted_labels = []
        query_counts = []

        for idx, (x_orig, label) in tqdm(enumerate(self.loader), total=max_images):
            if idx >= max_images:
                break

            print(f"\n[Image {idx}] Original label: {label.item()}")
            x_orig = x_orig.to(self.device)
            label = label.to(self.device)
            
            #mu = torch.zeros((B, 2, H, W), device=self.device)
             # Step 4: Decompose the image into low- and high-frequency components
            x_low, x_high = decompose_image(x_orig)
            B, C, H, W = x_high.shape

            # Reset model query count per image
            self.model.reset()

            # Step 2: Initialize mu with small random noise
            mu = torch.randn((B, 2, H, W), device=self.device) * 0.01
            flow_budget = self.flow_budget
            success = False
            
           
            #step 5: Starting of the iterative optimization
            max_steps = self.query_limit // self.nsample
            xi_0 = 0.1
            xi_max = 0.2
            alpha = (xi_max - xi_0) / max_steps  # Linear step increment
            for step in range(max_steps):
                # Step 6.1: Generate gaussian random noise for sampling
                epsilons = torch.randn((self.nsample, 2, H, W), device=self.device)
                grads = []
                loss_list = []
                for i in range(self.nsample):
                    #Step 6.2 and 7: Drawing samples of flow function from the Gaussian distribution and clip it within the range of flow budget
                    f_sample = mu + self.sigma * epsilons[i:i+1]
                    #f_sample = torch.clamp(f_sample, -flow_budget, flow_budget)
                   
                    #x_high_warped = apply_flow_to_high_freq(x_high, f_sample)


                     # Step 8: Create adversarial image by applying flow to high-frequency component
                    x_high_warped = apply_flow_to_high_freq(
                    x_high,
                    f_sample,
                    mode=self.config.get("interp_mode", "bilinear"),
                    padding_mode=self.config.get("padding_mode", "border"),
                    align_corners=self.config.get("align_corners", True))


                    x_adv = torch.clamp(x_low + x_high_warped, 0, 1)
                    # Step 9: Query the model with the adversarial image and compute loss
                    logits = self.model.query(x_adv)
                    #print(f"[DEBUG] lambda_={self.lambda_} used in loss")

                    loss, _, _ = compute_total_loss(logits, label, f_sample, self.lambda_)
                    loss_list.append(loss.item())
                    #grads.append(epsilons[i] * loss.item())

                    # ✅ Early stopping
                    if torch.argmax(logits, dim=1).item() != label.item():
                        print(f"[✓] Early success (step {step}, sample {i}, queries: {self.model.query_count})")
                        success = True
                        total_queries.append(self.model.query_count)
                        success_count += 1
                        final_adv = x_adv.squeeze().detach().cpu()

                        out_path = os.path.join(save_dir, f"adv_{idx}.png")
                        vutils.save_image(x_adv, out_path)
                        print(f"Saved to {out_path}")
                        break

                if success:
                    break

                 # === Step 10: Normalize losses
                loss_tensor = torch.tensor(loss_list, device=self.device)
                loss_norm = (loss_tensor - loss_tensor.mean()) / (loss_tensor.std() + 1e-8)

                # === Step 11: Estimate gradient using normalized losses
                grads = [epsilons[i] * loss_norm[i] for i in range(self.nsample)]
                grad_mu = torch.stack(grads).mean(dim=0)

                # === Step 12: Update mu
                mu = mu - self.lr * grad_mu

                # === Step 13+14+15: Update flow budget
                if (step + 1) % self.adjustnum == 0:
                    flow_budget = xi_0 + step * alpha

                # === Step 16: Compute the adversarial image
            f_final = torch.clamp(mu, -flow_budget, flow_budget)


            x_high_final = apply_flow_to_high_freq(x_high,f_final, mode=self.config.get("interp_mode", "bilinear"),padding_mode=self.config.get("padding_mode", "border"),align_corners=self.config.get("align_corners", True))
            x_adv_final = torch.clamp(x_low + x_high_final, 0, 1)

            # === Step 17–18: Check if final x_adv fools the model
            with torch.no_grad():
                final_logits = self.model.model(x_adv_final.to(self.device))  # Bypass query() to avoid adding +1


                    
            if final_pred != label:
                    print(f"[✓] Success (final attempt, queries: {self.model.query_count})")
                    success = True
                    total_queries.append(self.model.query_count)
                    success_count += 1

                    # Save adversarial image
                    out_path = os.path.join(save_dir, f"adv_{idx}.png")
                    vutils.save_image(x_adv_final, out_path)
                    print(f"Saved to {out_path}")
                    final_adv = x_adv_final.squeeze().detach().cpu()
            else:
                    print("[-] Failed on final clipped image.")
                    total_queries.append(self.query_limit)
                    final_adv = x_orig.squeeze().detach().cpu()

                




        


        
        
            # ✅ [NEW] Automatically call evaluation after attack

            # Append to logs
            original_images.append(x_orig.squeeze().detach().cpu())
            adversarial_images.append(final_adv)
            true_labels.append(label.item())
            predicted_labels.append(final_pred)
            query_counts.append(self.model.query_count)

            
        from evaluation import evaluate_adversarial_examples

        # Stack tensors if not saved/loaded from .pt files
        evaluate_adversarial_examples(
            originals=torch.stack(original_images),
            adversarials=torch.stack(adversarial_images),
            queries=query_counts,
            labels=true_labels,
            preds=predicted_labels,
            save_dir=self.config.get("eval_dir", "./eval_outputs"),
            config={"current_model_name": self.config.get("model_name", "unnamed_model")}
        )


