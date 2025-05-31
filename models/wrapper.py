import torch
import os
from models.load_models import load_victim_model

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_info, config):
        super().__init__()

        # ✅ Set custom RobustBench cache path BEFORE loading the model
        

        if "robustbench_cache" in config:
            cache_path = config["robustbench_cache"]
            os.environ["ROBUSTBENCH_CACHE"] = cache_path
            print(f"[DEBUG] ROBUSTBENCH_CACHE set to: {cache_path}")

        self.model = load_victim_model(model_info, config)  # ✅ use config-based loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.query_count = 0
        self.name = model_info["name"]       # ✅ store metadata if needed
        self.model_type = model_info["type"]

    def query(self, x):
        self.query_count += x.shape[0]
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
        return logits

    def reset(self):
        self.query_count = 0
