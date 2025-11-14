import torch
import torch.nn.functional as F
import math
import os
import zipfile

def loss_function(recon, x, mu, logvar, beta):
    """
    VAE Loss (ELBO): Reconstruction Loss + KLD
    """
    # Reconstruction Loss (Mean Squared Error)
    MSE = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    
    # KLD (Kullback-Leibler Divergence)
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return MSE + beta * KLD, MSE, KLD


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom LR Scheduler: Linear warmup followed by cosine annealing.
    (Based on your 'SafeWarmupCosine' class)
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5, max_lr=6e-4, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [self.max_lr * lr_scale for _ in self.base_lrs]
        
        # Cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (self.max_lr - self.min_lr) * cos_val for _ in self.base_lrs]


def zip_images(src_dir, zip_name="images.zip"):
    """
    Zips all images in a directory.
    """
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, src_dir)
                    zipf.write(file_path, arcname)
    print(f"All images from '{src_dir}' zipped into '{zip_name}'!")