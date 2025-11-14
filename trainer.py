import torch
import os
from tqdm import tqdm
import torchvision.utils as vutils
from utils import loss_function

def save_samples(model, fixed_z, epoch, config):
    """
    Generates and saves a grid of images from the model.
    """
    model.eval()
    with torch.no_grad():
        def gen(attr, name):
            # Create a batch of attribute vectors
            c = torch.tensor([attr] * 64, device=config.DEVICE, dtype=torch.float32)
            gen_imgs = model.decode(fixed_z, c)
            vutils.save_image(
                gen_imgs,
                f"{config.SAMPLE_DIR}/epoch{epoch}_{name}.png",
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )

        # Generate images for different attribute combinations
        gen([0,0,0], "neutral")
        gen([1,0,0], "eyeglasses_only")
        gen([0,1,0], "smiling_only")
        gen([0,0,1], "mustache_only")
        gen([1,1,1], "all_three")
        
    model.train()
    print(f"\nSamples saved to {config.SAMPLE_DIR}/epoch{epoch}_*.png")


def train(model, dataloader, optimizer, scheduler, config):
    """
    Main training loop.
    """
    print("Starting training...")
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    # Fixed noise for consistent sample generation
    fixed_z = torch.randn(64, config.LATENT_DIM, device=config.DEVICE)
    total_steps = len(dataloader) * config.NUM_EPOCHS
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")

        for imgs, attrs in pbar:
            imgs = imgs.to(config.DEVICE)
            attrs = attrs.to(config.DEVICE)

            # Forward pass
            recon, mu, logvar = model(imgs, attrs)
            loss, mse, kld = loss_function(recon, imgs, mu, logvar, config.BETA)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            if config.GRAD_CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            
            optimizer.step()
            scheduler.step() # Step the scheduler at each batch

            total_epoch_loss += loss.item()
            pbar.set_postfix({
                'Loss': f'{loss.item():.1f}',
                'MSE': f'{mse.item():.1f}',
                'KLD': f'{kld.item():.1f}',
                'LR': f'{scheduler.get_last_lr()[0]:.1e}'
            })

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"\n=== Epoch {epoch} | Avg Loss: {avg_loss:.2f} | LR: {scheduler.get_last_lr()[0]:.1e} ===\n")

        # Save samples periodically
        if epoch % 25 == 0 or epoch == 1:
            save_samples(model, fixed_z, epoch, config)

    # Save final model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\nTraining finished! Model saved as '{config.MODEL_SAVE_PATH}'")
    print(f"Check the '{config.SAMPLE_DIR}' folder for generated images.")