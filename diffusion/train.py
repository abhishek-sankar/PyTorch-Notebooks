import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model import UNet, DiffusionModel
from dataset_prep import get_dataloaders


def train_diffusion(
    num_epochs=100, batch_size=16, image_size=256, device="cuda", save_dir="models"
):
    model = UNet().to(device)
    diffusion = DiffusionModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_loader, val_loader = get_dataloaders(
        "data/processed_abstract_art", batch_size=batch_size, image_size=image_size
    )

    # Setup tensorboard
    writer = SummaryWriter("runs/diffusion_training")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
            noisy_images, noise = diffusion.add_noise(images, t)
            predicted_noise = model(noisy_images, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Log to tensorboard
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample(
                    model, n_samples=4, size=(image_size, image_size), device=device
                )
                writer.add_images("Generated", samples, epoch)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_dir / f"checkpoint_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    train_diffusion()
