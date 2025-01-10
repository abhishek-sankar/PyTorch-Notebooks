import torch
from model import UNet, DiffusionModel
from torchvision.utils import save_image


def generate_images(checkpoint_path, num_images=4, image_size=256, device="cuda"):
    model = UNet().to(device)
    diffusion = DiffusionModel()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        samples = diffusion.sample(
            model, n_samples=num_images, size=(image_size, image_size), device=device
        )

    for i, sample in enumerate(samples):
        save_image(sample, f"generated_image_{i}.png")


if __name__ == "__main__":
    generate_images("models/checkpoint_epoch_90.pt")
