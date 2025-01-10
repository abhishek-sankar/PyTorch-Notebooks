import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t):

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :].to(t.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self.proj(embeddings)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)  # B, H*W, heads, C//heads

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, H, W, C)

        x = self.proj(x)
        return x.permute(0, 3, 1, 2)  # Back to B, C, H, W


class UNet(nn.Module):
    def __init__(self, in_channels=3, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()

        self.time_mlp = TimeEmbedding(dim)
        self.init_conv = nn.Conv2d(in_channels, dim, 3, padding=1)

        # Downsampling
        dims = [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleList(
                    [
                        AttentionBlock(dim_in),
                        nn.Conv2d(dim_in, dim_out, 3, padding=1),
                        nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1),
                    ]
                )
            )

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = AttentionBlock(mid_dim)
        self.mid_block2 = nn.Conv2d(mid_dim, mid_dim, 3, padding=1)

        # Upsampling
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1),
                        nn.Conv2d(dim_out * 2, dim_out, 3, padding=1),
                        AttentionBlock(dim_out),
                    ]
                )
            )

        # Final conv
        self.final_conv = nn.Conv2d(dim, in_channels, 3, padding=1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.init_conv(x)

        # Downsampling
        h = []
        for attn, conv1, conv2 in self.downs:
            x = attn(x)
            x = conv1(x)
            x = F.silu(x)
            h.append(x)
            x = conv2(x)
            x = F.silu(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = F.silu(x)

        # Upsampling
        for up, conv, attn in self.ups:
            x = up(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = conv(x)
            x = F.silu(x)
            x = attn(x)

        return self.final_conv(x)


class DiffusionModel:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        """Add noise to the input image according to the noise schedule"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[
            :, None, None, None
        ]
        ε = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * ε, ε

    @torch.no_grad()
    def sample(self, model, n_samples, size, device):
        """Generate samples using the trained model"""
        model.eval()
        x = torch.randn(n_samples, 3, *size).to(device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise)
                + torch.sqrt(beta) * noise
            )

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


def train_step(model, diffusion, optimizer, images, device):
    """Single training step"""
    batch_size = images.shape[0]
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

    noisy_images, noise = diffusion.add_noise(images, t)
    predicted_noise = model(noisy_images, t)
    loss = F.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
