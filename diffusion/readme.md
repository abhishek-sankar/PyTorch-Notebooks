Overview:

This folder contains an implementation of diffusion models utilizing U-Net architectures enhanced with self-attention mechanisms. The model is designed to generate images through a structured noise scheduling process.

Components:

1. **Custom U-Net with Self-Attention**:

   - The U-Net architecture is augmented with self-attention layers, implemented in the `UNet` class using the `AttentionBlock`. This enhancement allows for better feature extraction and representation.

2. **Time Embeddings**:

   - The `TimeEmbedding` class provides sinusoidal time embeddings, which are crucial for encoding temporal information into the model, improving its ability to handle sequential data.

3. **Noise Scheduling**:
   - The `DiffusionModel` class employs a cosine decay schedule for noise addition, ensuring a smooth and effective diffusion process.

Features:

- **Enhanced Feature Extraction**: The integration of self-attention layers within the U-Net architecture significantly improves the model's ability to capture intricate details in the data.
- **Efficient Noise Scheduling and Sampling**: The use of cosine decay for noise scheduling ensures a robust and efficient sampling process, leading to high-quality image generation.
- **Optimized for 256x256 Image Generation**: The architecture is specifically optimized to handle 256x256 image dimensions, balancing computational efficiency and output quality.
