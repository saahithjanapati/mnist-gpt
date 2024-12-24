#!/usr/bin/env python3

import argparse
import torch
import torch.backends
from mnist_gpt import GPT, GPTConfig
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

PATH_TO_STATE_DICT = "./trained_model_state_dict.pt"
PATH_TO_IMAGE_GENERATIONS = "./generations/imgs/"
PATH_TO_GIF_GENERATIONS = "./generations/gifs/"

def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def create_pixel_gif(full_generation, gif_path, duration_ms=5000):
    """
    Create a GIF showing the pixel-by-pixel generation process.
    full_generation: 1D tensor or numpy array of length 784 (28x28).
    gif_path: where to save the resulting .gif.
    duration_ms: total duration of the GIF in milliseconds.
    """
    frames = []
    # Convert to numpy in case it's a torch tensor
    full_generation = full_generation.cpu().numpy() if hasattr(full_generation, 'cpu') else full_generation

    # We'll build up the image one pixel at a time
    # "Light blue" in RGB is (173, 216, 230)
    height, width = 28, 28
    num_pixels = height * width

    # The total duration is 5 seconds by default, so each frame has:
    frame_duration = max(1, duration_ms / float(num_pixels))  # in ms

    for step in range(1, num_pixels + 1):
        # Create a color image, default is light blue for unknown
        rgb_img = np.full((height, width, 3), [173, 216, 230], dtype=np.uint8)

        # For already generated pixels, fill with grayscale
        # e.g. if the pixel is in [0..255], then fill (val, val, val)
        for px_idx in range(step):
            val = full_generation[px_idx]
            # clamp for safety
            val = max(0, min(255, int(val)))
            row = px_idx // width
            col = px_idx % width
            rgb_img[row, col, :] = [val, val, val]

        frame = Image.fromarray(rgb_img, mode='RGB')
        frames.append(frame)

    # Save frames as an animated GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )

def main():
    parser = argparse.ArgumentParser(description="Generate MNIST images from a GPT model.")
    parser.add_argument("--targets", type=str, default="0-9",
                        help="Comma-separated list of digits to generate, e.g. '0,1,2,3'. Default: '0-9'")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Default: 1.0")
    parser.add_argument("--top_k", type=int, default=None,
                        help="top-k sampling. Default: None (disabled)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="top-p (nucleus) sampling. Default: None (disabled)")
    parser.add_argument("--generate_gifs", action="store_true",
                        help="If set, also generate pixel-by-pixel GIFs. Default: False")

    args = parser.parse_args()

    # Parse the targets
    if args.targets == "0-9":
        targets = list(range(10))
    else:
        # e.g. "0,1,2" -> [0, 1, 2]
        targets = [int(x) for x in args.targets.split(",")]

    # Load the GPT model
    model = GPT(GPTConfig())
    state_dict = torch.load(PATH_TO_STATE_DICT, map_location='cpu')
    model.load_state_dict(state_dict)

    device = get_best_device()
    model.to(device)

    num_parameters = model.get_num_params()
    print(f"Model has {num_parameters} parameters")

    print("Generating...")

    # Generate images in one go if desired, or in a loop; here we do one go:
    # The code in mnist_gpt.py likely supports passing a list of targets to generate a batch
    imgs = model.generate(
        targets, 
        temperature=args.temperature, 
        top_k=args.top_k,
        top_p=args.top_p
    )

    # imgs should now be shape [batch_size, seq_len], each row an image
    # be sure that your generate() function is indeed returning multiple images for multiple targets

    # Save outputs
    img_save_path = Path(PATH_TO_IMAGE_GENERATIONS)
    gif_save_path = Path(PATH_TO_GIF_GENERATIONS)

    # Create directories if needed
    img_save_path.mkdir(parents=True, exist_ok=True)
    if args.generate_gifs:
        gif_save_path.mkdir(parents=True, exist_ok=True)

    # For each target, save the PNG and optionally the GIF
    for i, target in enumerate(targets):
        # Remove the start and end token if your model uses them
        # (depending on how your generate function structures the output)
        generated = imgs[i, 1:-1]

        # Save PNG
        plt.imshow(generated.reshape(28, 28), cmap='gray')
        save_path = img_save_path / f"{target}.png"
        plt.savefig(save_path)
        plt.clf()

        if args.generate_gifs:
            # Save GIF
            gif_path = gif_save_path / f"{target}.gif"
            create_pixel_gif(generated, gif_path, duration_ms=1000)

    print("Done!")

if __name__ == "__main__":
    main()