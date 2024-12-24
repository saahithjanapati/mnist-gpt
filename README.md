# MNIST-GPT

### Autocomplete for MNIST images

This repository demonstrates an autoregressive GPT model trained on the MNIST dataset to “autocomplete” pixel values in a 28×28 image. Instead of text, the model completes a sequence of 784 discrete pixel values (one token per pixel).

---

## Samples

<div style="display: flex; gap: 10px;">
    <img src="./generations/imgs/4.png" alt="Samples 0-9" style="width: 30%;">
    <img src="./generations/imgs/2.png" alt="Samples 0-9" style="width: 30%;">
</div>




And here are GIFs of all the digits being generated:

<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
    <img src="./generations/gifs/0.gif" alt="GIF 0" style="width: 170px;">
    <img src="./generations/gifs/1.gif" alt="GIF 1" style="width: 170px;">
    <img src="./generations/gifs/2.gif" alt="GIF 2" style="width: 170px;">
    <img src="./generations/gifs/3.gif" alt="GIF 3" style="width: 170px;">
    <img src="./generations/gifs/4.gif" alt="GIF 4" style="width: 170px;">
    <img src="./generations/gifs/5.gif" alt="GIF 5" style="width: 170px;">
    <img src="./generations/gifs/6.gif" alt="GIF 6" style="width: 170px;">
    <img src="./generations/gifs/7.gif" alt="GIF 7" style="width: 170px;">
    <img src="./generations/gifs/8.gif" alt="GIF 8" style="width: 170px;">
    <img src="./generations/gifs/9.gif" alt="GIF 9" style="width: 170px;">
</div>

---
## Usage

First make sure you have the necessary dependencies (PyTorch, Matplotlib, and PIL) installed. You can run `pip install -r requirements.txt`.


You can generate images from the command line using the `run.py` script. For example:

```bash
python run.py \
    --targets "0,1,2" \
    --temperature 1.2 \
    --top_k 50 \
    --generate_gifs
```

### Available Arguments

- `--targets`: Comma-separated list of digits to generate (e.g. `"0,1,2"`). By default, it generates `[0..9]`.
- `--temperature`: Adjusts the randomness of sampling. Defaults to `1.0`.
- `--top_k`: Use top-k sampling. For example, `--top_k 50` keeps the top 50 tokens. Defaults to `None`.
- `--top_p`: Use nucleus (top-p) sampling. For example, `--top_p 0.9` keeps tokens whose cumulative probability ≤ 0.9. Defaults to `None`.
- `--generate_gifs`: If set, also generate pixel-by-pixel GIFs for each digit. Defaults to off.


## Descriptions of Files

- **`run.py`**  
  Command-line script that loads the trained GPT model (`trained_model_state_dict.pt`), accepts command-line arguments, generates digits, and saves outputs (PNGs and optionally GIFs).

- **`mnist_gpt.py`**  
  This file defines the `GPT` model architecture specialized for MNIST pixel (token) generation.

- **`mnist_gpt_train.ipynb`**  
  Jupyter notebook used to train the GPT model on MNIST data.

- **`trained_model_state_dict.pt`**  
  The PyTorch state dictionary for the pre-trained GPT.
