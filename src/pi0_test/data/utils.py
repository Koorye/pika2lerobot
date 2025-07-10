import imageio.v3 as imageio
import os


def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def load_image(image_path):
    return imageio.imread(image_path)
