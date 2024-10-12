# hack to import from dir
import os, sys

sys.path.insert(0, os.getcwd())

import argparse
from options.train_options import TrainOptions
from models.cycle_gan_model import CycleGANModel
from data.base_dataset import get_transform
from util.util import tensor2im
from PIL import Image
from torch import Tensor

extra_args = sys.argv[3:]
sys.argv[1:] = sys.argv[1:3]

opt_cls = TrainOptions()
opt = opt_cls.parse()

additional_parser = argparse.ArgumentParser(description="Additional options")
mode_group = additional_parser.add_mutually_exclusive_group(required=True)

# Single image mode
mode_group.add_argument(
    "--single_image", action="store_true", help="Use single image mode"
)
additional_parser.add_argument(
    "--image_path", type=str, help="Path to the single image file"
)


# Multiple image mode
mode_group.add_argument(
    "--multiple_images", action="store_true", help="Use multiple image mode"
)
additional_parser.add_argument(
    "--image_dir", type=str, help="Path to the directory containing multiple images"
)

# Rest of Params
additional_parser.add_argument(
    "--AtoB", default=True, type=bool, help="transformation direction"
)
additional_parser.add_argument(
    "--results_path",
    required=True,
    type=str,
    help="path to results of transformed images",
)

additional_opt, _ = additional_parser.parse_known_args(extra_args)


A_TO_B = additional_opt.AtoB

RESULTS_PATH = additional_opt.results_path

img2tensor = get_transform(opt, grayscale=False)


def save_output(tensor: Tensor, filename: str):
    img = tensor2im(tensor)
    img = Image.fromarray(img)
    img.save(f"{RESULTS_PATH}/{filename}")


def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = img2tensor(image)
    img_tensor = img_tensor.unsqueeze(0)
    output = transformer(img_tensor)
    filename = image_path.split("/")[-1]
    save_output(output, filename)


model = CycleGANModel(opt)
model.load_networks("latest")


# statue(A) to human(B)
if A_TO_B:
    transformer = getattr(model, "netG_A")
else:
    transformer = getattr(model, "netG_B")


if additional_opt.single_image:
    transform_image(additional_opt.image_path)
else:
    IMAGES_PATH = additional_opt.image_dir

    for filename in os.listdir(IMAGES_PATH):
        transform_image(f"{IMAGES_PATH}/{filename}")
