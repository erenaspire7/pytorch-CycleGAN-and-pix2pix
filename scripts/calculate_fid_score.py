import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from scipy import linalg
import argparse


parser = argparse.ArgumentParser(description="Additional options")
parser.add_argument(
    "--real_images_path", required=True, type=str, help="Path to the real images folder"
)
parser.add_argument(
    "--fake_images_path", required=True, type=str, help="Path to the real images folder"
)

opt = parser.parse_args()

REAL_IMAGES = opt.real_images_path
FAKE_IMAGES = opt.fake_images_path


class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates"
        # warnings.warn(msg)
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real_images, generated_images, device="cpu"):
    inception = InceptionV3().to(device)
    inception.eval()

    def get_features(images):
        with torch.no_grad():
            features = inception(images.to(device))
        return features.cpu().numpy()

    real_features = get_features(real_images)
    generated_features = get_features(generated_images)

    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_gen, sigma_gen = np.mean(generated_features, axis=0), np.cov(
        generated_features, rowvar=False
    )

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid_score


def load_images_from_directory(directory):
    # Define a transform to convert PIL image to tensor
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    image_tensors = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            # Construct full file path
            file_path = os.path.join(directory, filename)

            # Open the image
            image = Image.open(file_path)

            # Convert image to tensor
            image_tensor = transform(image)

            # Append to our lists
            image_tensors.append(image_tensor)

    # Stack all the tensors into a single tensor
    image_tensors = torch.stack(image_tensors)

    return image_tensors


real_images = load_images_from_directory(REAL_IMAGES)
generated_images = load_images_from_directory(FAKE_IMAGES)

# Usage example
# real_images and generated_images should be PyTorch tensors of shape (N, C, H, W)
fid_score = calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score}")
