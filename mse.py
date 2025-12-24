from PIL import Image
import numpy as np
from pathlib import Path

denoised_directory_path = Path("outputs/denoised/")
denoised_image_paths = [
    file
    for method_directory in denoised_directory_path.iterdir()
    if method_directory.is_dir()
    for file in method_directory.iterdir()
    if file.suffix.lower() in [".jpg", ".png", ".webp"]
]

clean_directory_path = Path("inputs/clean/")
clean_image_paths = [
    f
    for f in clean_directory_path.iterdir()
    if f.suffix.lower() in [".jpg", ".png", ".webp"]
]

if len(clean_image_paths) == 1:
    clean_image = clean_image_paths[0]
else:
    raise ValueError(f"Expected 1 Image, found {len(clean_image_paths)}")


for denoised_image in denoised_image_paths:
    with Image.open(denoised_image) as img1, Image.open(clean_image) as img2:
        if not img1.size == img2.size:
            raise ValueError("Image Dimensions Do Not Match")

        denoised_pixels = np.array(img1.convert("RGB")).astype(np.float64)
        clean_pixels = np.array(img2.convert("RGB")).astype(np.float64)

        mse = np.mean((clean_pixels - denoised_pixels) ** 2)
        print(f"MSE for {denoised_image.stem}: {round(mse,3)}")
