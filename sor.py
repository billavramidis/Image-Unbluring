from PIL import Image
import numpy as np
from pathlib import Path

image_directory_path = Path("outputs/noisy/")
image_paths = [
    f for f in image_directory_path.iterdir() if f.suffix in [".jpg", ".png", ".webp"]
]

a = 0.08
b = 0.23

save_path = Path("outputs/clean/")

iterations = int(input("Give the number of iterations: "))

while True:
    print("Omega must be between (0,2)")
    omega = float(input("Give omega value: "))

    if omega > 0 and omega < 2:
        break


def sor_method(original_pixels, current_pixels):
    height, width, channels = current_pixels.shape

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channels):
                center = current_pixels[i, j, k]

                left = current_pixels[i, j - 1, k] if j > 0 else center
                right = current_pixels[i, j + 1, k] if j < width - 1 else center
                up = current_pixels[i - 1, j, k] if i > 0 else center
                bottom = current_pixels[i + 1, j, k] if i < height - 1 else center

                current_pixels[i, j, k] = (1 - omega) * current_pixels[
                    i, j, k
                ] + omega * (
                    a * original_pixels[i, j, k] + b * (left + right + up + bottom)
                )

    return current_pixels


for noisy_image in image_paths:
    noisy_image_name = noisy_image.stem
    clean_image_name = noisy_image_name.split("_")[0]

    with Image.open(noisy_image).convert("RGB") as img:
        original_pixels = np.array(img).astype(np.float64)

        current_pixels = original_pixels.copy()

        for _ in range(0, iterations):
            current_pixels = sor_method(original_pixels, current_pixels)

        result = np.clip(current_pixels, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result, "RGB")

        clean_image_path = f"{clean_image_name}_clean_sor_{iterations}.png"
        result_img.save(save_path / clean_image_path)
