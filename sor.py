from PIL import Image
import numpy as np
from pathlib import Path

noisy_directory_path = Path("outputs/noisy/")
noisy_image_paths = [
    f for f in noisy_directory_path.iterdir() if f.suffix in [".jpg", ".png", ".webp"]
]

if len(noisy_image_paths) == 1:
    noisy_image = noisy_image_paths[0]
else:
    raise ValueError(f"Expected 1 Image, found {len(noisy_image_paths)}")

a = 0.08
b = 0.23

save_path = Path("outputs/denoised/")

iterations = int(input("Give the number of iterations: "))

while True:
    print("Omega must be between (0,2)")
    omega = float(input("Give omega value: "))

    if omega > 0 and omega < 2:
        break


def sor_method(original_pixels, current_pixels):
    height, width, channels = original_pixels.shape

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channels):
                left = current_pixels[i + 1, j, k]
                right = current_pixels[i + 1, j + 2, k]
                up = current_pixels[i, j + 1, k]
                bottom = current_pixels[i + 2, j + 1, k]

                current_pixels[i + 1, j + 1, k] = (1 - omega) * current_pixels[
                    i + 1, j + 1, k
                ] + omega * (
                    a * original_pixels[i, j, k] + b * (left + right + up + bottom)
                )


noisy_image_name = noisy_image.stem
clean_image_name = noisy_image_name.split("_")[0]

with Image.open(noisy_image).convert("RGB") as img:
    original_pixels = np.array(img).astype(np.float64)

    current_pixels = np.pad(
        original_pixels, pad_width=((1, 1), (1, 1), (0, 0)), mode="edge"
    )

    for _ in range(0, iterations):
        sor_method(original_pixels, current_pixels)

    current_pixels = current_pixels[1:-1, 1:-1, :]

    result = np.clip(current_pixels, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result, "RGB")

    clean_image_path = f"{clean_image_name}_clean_sor_{iterations}.png"
    result_img.save(save_path / clean_image_path)
