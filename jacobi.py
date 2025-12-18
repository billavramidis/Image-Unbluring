from PIL import Image
import numpy as np
import glob
import os

image_locations = glob.glob("outputs/noisy/*.png")
a = 0.08
b = 0.23

save_dir = "outputs/clean"

iterations = int(input("Give the number of iterations: "))

def find_neighbours(i, j, channel, width, height):
    center = channel[i, j]

    left = channel[i, j-1] if j > 0 else center
    right = channel[i, j+1] if j < width - 1 else center
    up = channel[i-1, j] if i > 0 else center
    bottom = channel[i+1, j] if i < height - 1 else center

    return left, right, up, bottom

def jacobi_update(original_channel, current_channel, i, j, width, height):
    left, right, up, bottom = find_neighbours(i, j, current_channel, width, height)
    return a * original_channel[i][j] + b * (left + right + up + bottom)

def jacobi_method(original_pixels, current_pixels):
        height, width, _ = current_pixels.shape

        orig_r = original_pixels[:, :, 0]
        orig_g = original_pixels[:, :, 1]
        orig_b = original_pixels[:, :, 2]

        curr_r = current_pixels[:,:,0]
        curr_g = current_pixels[:,:,1]
        curr_b = current_pixels[:,:,2]

        new_r = curr_r.copy()
        new_g = curr_g.copy()
        new_b = curr_b.copy()

        for i in range(0, height):
            for j in range(0, width):
                new_r[i][j] = jacobi_update(orig_r, curr_r, i, j, width, height)
                new_g[i][j] = jacobi_update(orig_g, curr_g, i, j, width, height)
                new_b[i][j] = jacobi_update(orig_b, curr_b, i, j, width, height)

        return np.dstack((new_r, new_g, new_b))
                

for noisy_image in image_locations:
    noisy_image_name = os.path.basename(noisy_image)
    clean_image_name = noisy_image_name.split("_")[0]

    with Image.open(noisy_image).convert("RGB") as img:
        original_pixels = np.array(img).astype(np.float64)
        
        current_pixels = original_pixels.copy()

        for k in range(0, iterations):
            new_pixels = jacobi_method(original_pixels, current_pixels)
            current_pixels = new_pixels.copy()
        
        result = np.clip(current_pixels, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result, "RGB")

        clean_image_path = f"{clean_image_name}_clean_{iterations}.png"
        result_img.save(os.path.join(save_dir, clean_image_path))
