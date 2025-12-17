from PIL import Image
import numpy as np
import glob

image_locations = glob.glob("**/*.png")
a = 0.08
b = 0.23

iterations = int(input("Give the number of iterations: "))

def find_neighbours(i, j, channel, width, height):
    center = channel[i, j]

    up = channel[i-1, j] if i > 0 else center
    bottom = channel[i+1, j] if i < height - 1 else center
    left = channel[i, j-1] if j > 0 else center
    right = channel[i, j+1] if j < width - 1 else center

    return left, right, up, bottom

def jacobi_update(original_channel, i, j, width, height):
    left, right, bottom, up = find_neighbours(i, j, original_channel, width, height)
    return a * original_channel[i][j] + b * (left + right + bottom + up)

def jacobi_method(current_pixels):
        height, width, _ = current_pixels.shape

        current_red_channel = current_pixels[:,:,0]
        current_green_channel = current_pixels[:,:,1]
        current_blue_channel = current_pixels[:,:,2]

        new_red_channel = current_red_channel.copy()
        new_green_channel = current_green_channel.copy()
        new_blue_channel = current_blue_channel.copy()

        for i in range(0, height):
            for j in range(0, width):
                new_red_channel[i][j] = jacobi_update(current_red_channel, i, j, width, height)
                new_green_channel[i][j] = jacobi_update(current_green_channel, i, j, width, height)
                new_blue_channel[i][j] = jacobi_update(current_blue_channel, i, j, width, height)

        return np.dstack((new_red_channel, new_green_channel, new_blue_channel))
                

for blurry_image in image_locations:
    with Image.open(blurry_image).convert("RGB") as img:
        original_pixels = np.array(img).astype(np.float64)
        
        current_pixels = original_pixels

        for k in range(0, iterations):
            new_pixels = jacobi_method(current_pixels)
            current_pixels = new_pixels.copy()
        
        result = np.clip(current_pixels, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result, "RGB")
        result_img.save("result.png")
