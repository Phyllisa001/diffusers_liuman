from PIL import Image
import concurrent.futures
import os
from tqdm import tqdm
import random

def center_crop_resize(image_path, output_dir):
    # Load the image
    image = Image.open(image_path)

    # Get the dimensions of the image
    width, height = image.size

    # Calculate the dimensions of the center square
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2

    # Crop the image to the center square
    image = image.crop((left, top, right, bottom))

    # Resize the image to 512x512 pixels
    image = image.resize((512, 512))

    # Save the cropped and resized image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)

if __name__ == '__main__':
    input_dir = '/mnt/ve_share/songyuhao/generation/data/train/diffusions/lsu_combine/imgs'
    output_dir = '/mnt/ve_share/songyuhao/generation/data/train/diffusions/lsu_combine/512'
    os.makedirs(output_dir, exist_ok=True)

    image_paths = random.sample([os.path.join(input_dir, filename) for filename in os.listdir(input_dir)], k=7768)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print(len(image_paths))
        future_results = [executor.submit(center_crop_resize, image_path, output_dir) for image_path in tqdm(image_paths)]
        concurrent.futures.wait(future_results)
