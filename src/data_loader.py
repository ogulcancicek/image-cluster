import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

EXTENSIONS = ("jpg", "JPG", "jpeg", "JPEG", "png", "PNG")


def filter_images(list_of_images):
    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    return [
        file
        for file in list_of_images
        if any(file.endswith(ext) for ext in valid_extensions)
    ]


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image


def read_image_paths_from_dir(dir_path):
    files = os.listdir(dir_path)
    files = sorted(files, key=(lambda x: int(x.split(".")[0])))
    image_files = filter_images(files)
    return image_files


def read_images_from_dir(dir_path):
    files = os.listdir(dir_path)
    files = sorted(files, key=(lambda x: int(x.split(".")[0])))
    image_files = filter_images(files)
    image_paths = [os.path.join(dir_path, file) for file in image_files]
    images = [
        load_image(image_path)
        for image_path in tqdm(image_paths, desc="Loading images", unit="image")
    ]
    return images


def plot_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def max_resolution_rescale(image, max_width, max_height):
    width, height = image.size
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height))
    return image


def min_resolution_filter(image, min_width, min_height):
    width, height = image.size
    return width >= min_width and height >= min_height


def smart_crop(image, square=True):
    img = np.array(image)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create(edgeThreshold=8)
    kp = sift.detect(gray, None)

    all_points = [i.pt for i in kp]
    x_points = [z[0] for z in all_points]
    y_points = [z[1] for z in all_points]
    thresh = 0
    x_min, y_min = int(min(x_points)) - thresh, int(min(y_points) - thresh)
    x_max, y_max = int(max(x_points)) + thresh, int(max(y_points) + thresh)
    min_side = min((x_max - x_min), (y_max - y_min))
    max_side = max((x_max - x_min), (y_max - y_min))
    x_mean, y_mean = int((x_max + x_min) / 2), int((y_max + y_min) / 2)
    # img = cv.drawKeypoints(img, kp, img)
    squared_x_min, squared_x_max = x_mean - int(min_side / 2), x_mean + int(
        min_side / 2
    )
    squared_y_min, squared_y_max = y_mean - int(min_side / 2), y_mean + int(
        min_side / 2
    )

    if not square:
        return img[y_min:y_max, x_min:x_max]

    elif square:
        return img[squared_y_min:squared_y_max, squared_x_min:squared_x_max]


def save_image(image, path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError("Image must be a numpy array or a PIL image.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(path)
    logging.info(f"Saved image to {path}")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    else:
        print(f"Directory {dir_path} already exists.")


def save_images_to_dir(images, dir_path):
    create_dir(dir_path)

    for idx, image in enumerate(images):
        save_path = os.path.join(dir_path, f"{idx}.png")
        save_image(image, save_path)

    return True


def create_sub_dirs(n_clusters, output_folder):
    for i in range(n_clusters):
        folder_path = os.path.join(output_folder, str(i))
        create_dir(folder_path)
