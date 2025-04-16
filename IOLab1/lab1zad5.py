import numpy as np
from PIL import Image
import cv2


def calculate_mse(image1, image2):
    diff = image1 - image2
    mse = np.mean(np.square(diff))
    return mse

def file_to_array(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)
    return img_array

def process(name1, name2):
    original_image = file_to_array(name1)
    transmitted_image = file_to_array(name2)

    mse = calculate_mse(original_image, transmitted_image)

    print(f"Mean Squared Error (MSE): {mse}")


#process("input1.jpg", "input1.jpg-transmitted.jpg")