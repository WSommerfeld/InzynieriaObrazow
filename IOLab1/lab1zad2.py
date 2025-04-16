import numpy as np
from PIL import Image

def file_to_array(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)
    return img_array

def inttofloat(img_array):
    width=img_array.shape[1]
    height=img_array.shape[0]
    output = np.zeros_like(img_array, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r, g, b = img_array[i, j] / 255.0

            rnew = 0.393*r+0.769*g+0.189*b
            gnew = 0.349*r+0.689*g+0.168*b
            bnew = 0.272*r+0.534*g+0.131*b

            output[i, j] = [min(rnew, 1.0), min(gnew, 1.0), min(bnew, 1.0)]

    return output

def save_float(array, name):
    array_uint8 = (array * 255).astype(np.uint8)
    image = Image.fromarray(array_uint8)
    image.save(name + "-floatoutput.jpg")

def show_image(filename):
    image = Image.open(filename)
    image.show()


#save_float(inttofloat(file_to_array("input1.jpg")), "input1")
#show_image("input1-floatoutput.jpg")
