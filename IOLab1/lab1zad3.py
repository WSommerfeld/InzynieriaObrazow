import numpy as np
from PIL import Image
import cv2


def file_to_array(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)
    return img_array

# Zamiana na model luminacja-chrominancja
def RGBtoYCBCR(RGBarray):
    width = RGBarray.shape[1]
    height = RGBarray.shape[0]

    ycbcr = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            R, G, B = RGBarray[j, i]

            Y = 0.229 * R + 0.587 * G + 0.114 * B
            Cb = 0.500 * R - 0.418 * G - 0.082* B + 128
            Cr = -0.168 * R - 0.331 * G + 0.5 * B + 128

            ycbcr[j, i] = [np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)]

    return ycbcr

# Konwersja z modelu ycbcr na RGB i zapis
def save_ycbcr(ycbcr_array, filename):
    rgb_image = cv2.cvtColor(ycbcr_array, cv2.COLOR_YCrCb2RGB)
    image = Image.fromarray(rgb_image)
    image.save(filename)

def save_image(array, filename):
    image = Image.fromarray(array)
    image.save(filename)

def show_image(filename):
    image = Image.open(filename)
    image.show()

def ycbcr(ycbcr_array):
    # Składowe Y, Cb, Cr
    Y = ycbcr_array[:, :, 0]
    Cb = ycbcr_array[:, :, 1]
    Cr = ycbcr_array[:, :, 2]

    save_image(Y, "Y.png")
    save_image(Cb, "Cb.png")
    save_image(Cr, "Cr.png")


#save_ycbcr(RGBtoYCBCR(file_to_array("c19.png")),"ycbcr1.png")
#(RGBtoYCBCR(file_to_array("c19.png")))
#show_image("Y.png")
#show_image("Cb.png")
#show_image("Cr.png")
#show_image("ycbcr1.png")