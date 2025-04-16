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

# ycbcr na rgb
def YCBCRtoRGB(ycbcr_array):
    height, width = ycbcr_array.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            Y, Cb, Cr = ycbcr_array[j, i]

            R = Y + 1.402 * (Cr - 128)
            G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
            B = Y + 1.772 * (Cb - 128)

            rgb[j, i] = [np.clip(R, 0, 255), np.clip(G, 0, 255), np.clip(B, 0, 255)]

    return rgb



def downsample(Cb, Cr):
    Cb_downsampled = Cb[::2, ::2]
    Cr_downsampled = Cr[::2, ::2]
    return Cb_downsampled, Cr_downsampled



def upsample(Cb_down, Cr_down, height, width):
    Cb_upsampled = cv2.resize(Cb_down, (width, height), interpolation=cv2.INTER_LINEAR)
    Cr_upsampled = cv2.resize(Cr_down, (width, height), interpolation=cv2.INTER_LINEAR)
    return Cb_upsampled, Cr_upsampled


# zapis obrazu z tablicy
def save_image(array, filename):
    image = Image.fromarray(array)
    image.save(filename)

def show_image(filename):
    image = Image.open(filename)
    image.show()


def process(name):
    img_array = file_to_array(name)

    # konwersja na YCbCr
    ycbcr = RGBtoYCBCR(img_array)

    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    # downsampling kanałów Cb i Cr
    Cb_down, Cr_down = downsample(Cb, Cr)

    # upsampling kanałów Cb i Cr do oryginalnych rozmiarów
    Cb_upsampled, Cr_upsampled = upsample(Cb_down, Cr_down, img_array.shape[0], img_array.shape[1])

    # złożenie Y, Cb i Cr z powrotem do YCbCr
    ycbcr_transmitted = np.zeros_like(ycbcr)
    ycbcr_transmitted[:, :, 0] = Y
    ycbcr_transmitted[:, :, 1] = Cb_upsampled
    ycbcr_transmitted[:, :, 2] = Cr_upsampled

    # konwersja z YCbCr z powrotem do RGB
    rgb_transmitted = YCBCRtoRGB(ycbcr_transmitted)


    save_image(Y, "tY.jpg")
    save_image(Cb, "tCb.jpg")
    save_image(Cr, "tCr.jpg")
    save_image(rgb_transmitted, name+"-transmitted.jpg")





#process("input1.jpg")



#show_image("input1.jpg-transmitted.jpg")
