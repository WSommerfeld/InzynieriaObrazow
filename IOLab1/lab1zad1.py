import numpy as np
from PIL import Image


def hpfilter(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)

    # Maska filtru
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])


    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2


    output = np.zeros_like(img_array, dtype=np.float32)


    for channel in range(3):
        channel_data = img_array[:, :, channel]
        padded = np.pad(channel_data, ((pad_h, pad_w), (pad_w, pad_h)), mode='reflect')

        for i in range(channel_data.shape[0]):
            for j in range(channel_data.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                value = np.sum(region * kernel)
                output[i, j, channel] = value

    # przycięcie wartości do [0, 255] i konwersja do uint8
    output_clipped = np.clip(output, 0, 255).astype(np.uint8)

    return output_clipped

def save(array,name):
    image = Image.fromarray(array)
    image.save(name+"-hpfoutput.jpg")

# Wyświetlanie obrazu
def show_image(filename):
    image = Image.open(filename)
    image.show()


#show_image("input1.jpg")
#save(hpfilter("input1.jpg"),"input1")
#show_image("input1-hpfoutput.jpg")
