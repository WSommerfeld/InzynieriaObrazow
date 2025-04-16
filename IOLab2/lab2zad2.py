import numpy as np
from PIL import Image

# zapis do formatu p3: PPM tekstowy
def save_ppm_p3(filename, image_array):
    height, width, _ = image_array.shape
    #odczyt do zapisu
    with open(filename, "w") as f:
        f.write(f"P3\n{width} {height}\n255\n") #nagłówek; format i rozmiar i max wartość RGB
        for row in image_array:
            for pixel in row:
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ") #zapis pikselmapy
            f.write("\n")

def gray():
    array = np.zeros((100, 256, 3), dtype=np.uint8)

    for i in range(256):
        for j in range(100):
            array[j, i] = [i,i,i]

    save_ppm_p3("gray.ppm", array)

    return array

def colors():
    transition_points = [
        (0, 0, 0),  # 1 - czarny
        (0, 0, 255),  # 2 - niebieski
        (0, 255, 255),  # 3 - cyjan
        (0, 255, 0),  # 4 - zielony
        (255, 255, 0),  # 5 - żółty
        (255,0,0),  # 6 - czerwony
        (255, 0, 255),  # 7 - magenta
        (255, 0, 255),  # 8 - biały
    ]

    array = np.zeros((100, 1792, 3), dtype=np.uint8)

    #czarny-niebieski
    for i in range(256):
            for j in range(100):
                array[j, i] = [0,0,i]

    #niebieski-cyjan
    for i in range(256):
            for j in range(100):
                array[j, 256+i] = [0,i,255]

    #cyjan-zielony
    for i in range(256):
            for j in range(100):
                array[j, 2*256+i] = [0,255,255-i]

    #zielony-żółty
    for i in range(256):
            for j in range(100):
                array[j, 3*256+i] = [i,255,0]

    #żółty-czerwony
    for i in range(256):
            for j in range(100):
                array[j, 4*256+i] = [255,255-i,0]

    # czerwony-magenta
    for i in range(256):
        for j in range(100):
            array[j, 5 * 256 + i] = [255, 0, i]

    # magenta-biały
    for i in range(256):
        for j in range(100):
            array[j, 6 * 256 + i] = [255, i, 255]



    save_ppm_p3("colors.ppm", array)
    return array

def show_image(filename):
    image = Image.open(filename)
    image.show()


colors()
gray()
show_image("colors.ppm")
show_image("gray.ppm")