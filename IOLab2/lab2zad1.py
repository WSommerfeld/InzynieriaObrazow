import numpy as np
import os
from PIL import Image


# odczyt ppm (do tablicy numpy)
def read_ppm(filename):
    with open(filename, "rb") as f:
        header = f.readline().decode().strip()
        width, height = map(int, f.readline().decode().split())
        max_color = int(f.readline().decode().strip())

        if max_color > 255:
            raise ValueError("Błędna głębia bitowa! To nie jest plik PPM!")

        #dla p3
        if header == "P3":
            data = []
            for line in f:
                data.extend(map(int, line.decode().split()))
            image_array = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        #dla p6
        elif header == "P6":
            data = np.frombuffer(f.read(), dtype=np.uint8)
            image_array = data.reshape((height, width, 3))
        else:
            raise ValueError("Błędny nagłówek!")

    return image_array



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

# p6 (binarny)
def save_ppm_p6(filename, image_array):
    height, width, _ = image_array.shape
    with open(filename, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(image_array.tobytes())

#odczyt pliku jpg/png do numpy array
def read_image(filename):

    image = Image.open(filename)
    image_array = np.array(image)
    return image_array

#konwersja png/jpg do ppm p3
def save_as_ppm_p3(filename):

    save_ppm_p3(filename+".ppm", read_image(filename))

#konwersja png/jpg do ppm p6
def save_as_ppm_p6(filename):
    save_ppm_p6(filename+".ppm", read_image(filename))

def show_image(filename):
    image = Image.open(filename)
    image.show()

#testy
show_image("photo.jpg")
#1.zapis do ppm p3
save_as_ppm_p3("photo.jpg")
#2.zapis do ppm p6
save_as_ppm_p6("photo1.jpg")

show_image("photo.jpg")
show_image("photo1.jpg")
#3.odczyt ww plików
array1 = read_ppm("photo.jpg.ppm")
array2 = read_ppm("photo1.jpg.ppm")
#4. porównanie rozmiaru
print("\n")
file1 = 'photo.jpg.ppm'
file2 = 'photo1.jpg.ppm'

# Pobranie rozmiarów plików
size1 = os.path.getsize(file1)
size2 = os.path.getsize(file2)
#Wyświetlenie rozmiaru plików
print("Rozmiar pliku w formacie p3: ", size1, " bajtów")
print("Rozmiar pliku w formacie p6: ", size2," bajtów" )
print("Stosunek p3/p6: ", size1/size2)



