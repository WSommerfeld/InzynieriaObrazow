import cv2
import numpy as np
from PIL import Image

#elementy strukturalne
def kernel1():
    kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    return kernel

def kernel2():
    kernel = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=np.uint8)
    return kernel

def kernel3():
    kernel  = np.array([[1],[1], [1],[1],[1], [1],[1]], dtype=np.uint8)
    return kernel


def kernel4():
    kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]
                              ], dtype=np.uint8)
    return kernel

def choose_kernel(kernel_no):
    kernel = np.array([0])
    anchor = (0, 0)

    if kernel_no == 1:
        kernel = kernel1()
        anchor = (1, 1)

    if kernel_no == 2:
        kernel = kernel2()
        anchor = (1, 1)

    if kernel_no == 3:
        kernel = kernel3()
        anchor = (0, 4)

    if kernel_no == 4:
        kernel = kernel4()
        anchor = (4, 4)

    return kernel, anchor


def erode(name,kernel_no):

    #wczytanie obrazu jako binarny
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


    #element strukturalny
    kernel, anchor = choose_kernel(kernel_no)


    #erozja
    eroded = cv2.erode(binary, kernel,anchor=anchor,iterations=1)

    #różnica
    difference = cv2.subtract(binary, eroded)

    #zapis
    img = Image.fromarray(eroded)
    img.save(name+"_eroded"+str(kernel_no)+".png")


    img2 = Image.fromarray(difference)
    img2.save(name+"_difference"+str(kernel_no)+".png")


    return eroded

#dylacja

def dilate(name,kernel_no):

    #wczytanie obrazu jako binarny
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    #element strukturalny
    kernel, anchor = choose_kernel(kernel_no)

    #dylacja
    dilated = cv2.dilate(binary, kernel,anchor=anchor, iterations=1)

    #zapis
    img = Image.fromarray(dilated)
    img.save(name+"_dilated"+str(kernel_no)+".png")

    return dilated



#domknięcie

def close(name,kernel_no):
    erode(name,kernel_no)
    dilate(name+"_eroded"+str(kernel_no)+".png",kernel_no)


#otwarcie

def open(name,kernel_no):
    dilate(name, kernel_no)
    erode(name+"_dilated"+str(kernel_no)+".png",kernel_no)

#gradient

def gradient_laplacian(name, kernel_no):
    #wczytanie i binaryzacja
    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



    # dylacja i erozja
    dilation = dilate(name, kernel_no)
    erosion = erode(name, kernel_no)

    # gradient: dylacja - erozja
    gradient = cv2.subtract(dilation, erosion)

    #laplasjan: dylacja + erozja - 2 * oryginał
    binary_16 = np.int16(binary)
    laplacian = np.clip(dilation.astype(np.int16) + erosion.astype(np.int16) - 2 * binary_16, 0, 255).astype(np.uint8)

    img = Image.fromarray(gradient)
    img.save(name+"_gradient"+str(kernel_no)+".png")


    img2 = Image.fromarray(laplacian)
    img2.save(name+"_laplacian"+str(kernel_no)+".png")



def zad_1():
    erode("test2.png", 1)
    erode("test2.png", 2)
    erode("test2.png", 3)
    erode("test2.png",4)


def zad_2():
    erode("test1.png", 1)
    erode("test1.png", 2)
    erode("test1.png", 3)
    erode("test1.png",4)

    dilate("test1.png", 1)
    dilate("test1.png", 2)
    dilate("test1.png", 3)
    dilate("test1.png", 4)

def zad_3():
    close("test3.png", 1)
    close("test3.png", 2)
    close("test3.png", 3)
    close("test3.png", 4)

def zad_4():
    open("test3.png", 1)
    open("test3.png", 2)
    open("test3.png", 3)
    open("test3.png", 4)

def zad_5():
    gradient_laplacian("test1.png", 1)
    gradient_laplacian("test1.png", 2)
    gradient_laplacian("test1.png", 3)
    gradient_laplacian("test1.png", 4)

    gradient_laplacian("test2.png", 1)
    gradient_laplacian("test2.png", 2)
    gradient_laplacian("test2.png", 3)
    gradient_laplacian("test2.png", 4)


zad_5()