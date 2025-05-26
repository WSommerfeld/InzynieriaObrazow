"""Function definitions that are used in LSB steganography."""
#from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

from PIL import Image

#plt.rcParams["figure.figsize"] = (18,10)
def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = [ "{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)

def decode_from_binary_array(array):
    """Decode a binary string to utf8."""
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = [ "{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")


def load_image(path, pad=False):
    """Load an image.
     If pad is set then pad an image to multiple of 8 pixels.
     """
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
        image, ((0, y_pad), (0, x_pad) ,(0, 0)), mode='constant')
    return image

def save_image(path, image):
    """Save an image."""
    #plt.imsave(path, image)
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
        img.save(path)
    else:
        raise TypeError("Input must be a NumPy array")

def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)

def hide_message(image, message, nbits):
    """Hide a message in an image (LSB).
     nbits: number of least significant bits
     """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) > len(image) * nbits:
        raise ValueError("Message is to long :(")

    chunks = [message[i:i + nbits] for i in range(0, len(message),
        nbits)]

    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i])
        new_byte = byte[:-nbits] + chunk
        image[i] = int(new_byte, 2)
    return image.reshape(shape)


def reveal_message(image, nbits=1, length=0):
    """Reveal the hidden message.
     nbits: number of least significant bits
     length: length of the message in bits.
     """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length/nbits)
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)

    message = ""
    i = 0
    while i < length_in_pixels:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        i += 1
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message

def hide_image(image, secret_image_path, nbits=1):
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
    secret_img = secret_img.hex()
    secret_img = [secret_img[i:i + 2] for i in range(0,len(secret_img), 2)]
    secret_img = ["{:08b}".format(int(el, base=16)) for el in
    secret_img]
    secret_img = "".join(secret_img)
    return hide_message(image, secret_img, nbits), len(secret_img)

#zad3
def hide_messagepos(image, message, nbits=1, spos=0):

    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    available_bits = (len(image) - spos) * nbits
    if len(message) > available_bits:
        raise ValueError("Message is too long for the given starting position")

    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]

    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[spos + i])
        new_byte = byte[:-nbits] + chunk
        image[spos + i] = int(new_byte, 2)
    return image.reshape(shape)

#zad3
def reveal_messagepos(image, nbits=1, length=0, spos=0):

    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    available_bits = (len(image) - spos) * nbits

    if length <= 0:
        length = available_bits
    else:
        length = min(length, available_bits)

    length_in_pixels = math.ceil(length / nbits)
    message = ""
    i = 0
    while i < length_in_pixels and (spos + i) < len(image):
        byte = "{:08b}".format(image[spos + i])
        message += byte[-nbits:]
        i += 1

    if length > 0:
        message = message[:length]
    return message





def reveal_image(image, length, nbits=1):

    binary_data = reveal_message(image, nbits=nbits, length=length)

    bytes_data = []
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        bytes_data.append(int(byte, 2))



    with open("recovered_secret.png", "wb") as f:
        f.write(bytes(bytes_data))

    print("Image recovered successfully!")
    return bytes(bytes_data)






#zad 5
def reveal_image_until_footer(image_name, nbits=1, footer=b'\x00\x00\x00\x00IEND\xaeB`\x82'):
    stego_image = load_image(image_name)

    with open(image_name, "rb") as file:
        secret_img = file.read()
    secret_img_hex = secret_img.hex()
    secret_img_binary = ["{:08b}".format(int(el, base=16)) for el in
                         [secret_img_hex[i:i + 2] for i in range(0, len(secret_img_hex), 2)]]
    secret_img_binary = "".join(secret_img_binary)

    binary_data = reveal_message(stego_image, nbits=nbits, length=len(secret_img_binary))

    bytes_data = []
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        bytes_data.append(int(byte, 2))

        if len(bytes_data) >= 12:
            f=bytes(bytes_data[-12:])
            if f == footer:

                print("Found footer!")
                break

    with open("recovered_secret0.png", "wb") as f:
        f.write(bytes(bytes_data))

    print("Image recovered successfully!")
    return bytes(bytes_data)














