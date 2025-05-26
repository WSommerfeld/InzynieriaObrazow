"""Function definitions that are used in LSB steganography."""

import numpy as np

from PIL import Image
import steganography

'''
message = "Moja tajna wiadomość"
binary = encode_as_binary_array(message)
print("Binary:", binary)
message = decode_from_binary_array(binary)
print("Retrieved message:", message)
'''

#zad 1
def hide(img_name, msg, nLSB):
    original_image = steganography.load_image(img_name)
    message = msg
    message = steganography.encode_as_binary_array(message)

    image_with_message = steganography.hide_message(original_image, message, nLSB)
    steganography.save_image(img_name+"_with_message_"+str(nLSB)+".png", image_with_message)

#zad 1
def show(img_name, nLSB, length):


    image_with_message_png =  steganography.load_image(img_name)
    secret_message_png =  steganography.decode_from_binary_array(steganography.reveal_message(
        image_with_message_png, nbits=nLSB, length=length))
    print(secret_message_png)

# Wyświetlanie obrazu
def show_image(filename):
    image = Image.open(filename)
    image.show()


#zad2
def calculate_mse(image1, image2):
    diff = image1 - image2
    mse = np.mean(np.square(diff))
    return mse

def file_to_array(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)
    return img_array
#zad2
def process_mse(name1, name2):
    original_image = file_to_array(name1)
    transmitted_image = file_to_array(name2)
    mse = calculate_mse(original_image, transmitted_image)
    print(f"Mean Squared Error (MSE): {mse}")

#zad3
def hide_pos(img_name, msg, nLSB,spos):
    original_image = steganography.load_image(img_name)
    message = msg
    message = steganography.encode_as_binary_array(message)

    image_with_message = steganography.hide_messagepos(original_image, message, nLSB,spos)
    steganography.save_image(img_name+"_with_message_pos_"+str(spos)+"_"+str(nLSB)+".png", image_with_message)

#zad3
def show_pos(img_name, nLSB, length,spos):


    image_with_message_png =  steganography.load_image(img_name)
    secret_message_png =  steganography.decode_from_binary_array(steganography.reveal_messagepos(
        image_with_message_png, nbits=nLSB, length=length,spos=spos))
    print(secret_message_png)

#zad4
def image_in_image(name1,name2, nLSB):
    image=steganography.load_image(name1)
    image_with_secret, length_of_secret = steganography.hide_image(image,name2,nLSB)
    steganography.save_image(name2 + "_in_"+name1 + str(nLSB) + ".png", image_with_secret)
    return image_with_secret,length_of_secret
#zad4
def show_image_in_image(img, length, nLSB):
    steganography.reveal_image(img, length, nLSB)


#zad5
def reveal_image_with_0_save(stego_image_name, nLSB, output_filename):
    steganography.reveal_image_until_footer(stego_image_name, nLSB)

    print(f"Odkryty obrazek: {output_filename}")






