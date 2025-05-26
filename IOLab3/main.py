import lab3
import steganography
import JPEG
import sys
import os

while True:
    print("Wybierz polecenie: ")
    print("1. Algorytm JPEG")
    print("2. Ukrycie wiadomości w obrazku")
    print("3. Badanie wpływu liczby użytych najmniej znaczących bitów")
    print("4. Zapis wiadomości od zadanej pozycji")
    print("5. Odkrycie obrazka z obrazka")
    print("6. Odkrycie obrazka z użyciem stopki")
    print("Q. Wyjście")
    choose=input()


    match choose:
        case "1":



            JPEG.save_image(JPEG.colors(), "original_image.png")
            reconstructed_image = JPEG.process(2, 100)
            JPEG.save_image(reconstructed_image, "reconstructed_image.png")

            JPEG.process_mse("original_image.png", "reconstructed_image.png")
            size = os.path.getsize('reconstructed_image.png')
            print(f"Rozmiar pliku na dysku: {size} bajtów")
            print("Oryginalny obraz: original_image.png ")
            print("Obraz po kompresji i dekompresji: reconstructed_image.png")
            JPEG.show_image("original_image.png")
            JPEG.show_image("reconstructed_image.png")



        case "2":
            nLSB=2
            msg = "Stanisław Wyspiański, Macierzyństwo,1905"
            binary=steganography.encode_as_binary_array(msg)
            length=len(binary)
            print("Liczba najmniej znaczących bitów użyta do ukrycia wiadomości: ", nLSB)
            lab3.hide("img1.png",msg, nLSB)
            print("Wiadomość ukryto w obrazie: img1.png_with_message_"+str(nLSB)+".png")
            lab3.show("img1.png_with_message_"+str(nLSB)+".png",nLSB,length)

        case "3":
            nLSB = 1
            msg = "Stanisław Wyspiański, Macierzyństwo,1905"
            msg = msg * 4000
            binary = steganography.encode_as_binary_array(msg)
            length = len(binary)


            for i in range(8):
                lab3.hide("img1.png", msg, nLSB)
                lab3.show("img1.png_with_message_" + str(nLSB) + ".png", nLSB, length)


                nLSB = nLSB + 1

            for i in range(8):
                print("Plik: img1.png_with_message_" + str(i+1) + ".png")
                print("n=", i + 1)
                lab3.process_mse("img1.png", "img1.png_with_message_" + str(i + 1) + ".png")

        case "4":
            nLSB=2
            spos=2025
            msg = "Stanisław Wyspiański, Macierzyństwo,1905"
            binary=steganography.encode_as_binary_array(msg)
            length=len(binary)
            print("Liczba najmniej znaczących bitów użyta do ukrycia wiadomości: ", nLSB)
            lab3.hide_pos("img1.png",msg, nLSB,spos)
            print("Wiadomość ukryto w obrazie: img1.png_with_message_pos_"+str(spos)+"_"+str(nLSB)+".png")
            lab3.show_pos("img1.png_with_message_pos_"+str(spos)+"_"+str(nLSB)+".png",nLSB,length,spos)

        case "5":
            nLSB=2
            image_with_secret, length_of_secret=lab3.image_in_image("img1.png", "img2.png", nLSB)
            lab3.show_image_in_image(image_with_secret, length_of_secret, nLSB)

        case "6":
            nLSB = 4
            name1="img1.png"
            name2="img2.png"
            image_with_secret= lab3.image_in_image(name1, name2, nLSB)
            lab3.reveal_image_with_0_save(name2 + "_in_"+name1 + str(nLSB) + ".png", nLSB, "recovered_img2_with0.png")

        case "q":
             print("q")
             sys.exit()

        case "Q":
             print("q")
             sys.exit()

        case _:
            print("x")