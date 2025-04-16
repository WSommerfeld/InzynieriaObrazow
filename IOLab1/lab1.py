import sys

import lab1zad1
import lab1zad2
import lab1zad3
import lab1zad4
import lab1zad5
from lab1zad3 import ycbcr

while True:
    print("Wybór zadania: 1 - 5 ")
    print("Wyjście: q")
    choose = input()
    if choose == "q":
        sys.exit()

    elif choose == "1":
        lab1zad1.show_image("input1.jpg")
        lab1zad1.save(lab1zad1.hpfilter("input1.jpg"),"input1")
        lab1zad1.show_image("input1-hpfoutput.jpg")

    elif choose == "2":
        lab1zad2.save_float(lab1zad2.inttofloat(lab1zad2.file_to_array("input1.jpg")), "input1")
        lab1zad2.show_image("input1-floatoutput.jpg")

    elif choose == "3":
        array=lab1zad3.RGBtoYCBCR(lab1zad3.file_to_array("c19.png"))
        lab1zad3.save_ycbcr(array,"ycbcr1.png")
        lab1zad3.ycbcr(array)
        lab1zad3.show_image("Y.png")
        lab1zad3.show_image("Cb.png")
        lab1zad3.show_image("Cr.png")
        lab1zad3.show_image("ycbcr1.png")

    elif choose == "4":
        lab1zad4.process("input1.jpg")
        lab1zad4.show_image("input1.jpg-transmitted.jpg")

    elif choose == "5":
        lab1zad5.process("input1.jpg", "input1.jpg-transmitted.jpg")