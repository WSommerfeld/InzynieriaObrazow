import sys
import lab6


while True:
    print("Wybierz polecenie: ")
    print("1. Erozja")
    print("2. Dylatacja")
    print("3. Domkniecie")
    print("4. Otwarcie")
    print("5. Gradient i laplasjan")
    print("Q. Wyjście")
    choose=input()


    match choose:
        case "1":
            lab6.zad_1()

        case "2":
            lab6.zad_2()

        case "3":
            lab6.zad_3()

        case "4":
            lab6.zad_4()

        case "5":
            lab6.zad_5()


        case "q":
             print("q")
             sys.exit()

        case "Q":
             print("q")
             sys.exit()

        case _:
            print("x")