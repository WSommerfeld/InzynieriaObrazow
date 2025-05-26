import sys
import lab4


while True:
    print("Wybierz polecenie: ")
    print("1. Dithering w skali szarości")
    print("2. Dithering w kolorze")
    print("3. Linie i trójkąt")
    print("4. Interpolacja kolorów")
    print("5. SSAA")
    print("Q. Wyjście")
    choose=input()


    match choose:
        case "1":

            lab4.process_1a('munch.png')
            lab4.process_1('munch.png')



        case "2":
            lab4.process_2('munch.png')

        case "3":
            lab4.process_3a()
            lab4.process_3()



        case "4":
            lab4.process_4()

        case "5":
            lab4.process_5()
            lab4.process_5a()


        case "q":
             print("q")
             sys.exit()

        case "Q":
             print("q")
             sys.exit()

        case _:
            print("x")