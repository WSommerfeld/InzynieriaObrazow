import sys
import lab5


while True:
    print("Wybierz polecenie: ")
    print("1. Kule")
    print("2. Promienie wtorne")
    print("3. Cienie")
    print("4. Przezroczystosc")
    print("5. Trojkat")
    print("Q. Wyjście")
    choose=input()


    match choose:
        case "1":
            lab5.zad1()

        case "2":
            lab5.zad2()

        case "3":
            lab5.zad3()

        case "4":
            lab5.zad4()

        case "5":
            lab5.zad5()


        case "q":
             print("q")
             sys.exit()

        case "Q":
             print("q")
             sys.exit()

        case _:
            print("x")