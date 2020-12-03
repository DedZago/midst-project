import os
import pickle
import sys

# stampare a video pickle file (load permette di leggere una lista alla volta: ne ho caricate n)
# nota: per i centroidi basta fare un pickle solo poiche' si ha solo una lista di liste
def print_file(file_to_print):
    with open(file_to_print, "rb") as f:
        while True:
            try:
                print(pickle.load(f))
            except EOFError:
                break


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        print("[ERROR] Launch the module as follows: python3 print_file.py <file_to_print>")
    else:
        try:
            print_file(sys.argv[1])
        except:
            raise
            print("[ERROR] Something goes wrong while reading the file.")
