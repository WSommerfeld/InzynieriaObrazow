import numpy as np
from PIL import Image
import cv2
from collections import Counter
import heapq


def colors():
    # Generowanie obrazu tęczy
    array = np.zeros((100, 1792, 3), dtype=np.uint8)

    # czarny-niebieski
    for i in range(256):
        for j in range(100):
            array[j, i] = [0, 0, i]

    # niebieski-cyjan
    for i in range(256):
        for j in range(100):
            array[j, 256 + i] = [0, i, 255]

    # cyjan-zielony
    for i in range(256):
        for j in range(100):
            array[j, 2 * 256 + i] = [0, 255, 255 - i]

    # zielony-żółty
    for i in range(256):
        for j in range(100):
            array[j, 3 * 256 + i] = [i, 255, 0]

    # żółty-czerwony
    for i in range(256):
        for j in range(100):
            array[j, 4 * 256 + i] = [255, 255 - i, 0]

    # czerwony-magenta
    for i in range(256):
        for j in range(100):
            array[j, 5 * 256 + i] = [255, 0, i]

    # magenta-biały
    for i in range(256):
        for j in range(100):
            array[j, 6 * 256 + i] = [255, i, 255]

    return array


# Zamiana na model luminacja-chrominancja
def stepOne(RGBarray):
    print(f"stepOne: RGB -> YCbCr. Rozmiar obrazu: {RGBarray.shape}")
    width = RGBarray.shape[1]
    height = RGBarray.shape[0]

    ycbcr = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            R, G, B = RGBarray[j, i]

            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.1687 * R - 0.3313 * G + 0.5* B + 128
            Cr = 0.5* R - 0.4187 * G -0.0813 * B + 128

            ycbcr[j, i] = [np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)]

    return ycbcr


# Próbkowanie
def stepTwo(channel, factor):
    print(f"stepTwo: Próbkowanie (factor={factor}). Rozmiar przed próbkowaniem: {channel.shape}")
    if factor == 1:
        return channel
    channel_sampled = channel[::factor, ::factor]
    print(f"stepTwo: Rozmiar po próbkowaniu: {channel_sampled.shape}")
    return channel_sampled


# Padding na bloki 8x8
def stepThree(arrayChannel):
    print(f"stepThree: Padding. Rozmiar przed paddingiem: {arrayChannel.shape}")
    height, width = arrayChannel.shape
    h_pad = (8 - height % 8) % 8
    w_pad = (8 - width % 8) % 8
    channel_padded = np.pad(arrayChannel, ((0, h_pad), (0, w_pad)), mode='constant')
    print(f"stepThree: Rozmiar po paddingu: {channel_padded.shape}")
    return channel_padded




# Zwinięcie bloków algorytmem ZigZag
def zigzag_block(block):
    zigzag_index = [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
    ]
    return [block[i, j] for i, j in zigzag_index]


def stepSeven(channel):
    h, w = channel.shape
    result = []
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            result.append(zigzag_block(block))
    print(f"stepSeven: ZigZag. Liczba bloków: {len(result)}")
    return result


# Drzewo huffmana + słownik kodowy
def build_huffman_tree(symbols):
    freq = Counter(symbols)
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]: pair[1] = '0' + pair[1]
        for pair in hi[1:]: pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = {symbol: code for symbol, code in heap[0][1:]}
    return huffman_dict


# Kodowanie RLE + Huffman dla bloków ZigZag
def stepEight(zigzag_blocks):
    encoded_blocks = []

    all_symbols = []

    for block in zigzag_blocks:
        rle = []
        zeros = 0
        for val in block[1:]:
            if val == 0:
                zeros += 1
            else:
                rle.append((zeros, val))
                all_symbols.append((zeros, val))
                zeros = 0
        rle.append((0, 0))  # EOB
        all_symbols.append((0, 0))
        encoded_blocks.append(rle)

    # słownik
    huffman_dict = build_huffman_tree(all_symbols)

    # kodowanie bloków
    bitstream = ""
    for rle in encoded_blocks:
        for symbol in rle:
            bitstream += huffman_dict[symbol]

    print(f"stepEight: Huffman. Liczba bitów: {len(bitstream)}")
    return bitstream, huffman_dict


def decode_bitstream(bitstream, huffman_dict, num_blocks):
    reverse_dict = {v: k for k, v in huffman_dict.items()}

    decoded_blocks = []
    current_block = [0]
    buffer = ""
    blocks_decoded = 0

    for bit in bitstream:
        buffer += bit
        if buffer in reverse_dict:
            symbol = reverse_dict[buffer]
            if symbol == (0, 0):
                current_block += [0] * (64 - len(current_block))
                decoded_blocks.append(current_block)
                current_block = [0]
                blocks_decoded += 1
                if blocks_decoded >= num_blocks:
                    break
            else:
                zeros, val = symbol
                current_block += [0] * zeros + [val]
            buffer = ""

    print(f"decode_bitstream: Liczba zdekodowanych bloków: {len(decoded_blocks)}")
    return decoded_blocks


# Konwersja z modelu ycbcr na RGB i zapis
def save_ycbcr(ycbcr_array, filename):
    Y, Cb, Cr = ycbcr_array[:, :, 0], ycbcr_array[:, :, 1], ycbcr_array[:, :, 2]
    buff=Cb
    Cb=Cr
    Cr=buff
    ycbcr_array[:, :, 0]=Y
    ycbcr_array[:, :, 1]=Cr
    ycbcr_array[:, :, 2]=Cb
    rgb_image = cv2.cvtColor(ycbcr_array, cv2.COLOR_YCrCb2RGB)
    image = Image.fromarray(rgb_image)
    image.save(filename)


# Wyświetlanie obrazu
def show_image(filename):
    image = Image.open(filename)
    image.show()


def inverse_zigzag(zigzag):
    # Przekształcenie z powrotem z formatu ZigZag do 8x8 blok
    zigzag_index = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
    ]
    # Tworzenie pustej macierzy 8x8 z typem danych np.int16
    block = np.zeros((8, 8), dtype=np.int16)

    # Wypełnianie bloków w odwrotnej kolejności
    for idx, (i, j) in enumerate(zigzag_index):
        block[i, j] = zigzag[idx]

    return block

def process(sampling_factor):
    print(f"\n=== Próbkowanie {sampling_factor}x ===")

    # Krok 0: Obraz RGB (tęcza)
    img = colors()
    original_h, original_w = img.shape[:2]  # Zapisz oryginalne wymiary

    # Krok 1: RGB -> YCbCr
    ycbcr = stepOne(img)
    save_ycbcr(ycbcr, "input.jpg")

    # Kanały Y, Cb, Cr
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    # Krok 2: Próbkowanie
    Y_s = Y
    Cb_s = stepTwo(Cb, sampling_factor)
    Cr_s = stepTwo(Cr, sampling_factor)

    # Krok 3: Padding na bloki 8x8
    Y_p = stepThree(Y_s)
    Cb_p = stepThree(Cb_s)
    Cr_p = stepThree(Cr_s)

    # Krok 7: ZigZag
    Y_zz = stepSeven(Y_p)
    Cb_zz = stepSeven(Cb_p)
    Cr_zz = stepSeven(Cr_p)

    # Krok 8: Huffman + RLE
    y_stream, y_dict = stepEight(Y_zz)
    cb_stream, cb_dict = stepEight(Cb_zz)
    cr_stream, cr_dict = stepEight(Cr_zz)

    # Dekodowanie
    y_blocks = decode_bitstream(y_stream, y_dict, len(Y_zz))
    cb_blocks = decode_bitstream(cb_stream, cb_dict, len(Cb_zz))
    cr_blocks = decode_bitstream(cr_stream, cr_dict, len(Cr_zz))

    # Rekonstrukcja kanałów
    def rebuild_channel(blocks, h, w):
        channel = np.zeros((h, w), dtype=np.uint8)
        i = 0
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if i < len(blocks):
                    block = inverse_zigzag(blocks[i])
                    block = np.clip(block, 0, 255).astype(np.uint8)
                    if y + 8 <= h and x + 8 <= w:
                        channel[y:y+8, x:x+8] = block
                    i += 1
        return channel

    Y_rec = rebuild_channel(y_blocks, Y_p.shape[0], Y_p.shape[1])
    Cb_rec = rebuild_channel(cb_blocks, Cb_p.shape[0], Cb_p.shape[1])
    Cr_rec = rebuild_channel(cr_blocks, Cr_p.shape[0], Cr_p.shape[1])

    # Upsampling i przycięcie do oryginalnych wymiarów
    if sampling_factor > 1:
        Cb_rec = cv2.resize(Cb_rec, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        Cr_rec = cv2.resize(Cr_rec, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    else:
        Cb_rec = Cb_rec[:original_h, :original_w]  # Przycięcie paddingu
        Cr_rec = Cr_rec[:original_h, :original_w]

    Y_rec = Y_rec[:original_h, :original_w]  # Przycięcie paddingu

    #  YCbCr -> RGB
    ycbcr_rec = np.stack([Y_rec, Cb_rec, Cr_rec], axis=-1)
    '''
    save_ycbcr( np.stack([Y_rec, np.zeros_like(Y_rec), np.zeros_like(Y_rec)], axis=-1), "outputY" + str(sampling_factor) + ".jpg")
    save_ycbcr(np.stack([np.zeros_like(Y_rec), Cb_rec, np.zeros_like(Y_rec)], axis=-1), "outputCb" + str(sampling_factor) + ".jpg")
    save_ycbcr(np.stack([np.zeros_like(Y_rec), np.zeros_like(Y_rec), Cr_rec], axis=-1), "outputCr" + str(sampling_factor) + ".jpg")
    '''
    save_ycbcr(ycbcr_rec, "output"+str(sampling_factor)+".jpg")
    show_image("output"+str(sampling_factor)+".jpg")

process(1)
process(2)
process(4)