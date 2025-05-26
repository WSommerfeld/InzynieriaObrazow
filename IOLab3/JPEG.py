import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
from collections import Counter
import heapq

#spektrum przejść RGB
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

#krok 1 : konwersja z RGB do YCbCr
def stepOne(RGBarray):
    print(f"stepOne: RGB -> YCbCr. Rozmiar obrazu: {RGBarray.shape}")
    width = RGBarray.shape[1]
    height = RGBarray.shape[0]

    ycbcr = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            R, G, B = RGBarray[j, i]

            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
            Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

            ycbcr[j, i] = [np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)]

    return ycbcr

# odwrotny krok 1: konwersja z YCbCr do RGB
def reverse_stepOne(ycbcr_array):
    height, width = ycbcr_array.shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        for i in range(width):
            Y, Cb, Cr = ycbcr_array[j, i]

            R = Y + 1.402 * (Cr - 128)
            G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
            B = Y + 1.772 * (Cb - 128)

            rgb[j, i] = [np.clip(R, 0, 255), np.clip(G, 0, 255), np.clip(B, 0, 255)]

    return rgb
#krok 2: podpróbkowanie
def stepTwo(channel, factor):
    print(f"stepThree: Próbkowanie (factor={factor}). Rozmiar przed próbkowaniem: {channel.shape}")
    if factor == 1:
        return channel
    channel_sampled = channel[::factor, ::factor]
    print(f"stepThree: Rozmiar po próbkowaniu: {channel_sampled.shape}")
    return channel_sampled

#odwrotny krok 2: nadpróbkowanie
def reverse_stepTwo(channel_sampled, factor):
    if factor == 1:
        return channel_sampled

    from scipy.ndimage import zoom
    channel_upsampled = zoom(channel_sampled, factor, order=1)
    return channel_upsampled

# wyrównanie
def padding(arrayChannel):
    print(f"stepTwo: Padding. Rozmiar przed paddingiem: {arrayChannel.shape}")
    height, width = arrayChannel.shape
    h_pad = (8 - height % 8) % 8
    w_pad = (8 - width % 8) % 8
    channel_padded = np.pad(arrayChannel, ((0, h_pad), (0, w_pad)), mode='reflect')
    print(f"stepTwo: Rozmiar po paddingu: {channel_padded.shape}")
    return channel_padded

#usunięcie wyrównania
def remove_padding(channel_padded, original_height, original_width):
    print(f"remove_padding: Rozmiar przed usunięciem paddingu: {channel_padded.shape}")
    channel_unpadded = channel_padded[:original_height, :original_width]
    print(f"remove_padding: Rozmiar po usunięciu paddingu: {channel_unpadded.shape}")
    return channel_unpadded

#krok 3: podział na bloki 8x8
def stepThree(channel):

    height, width = channel.shape
    blocks = []
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel[i:i+8, j:j+8]
            blocks.append(block)
    return blocks

#odwrotny krok 3: scalenie bloków 8x8
def reverse_stepThree(blocks, original_height, original_width):
    print(f"reverse_stepThree: Scalanie bloków. Liczba bloków: {len(blocks)}")
    padded_height = ((original_height + 7) // 8) * 8
    padded_width = ((original_width + 7) // 8) * 8

    channel = np.zeros((padded_height, padded_width), dtype=np.float32)
    idx = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            channel[i:i + 8, j:j + 8] = blocks[idx]
            idx += 1
    return channel

#dyskretna transformata kosinusowa
def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

#odwrotna dyskretna transformata kosinusowa
def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1,
    norm='ortho')

#krok 4: transformata kosinusowa każdego bloku
def stepFour(blocks):
    blocks=[dct2(block) for block in blocks]
    return blocks

#odwrotny krok 4: odwrotna transformata kosinusowa każdego bloku
def reverse_stepFour(blocks):
    blocks = [idct2(block) for block in blocks]
    return blocks



# macierz kwantyzacji dla luminancji
def get_quantization_matrix_luminance(QF):
    Q_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 48, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)

    if QF < 50:
        S = 5000 / QF
    else:
        S = 200 - 2 * QF

    Q = np.floor((Q_base * S + 50) / 100)
    Q = np.clip(Q, 1, 255)
    return Q

# macierz kwantyzacji dla chrominancji
def get_quantization_matrix_chrominance(QF):
    Q_base = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float64)

    if QF < 50:
        S = 5000 / QF
    else:
        S = 200 - 2 * QF

    Q = np.floor((Q_base * S + 50) / 100)
    Q = np.clip(Q, 1, 255)
    return Q

#krok 5: kwantyzacja
def stepFive(blocks, quant_matrix):

    quantized_blocks = []
    for block in blocks:
        quantized = block / quant_matrix
        quantized_blocks.append(quantized.astype(np.int32))
    return quantized_blocks

#odwrotny krok 5: dekwantyzacja
def reverse_stepFive(blocks, quant_matrix):

    dequantized_blocks = []
    for block in blocks:
        dequantized = block * quant_matrix
        dequantized_blocks.append(dequantized)
    return dequantized_blocks

#krok 6: zaokrąglanie
def stepSix(blocks):

    rounded_blocks = []
    for block in blocks:
        rounded = np.round(block)
        rounded_blocks.append(rounded.astype(np.int32))
    return rounded_blocks


#indeksy zigzag
def get_zigzag_index():
    return [
        (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
        (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
        (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
        (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
        (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
        (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
        (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
        (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
    ]
#zygzakowanie pojedynczego bloku
def zigzag_block(block):
    zigzag_index = get_zigzag_index()
    return [block[i, j] for i, j in zigzag_index]

#krok 7: zygzakowanie każdego bloku
def stepSeven(blocks):
    result=[zigzag_block(block) for block in blocks]
    return result

#odwrotny krok 7: dezygzakowanie każdego bloku
def reverse_stepSeven(zigzag):
    zigzag_index = get_zigzag_index()
    block = np.zeros((8, 8), dtype=np.int16)
    for idx, (i, j) in enumerate(zigzag_index):
        block[i, j] = zigzag[idx]
    return block



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


#krok 8: odowanie RLE + Huffman dla bloków ZigZag
def stepEight(zigzag_blocks):
    encoded_blocks = []
    all_symbols = []

    for block in zigzag_blocks:
        rle = []
        zeros = 0

        for val in block:
            if val == 0:
                zeros += 1
            else:
                rle.append((zeros, val))
                all_symbols.append((zeros, val))
                zeros = 0
        rle.append((0, 0))  # EOB
        all_symbols.append((0, 0))
        encoded_blocks.append(rle)

    huffman_dict = build_huffman_tree(all_symbols)
    bitstream = "".join(huffman_dict[symbol] for rle in encoded_blocks for symbol in rle)
    print(f"stepEight: Huffman. Liczba bitów: {len(bitstream)}")
    return bitstream, huffman_dict

#odwrotny krok 8: dekodowanie
def reverse_stepEight(bitstream, huffman_dict, num_blocks):
    reverse_dict = {v: k for k, v in huffman_dict.items()}
    decoded_blocks = []
    current_block = []
    buffer = ""
    blocks_decoded = 0

    for bit in bitstream:
        buffer += bit
        if buffer in reverse_dict:
            symbol = reverse_dict[buffer]
            if symbol == (0, 0):  # EOB
                current_block += [0] * (64 - len(current_block))
                decoded_blocks.append(current_block)
                current_block = []
                blocks_decoded += 1
                if blocks_decoded >= num_blocks:
                    break
            else:
                zeros, val = symbol
                current_block += [0]*zeros + [val]
            buffer = ""

    print(f"decode_bitstream: Liczba zdekodowanych bloków: {len(decoded_blocks)}")
    return decoded_blocks



def process(sampling_factor, QF):
    # --- 0.Generowanie obrazu ---
    img = colors()

    # --- 1.RGB -> YCbCr ---
    ycbcr = stepOne(img)

    #rozdzielenie kanałów
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    # --- 2.Próbkowanie Cb i Cr ---
    factor = sampling_factor
    Cb_sub = stepTwo(Cb, factor)
    Cr_sub = stepTwo(Cr, factor)

    #wyrównanie
    Y_padded = padding(Y)
    Cb_padded = padding(Cb_sub)
    Cr_padded = padding(Cr_sub)

    # --- 3.Podział na bloki 8x8 ---
    Y_blocks = stepThree(Y_padded)
    Cb_blocks = stepThree(Cb_padded)
    Cr_blocks = stepThree(Cr_padded)


    # --- 4.DCT ---
    Y_blocks_dct = stepFour(Y_blocks)
    Cb_blocks_dct = stepFour(Cb_blocks)
    Cr_blocks_dct = stepFour(Cr_blocks)

    # --- 5.Kwantyzacja ---
    QY = get_quantization_matrix_luminance(QF)
    QC = get_quantization_matrix_chrominance(QF)

    Y_blocks_quant = stepFive(Y_blocks_dct, QY)
    Cb_blocks_quant = stepFive(Cb_blocks_dct, QC)
    Cr_blocks_quant = stepFive(Cr_blocks_dct, QC)

    # --- 6.Zaokrąglanie ---
    rounded_Y_blocks = stepSix(Y_blocks_quant)
    rounded_Cb_blocks = stepSix(Cb_blocks_quant)
    rounded_Cr_blocks = stepSix(Cr_blocks_quant)

    # --- 7.ZigZag ---
    Y_zz = stepSeven(rounded_Y_blocks)
    Cb_zz = stepSeven(rounded_Cb_blocks)
    Cr_zz = stepSeven(rounded_Cr_blocks)

    # --- 8. RLE + Huffman ---
    Y_encoded, Y_huffman = stepEight(Y_zz)
    Cb_encoded, Cb_huffman = stepEight(Cb_zz)
    Cr_encoded, Cr_huffman = stepEight(Cr_zz)


    # --- [Dekodowanie] ---

    # --- reverse 8.Huffman + RLE dekodowanie ---
    Y_zz_decoded = reverse_stepEight(Y_encoded, Y_huffman, len(Y_blocks))
    Cb_zz_decoded = reverse_stepEight(Cb_encoded, Cb_huffman, len(Cb_blocks))
    Cr_zz_decoded = reverse_stepEight(Cr_encoded, Cr_huffman, len(Cr_blocks))

    # --- reverse 7.ZigZag odwrotny ---
    Y_blocks_dezigzag = [reverse_stepSeven(block) for block in Y_zz_decoded]
    Cb_blocks_dezigzag = [reverse_stepSeven(block) for block in Cb_zz_decoded]
    Cr_blocks_dezigzag = [reverse_stepSeven(block) for block in Cr_zz_decoded]

    # --- reverse 6.brak ---

    # --- reverse 5. Dekwantyzacja ---
    Y_blocks_dequant = reverse_stepFive(Y_blocks_dezigzag, QY)
    Cb_blocks_dequant = reverse_stepFive(Cb_blocks_dezigzag, QC)
    Cr_blocks_dequant = reverse_stepFive(Cr_blocks_dezigzag, QC)

    # --- 4.IDCT ---
    Y_blocks_idct = reverse_stepFour(Y_blocks_dequant)
    Cb_blocks_idct = reverse_stepFour(Cb_blocks_dequant)
    Cr_blocks_idct = reverse_stepFour(Cr_blocks_dequant)

    # --- 3.Scalanie bloków ---
    width_Y = Y_padded.shape[1]
    width_C = Cb_padded.shape[1]

    Y_channel = reverse_stepThree(Y_blocks_idct, Y_padded.shape[0], Y_padded.shape[1])
    Cb_channel = reverse_stepThree(Cb_blocks_idct, Cb_padded.shape[0], Cb_padded.shape[1])
    Cr_channel = reverse_stepThree(Cr_blocks_idct, Cr_padded.shape[0], Cr_padded.shape[1])

    #Usunięcie wyrównania
    Y_channel = remove_padding(Y_channel, Y.shape[0], Y.shape[1])
    Cb_channel = remove_padding(Cb_channel, Cb_sub.shape[0], Cb_sub.shape[1])
    Cr_channel = remove_padding(Cr_channel, Cr_sub.shape[0], Cr_sub.shape[1])

    # --- 2.Nadpróbkowanie ---
    Cb_upsampled = reverse_stepTwo(Cb_channel, factor)
    Cr_upsampled = reverse_stepTwo(Cr_channel, factor)

    #złączenie kanałów
    ycbcr_reconstructed = np.stack([Y_channel, Cb_upsampled, Cr_upsampled], axis=2)

    # --- 1.Konwersja YCbCr -> RGB ---
    rgb_reconstructed = reverse_stepOne(ycbcr_reconstructed)


    return rgb_reconstructed

#funkcje pomocnicze z porpzednich laboratoriów
def save_image(array, filename):
    image = Image.fromarray(array)
    image.save(filename)

def calculate_mse(image1, image2):
    diff = image1 - image2

    mse = np.mean(np.square(diff))
    return mse

def process_mse(name1, name2):
    original_image = file_to_array(name1)
    transmitted_image = file_to_array(name2)

    mse = calculate_mse(original_image, transmitted_image)

    print(f"Mean Squared Error (MSE): {mse}")
    return mse

def file_to_array(name):
    img = Image.open(name).convert("RGB")
    img_array = np.array(img)
    return img_array

def show_image(filename):
    image = Image.open(filename)
    image.show()



