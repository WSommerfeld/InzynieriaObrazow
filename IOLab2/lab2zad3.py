from PIL import Image
import numpy as np
import zlib
import struct



def colors():

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



def save_png(filename, array):
    width, height = array.shape[1], array.shape[0]
    raw_data = b""

    for y in range(height):
        raw_data += b"\x00" + array[y].tobytes()  # Bajt filtru + dane wiersza

    compressed =  zlib.compress(raw_data)

    signature = b'\x89PNG\r\n\x1a\n'

    # chunk IHDR
    ihdr_chunk_type = b'IHDR'
    bit_depth = 8
    color_type = 2
    compression_method = 0
    filter_method = 0
    interlace_method = 0

    # dane IHDR,
    ihdr_data = struct.pack(">2I5B", width, height, bit_depth, color_type, compression_method, filter_method,
                            interlace_method)

    # Chunk IHDR
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + ihdr_chunk_type + ihdr_data + struct.pack(">I", zlib.crc32(
        ihdr_chunk_type + ihdr_data) & 0xFFFFFFFF)

    # Chunk IDAT (dane obrazu)
    idat_chunk_type = b'IDAT'


    # Chunk IDAT
    idat_chunk = struct.pack(">I", len(compressed)) + idat_chunk_type + compressed + struct.pack(">I",zlib.crc32(idat_chunk_type + compressed) & 0xFFFFFFFF)

    # Chunk IEND (zakończenie)
    iend_chunk_type = b'IEND'
    iend_chunk = struct.pack(">I", 0) + iend_chunk_type + struct.pack(">I", zlib.crc32(iend_chunk_type) & 0xFFFFFFFF)

    # Zapis pliku PNG
    with open(filename, "wb") as f:
        f.write(signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)

    print(f"Obraz zapisany jako {filename}")




def show_image(filename):
    image = Image.open(filename)
    image.show()



array = colors()

width, height = array.shape[1], array.shape[0]

save_png("rainbow.png", array)

show_image('rainbow.png')


