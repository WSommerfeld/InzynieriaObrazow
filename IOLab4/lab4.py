import numpy as np
from PIL import Image

#zad1
def find_closest_palette_color(value):
    return round(value / 255) * 255

def floyd_steinberg_dithering(image):

    pixels = np.array(image, dtype=float)

    height, width = pixels.shape

    #kod ze slajdu 5
    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = find_closest_palette_color(old_pixel)
            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height and x > 0:
                pixels[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                pixels[y + 1, x] += quant_error * 5 / 16
            if y + 1 < height and x + 1 < width:
                pixels[y + 1, x + 1] += quant_error * 1 / 16

    #

    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

#redukcja palety bez ditheringu
def reduce_palette(image):

    pixels = np.array(image, dtype=np.uint8)

    reduced_pixels = (np.round(pixels / 255) * 255).astype(np.uint8)

    return Image.fromarray(reduced_pixels)


#zad 2
#jak w 1 z parametrem k
def find_closest_palette_color_k(value,k):
    return round((k - 1) * value / 255) * 255 / (k - 1)


#redukcja palety bez ditheringu
def reduce_colors(image, k=2):

    pixels = np.array(image, dtype=float)
    height, width, channels = pixels.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                pixels[y, x, c] = find_closest_palette_color_k(pixels[y, x, c], k)

    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))

#floyd-steinberg dla koloru
def floyd_steinberg_dithering_color(image, k=2):

    pixels = np.array(image, dtype=float)
    height, width, channels = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x].copy()
            new_pixel = np.zeros(3)
            for c in range(channels):
                new_pixel[c] = find_closest_palette_color_k(old_pixel[c], k)
            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += quant_error * (7 / 16)
            if y + 1 < height and x > 0:
                pixels[y + 1, x - 1] += quant_error * (3 / 16)
            if y + 1 < height:
                pixels[y + 1, x] += quant_error * (5 / 16)
            if y + 1 < height and x + 1 < width:
                pixels[y + 1, x + 1] += quant_error * (1 / 16)

    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


#zad3
#~datko.pl rysowanie punktu
def draw_point(image, x, y, color=(255, 255, 255)):
    image[image.shape[0] - 1 - y, x, :] = color


#funkcja krawedziowa (iloczyn wektorowy)
def edge_function(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

#sprawdzenie czy punkt leży w trójkącie
def is_point_in_triangle(p, a, b, c):

    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c


    w1 = edge_function(ax, ay, bx, by, px, py)
    w2 = edge_function(bx, by, cx, cy, px, py)
    w3 = edge_function(cx, cy, ax, ay, px, py)

    has_neg = (w1 < 0) or (w2 < 0) or (w3 < 0)
    has_pos = (w1 > 0) or (w2 > 0) or (w3 > 0)


    return not (has_neg and has_pos)

#"naiwne" rysowanie linii
def draw_line_rounded(image, x0, y0, x1, y1, color):
    if x0!=x1:
        a=(y1-y0)/(x1-x0)
        b=y1-a*x1
        dx=abs(x1-x0)

        for x in range(dx):
            y=a*(x+x0)+b
            draw_point(image, int(x+x0), int(round(y)), color)

    else:
        dy=abs(y1-y0)
        if y1>y0:
            for y in range(dy):
                draw_point(image, x0, y+y0, color)
        else:
            for y in range(dy):
                draw_point(image, x0, y+y1, color)





#rysowanie lini algorytmem bresenhama
def draw_line(image, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    D = dx - dy

    while True:
        draw_point(image, x0, y0, color)

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * D
        if e2 > -dy:
            D -= dy
            x0 += sx
        if e2 < dx:
            D += dx
            y0 += sy

#rysowanie wypelnionego trójkąta
def draw_filled_triangle(image, a, b, c, color):
    xmin = min(a[0], b[0], c[0])
    xmax = max(a[0], b[0], c[0])
    ymin = min(a[1], b[1], c[1])
    ymax = max(a[1], b[1], c[1])

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if is_point_in_triangle((x, y), a, b, c):
                draw_point(image, x, y, color)


#zad 4
#interpolacja między dwoma kolorami
def interpolate_color(color_a, color_b, t):
    return tuple([
        int(color_a[i] + t * (color_b[i] - color_a[i]))
        for i in range(3)
    ])

#rysowanie kolorowej linii algorytmem bresenhama z interpolacją koloru
def draw_line_with_color(image, x0, y0, color0, x1, y1, color1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    steps = max(dx, dy)
    step = 0

    while True:
        if steps != 0:
            t = step / steps
        else:
            t = 0
        color = interpolate_color(color0, color1, t)
        draw_point(image, x0, y0, color)

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        step += 1


#wspolczynniki barycentryczne
def barycentric_weights(p, a, b, c):
    det = (b[1] - c[1])*(a[0] - c[0]) + (c[0] - b[0])*(a[1] - c[1])
    if det == 0:
        return 0, 0, 0
    w1 = ((b[1] - c[1])*(p[0] - c[0]) + (c[0] - b[0])*(p[1] - c[1])) / det
    w2 = ((c[1] - a[1])*(p[0] - c[0]) + (a[0] - c[0])*(p[1] - c[1])) / det
    w3 = 1 - w1 - w2
    return w1, w2, w3

#kolorowy trójkąt z intrpolacją kolorów
def draw_filled_triangle_with_color(image, a, color_a, b, color_b, c, color_c):
    xmin = min(a[0], b[0], c[0])
    xmax = max(a[0], b[0], c[0])
    ymin = min(a[1], b[1], c[1])
    ymax = max(a[1], b[1], c[1])

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            w1, w2, w3 = barycentric_weights((x, y), a, b, c)
            if (w1 >= 0) and (w2 >= 0) and (w3 >= 0):
                r = int(w1 * color_a[0] + w2 * color_b[0] + w3 * color_c[0])
                g = int(w1 * color_a[1] + w2 * color_b[1] + w3 * color_c[1])
                b_col = int(w1 * color_a[2] + w2 * color_b[2] + w3 * color_c[2])
                draw_point(image, x, y, (r, g, b_col))



#zad 5
#przeskalowanie obrazu w dół
def downscale_image(image, scale):
    small_height = image.shape[0] // scale
    small_width = image.shape[1] // scale
    small_image = np.zeros((small_height, small_width, 3), dtype=np.uint8)

    for y in range(small_height):
        for x in range(small_width):
            block = image[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale, :]
            avg_color = block.mean(axis=(0, 1))
            small_image[y, x, :] = avg_color

    return small_image


#f. pomocnicze z poprzednich zajęć
def show_image(filename):
    image = Image.open(filename)
    image.show()

#przetwarzanie
def process_1(name):

    input_image = Image.open(name).convert('L')
    output_image = floyd_steinberg_dithering(input_image)


    output_image.save(name+'output_dithered.png')
    show_image(name+'output_dithered.png')

def process_1a(name):
    input_image = Image.open(name).convert('L')
    output_image = reduce_palette(input_image)
    output_image.save(name+'output_palette.png')
    show_image(name + 'output_palette.png')



def process_2(name):

    input_image = Image.open(name)

    reduced_image = reduce_palette(input_image)
    output_image = floyd_steinberg_dithering_color(input_image,2)

    reduced_image.save(name+'reduced_palette_color.png')
    output_image.save(name+'output_dithered_color.png')

    show_image(name+'reduced_palette_color.png')
    show_image(name+'output_dithered_color.png')



def process_3():

    width, height = 200, 200
    #czarne tło
    image = np.zeros((height, width, 3), dtype=np.uint8)
    #rysowanie linii
    draw_line(image, 5, 5, 50, 50, (255, 0, 0))
    draw_line(image, 50, 50, 50, 100, (0, 255, 0))

    #rysowanie trójkąta
    a = (60, 30)
    b = (160, 80)
    c = (100, 160)
    draw_filled_triangle(image, a, b, c, (0, 0, 255))

    Image.fromarray(image).save("lines_and_triangle.png")
    Image.fromarray(image).show()

def process_3a():
    width, height = 70, 70
    image = np.zeros((height, width, 3), dtype=np.uint8)

    draw_line_rounded(image, 25, 0, 50, 50, (255, 0, 0))
    draw_line(image, 25, 10, 50, 60, (255, 0, 0))

    Image.fromarray(image).save("lines.png")
    Image.fromarray(image).show()

def process_3b():
    width, height = 60, 60
    image = np.zeros((height, width, 3), dtype=np.uint8)

    draw_line(image, 25, 0, 50, 50, (255, 0, 0))

    Image.fromarray(image).show()

def process_4():
    width, height = 300, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    #kolorowa linia
    draw_line_with_color(image, 50, 5, (255, 0, 0), 250, 100, (0, 0, 255))

    #kolorowy trójkąt
    a = (50, 110)
    color_a = (255, 0, 0)
    b = (150, 290)
    color_b = (0, 255, 0)
    c = (250, 110)
    color_c = (0, 0, 255)
    draw_filled_triangle_with_color(image, a, color_a, b, color_b, c, color_c)

    Image.fromarray(image).save("interpolated_lines_and_triangle.png")
    Image.fromarray(image).show()

def process_5():
    scale = 4
    width, height = 300, 300
    big_image = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

    # kolorowa linia
    draw_line_with_color(big_image, 50* scale, 5* scale, (255, 0, 0),
                         250* scale, 100* scale, (0, 0, 255))


    # kolorowy trójkąt
    a = (50* scale, 110* scale)
    color_a = (255, 0, 0)
    b = (150* scale, 290* scale)
    color_b = (0, 255, 0)
    c = (250* scale, 110* scale)
    color_c = (0, 0, 255)
    draw_filled_triangle_with_color(big_image, a, color_a, b, color_b, c, color_c)

    #przeskalowanie
    final_image = downscale_image(big_image, scale)

    Image.fromarray(final_image).save("SSAA_interpolated_lines_and_triangle.png")
    Image.fromarray(final_image).show()


def process_5a():
    scale = 4
    width, height = 300, 300
    big_image = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

    # kolorowa linia
    draw_line_with_color(big_image, 50* scale, 4* scale, (255, 0, 0),
                         250* scale, 99* scale, (0, 0, 255))
    draw_line_with_color(big_image, 50* scale, 5* scale, (255, 0, 0),
                         250* scale, 100* scale, (0, 0, 255))
    draw_line_with_color(big_image, 50* scale, 6* scale, (255, 0, 0),
                         250* scale, 101* scale, (0, 0, 255))

    # kolorowy trójkąt
    a = (50* scale, 110* scale)
    color_a = (255, 0, 0)
    b = (150* scale, 290* scale)
    color_b = (0, 255, 0)
    c = (250* scale, 110* scale)
    color_c = (0, 0, 255)
    draw_filled_triangle_with_color(big_image, a, color_a, b, color_b, c, color_c)

    #przeskalowanie
    final_image = downscale_image(big_image, scale)

    Image.fromarray(final_image).save("wide_SSAA_interpolated_lines_and_triangle.png")
    Image.fromarray(final_image).show()
