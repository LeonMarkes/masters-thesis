from typing import Tuple, List
from PIL import Image, ImageFilter
import numpy as np


# im = Image.open('test.png')
# print(im.format, im.size, im.mode)
#
# box: Tuple[int] = (100, 100, 400, 400)
#
# region = im.crop(box)
# region = region.transpose(Image.ROTATE_180)
# im.paste(region, box)
#
# r, g, b, a = im.split()
# im = Image.merge('RGB', (b, r, g))
#

# blur = np.ones((3, 3))
def provjeri_najvecu_vrijednost(vrijednosti: Tuple[float]) -> float:
    return max(vrijednosti)


# with Image.open('test.png') as im:
#     im = np.asarray(im.convert('L'), dtype=float)
#     # pooled_im = np.array()
#     pooled_image: List[float] = []
#     for i in range(1, im.shape[0] - 1, 2):
#         pooled_pixels: List[float] = []
#         for j in range(1, im.shape[1] - 1, 2):
#             pixels: Tuple[float] = (im[i][j], im[i][j + 1], im[i + 1][j], im[i + 1][j + 1])
#             pooled_pixel: float = check_highest_value(pixels)
#             pooled_pixels.append(pooled_pixel)
#         pooled_image.append(pooled_pixels)
#
#     new_image = Image.fromarray(np.asarray(pooled_image))
#     new_im = Image.fromarray(im)
#     new_im.show()
#     new_image.show()


def udruzi_sliku(naziv_slike: str) -> Image:
    with Image.open(naziv_slike) as slika:
        slika = np.asarray(slika.convert('L'), dtype=float)
        udruzena_slika: List[float] = []
        for i in range(1, slika.shape[0] - 1, 2):
            udruzeni_pikseli: List[float] = []
            for j in range(1, slika.shape[1] - 1, 2):
                pikseli: Tuple[float] = (slika[i][j], slika[i][j + 1], slika[i + 1][j], slika[i + 1][j + 1])
                udruzen_piksel: float = provjeri_najvecu_vrijednost(pikseli)
                udruzeni_pikseli.append(udruzen_piksel)
            udruzena_slika.append(udruzeni_pikseli)

        nova_slika = Image.fromarray(np.asarray(udruzena_slika))
        return nova_slika


udruzi_sliku('test.png').show()