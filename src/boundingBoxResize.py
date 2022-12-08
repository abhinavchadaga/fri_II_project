from PIL import Image, ImageDraw
import copy
from math import ceil, floor

def padded_resize(image, bbox, resize = (224, 224), paddingColor = (0, 0, 0)):
    width, height = image.size
    lBorder = 0
    rBorder = 0
    tBorder = 0
    bBorder = 0
    if width < height:
        half = (height - width)/2
        lBorder = floor(half)
        rBorder = ceil(half)
    elif width > height:
        half = (width - height)/2
        tBorder = floor(half)
        bBorder = ceil(half)

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    paddedIm = add_margin(image, tBorder, rBorder, bBorder, lBorder, paddingColor)
    
    bbox = [bbox[0] + lBorder, bbox[1] + tBorder, bbox[2] + lBorder, bbox[3] + tBorder]

    rat = resize[0]/width

    paddedIm = paddedIm.resize((int(resize[0]), int(resize[1])))

    bbox = [int(rat * p) for p in bbox]

    # paddedIm.show()

    return paddedIm, bbox

if __name__ == "__main__":
    img = Image.open("./szb/IMG_0142.JPG")
    # img = img.rotate(-90, Image.NEAREST, expand = 1)
    img, bbox = padded_resize(img, [2016, 1512, 4032, 3024])
    print(bbox)