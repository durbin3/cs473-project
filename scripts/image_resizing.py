import os
import sys
from PIL import Image


## Running Directions:
# python image_resizing.py process [size] to convert all of the raw images to
# be [size] X [size]
#
# python image_resizing.py resize [size] to convert all existing resized images
# and their corresponding xml labels to be the new [size] X [size]

def process_raw(size):
    """
    Resizes all of the images in the raw images folder to be the given size

    inputs:
        - size (int,int)
    """
    directory = 'dataset/images/raw_images'
    out_dir = 'dataset/images/resized_images'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    for file in os.listdir(directory):
        if file.endswith(('jpeg', 'png', 'jpg')):
            outfile = os.path.join(out_dir, file)
            with Image.open(directory+'/'+file) as im:
                square = expand2square(im)
                out = square.resize(size,resample=Image.Resampling.LANCZOS)
                out.save(outfile,quality=100)

def expand2square(image):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), (255,255,255))
        result.paste(image, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(image.mode, (height, height), (255,255,255))
        result.paste(image, ((height - width) // 2, 0))
        return result

def resize_all(size):
    """
    Takes all images in resized_images, along with the corresponding xml labels
    and resizes them to be the given size.

    inputs:
        - size (int,int)
    """
    resized_dir = 'dataset/images/resized_images'
    label_dir = 'dataset/labels'
    if not os.path.exists(resized_dir): return
    in_files = [file for file in os.listdir(resized_dir) if file.endswith(('jpeg', 'png', 'jpg'))]
    in_labels = [file for file in os.listdir(label_dir)]

    # Resize all of the resized images
    print("Resizing images to be " , size)
    old_size = None
    for path in in_files:
        with Image.open(resized_dir+'/'+path) as img:
            if old_size is None: old_size = img.size
            out = img.resize(size,resample=Image.Resampling.LANCZOS)
            out.save(resized_dir+'/'+path,quality=100)
    print("Images Resized")
    # Resize all of the annotations coordinates to match
    if old_size is None: return
    old_width,old_height = old_size
    width,height = size
    ratio = width/old_width
    import xml.etree.ElementTree as ET
    print("Resizing all labels")
    for file in in_labels:
        path = label_dir + '/' + file
        mytree = ET.parse(path)
        root = mytree.getroot()
        root.find('size').find('width').text = str(width)
        root.find('size').find('height').text = str(height)
        for tag in root.iter('xmin'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('xmax'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('ymin'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('ymax'): tag.text = str(round(int(tag.text) * ratio))
        mytree.write(label_dir + '/' + file)
    print("Labels Resized")

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv != 3):
        process_raw((1024,1024))
        exit()

    operation = sys.argv[1]
    size = int(sys.argv[2])
    if operation == 'process': process_raw((size,size))
    elif operation == 'resize': resize_all((size,size))
    else: print("Arguments should include either 'process' or 'resize' along with a given side length\tE.g. python image_resizing.py process 1024")
