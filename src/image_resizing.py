import os
import sys
from PIL import Image
import xml.etree.ElementTree as ET
import shutil

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
    directory = os.path.join('dataset', 'images', 'raw_images')
    out_dir = os.path.join('dataset', 'images', 'resized_images')

    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for file in os.listdir(directory):
        if file.endswith(('jpeg', 'png', 'jpg')):
            outfile = out_dir + '/' + file
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
        result.paste(image)
        return result
    else:
        result = Image.new(image.mode, (height, height), (255,255,255))
        result.paste(image)
        return result

def resize_all(size):
    """
    Takes all images in resized_images, along with the corresponding xml labels
    and resizes them to be the given size.

    inputs:
        - size (int,int)
    """
    label_dir = os.path.join('dataset', 'raw_labels')
    raw_dir = os.path.join('dataset', 'images', 'raw_images')
    
    in_files = [file for file in os.listdir(raw_dir) if file.endswith(('jpeg', 'png', 'jpg','PNG','JPG','JPEG'))]
    in_labels = [file for file in os.listdir(label_dir)]

    # Resize all of the resized images
    print("Resizing images to be " , size)
    process_raw(size)
    print("Images Resized")

    # Resize all of the annotations coordinates to match
    new_label_dir = 'dataset/labels'
    if os.path.exists(new_label_dir): shutil.rmtree(new_label_dir)
    shutil.copytree(label_dir, new_label_dir)
    print("Resizing all labels")
    width,height = size
    for img_path,label_path in zip(in_files,in_labels):
        assert(img_path[0:3] == label_path[0:3])
        with Image.open(raw_dir + '/' + img_path) as img:
            side_length = max(img.size[0],img.size[1])
            ratio = width/side_length
        path = new_label_dir + '/' + label_path
        mytree = ET.parse(path)
        root = mytree.getroot()
        root.find('size').find('width').text = str(width)
        root.find('size').find('height').text = str(height)
        for tag in root.iter('xmin'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('xmax'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('ymin'): tag.text = str(round(int(tag.text) * ratio))
        for tag in root.iter('ymax'): tag.text = str(round(int(tag.text) * ratio))
        mytree.write(new_label_dir + '/' + label_path)
    print("Labels Resized")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        process_raw((1024,1024))
        exit()

    operation = sys.argv[1]
    size = int(sys.argv[2])
    if operation == 'process': process_raw((size,size))
    elif operation == 'resize': resize_all((size,size))
    else: print("Arguments should include either 'process' or 'resize' along with a given side length\tE.g. python image_resizing.py process 1024")
