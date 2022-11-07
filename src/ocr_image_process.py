# dividing od output images to coordinates
# outputs directory: image # with object type images and text file with all the types


"""
usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]
optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output directory
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored.
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


from PIL import Image

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-x",
                    "--xml_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output directory.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str, default=None)


args = parser.parse_args()

def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (filename,
                     width,
                     height,
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text),
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def label_to_pic(img_path, output_path, xml_df):
    grouped = xml_df.groupby("filename")

    for fn in grouped.groups:
        file_count = 0
        tmp_path = img_path + "/" + fn
        save_path = output_path + "/" + fn[:-4]
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        file = save_path+"/"+"types.txt"
        if os.path.exists(file):
            os.remove(file)

        f = open(file,"w+")
        img = Image.open(tmp_path)
        filename_df = grouped.get_group(fn)
        for index, row in filename_df.iterrows():
            tmp_save_path = save_path+"/"+str(file_count)+".png"
            x1, x2 = row["xmin"], row["xmax"]
            y1, y2 = row["ymin"], row["ymax"]

            img2 = img.crop((x1,y1,x2,y2))

            img2.save(tmp_save_path)

            write_str = str(file_count) + " " + row["class"] + "\n"
            f.write(write_str)

            file_count+=1

        f.close()


def main():
    examples = xml_to_csv(args.xml_dir)
    label_to_pic(args.image_dir, args.output_path, examples)
    print("Successfully processed Images!!")

if __name__ == '__main__':
    main()