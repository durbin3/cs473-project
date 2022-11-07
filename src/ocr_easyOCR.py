
"""
usage: ocr_easyOCR [-h] [-o OUTPUT_PATH] [-i IMAGE_DIR]
optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output directory
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored.
"""

import os
import argparse
import cv2
import easyocr

from PIL import Image


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="OCR")
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output directory.", type=str)
parser.add_argument("-i",
                    "--processed_image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str, default=None)

args = parser.parse_args()


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_img = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist_img = (dist_img * 255).astype("uint8")
    dist_img = cv2.threshold(dist_img, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(dist_img, cv2.MORPH_OPEN, kernel)





    #cv2.imshow("window",dist_img)
    #cv2.waitKey(0)

def ocr(reader, img_path):
    return reader.readtext(img_path,paragraph="False")



def image_to_string(save_path, image_path):
    reader = easyocr.Reader(["en"])
    if not os.path.exists(save_path):
      os.mkdir(save_path)

    for subdir in os.listdir(image_path):
        with open(image_path+"/"+subdir+"/"+"types.txt","r") as f:
            save_file = save_path+"/"+subdir+".txt"
            if os.path.exists(save_file):
              os.remove(save_file)
            text_file = open(save_file, "w+")
            for line in f:
                filenum, object_type = line.split()
                result = ocr(reader, image_path+"/"+subdir+"/"+filenum+".png")
                ans_arr = []
                for idx in result:
                  ans_arr.append(idx[1])

                print(f"{subdir}_{filenum}: {object_type} {ans_arr}")
                write_str = subdir + "_" + filenum + ":" + " " + object_type + " " + str(ans_arr) + "\n"
                text_file.write(write_str)
            text_file.close()



def main():
    image_to_string(args.output_path, args.processed_image_dir)

    print("Successfully Extracted Texts!")

if __name__ == '__main__':
    main()