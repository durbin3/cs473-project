
"""
usage: ocr_main.py [-h] [-r OBJECT_DETECT_RESULTS] [-o OUTPUT_PATH] [-i IMAGE_DIR]
optional arguments:
  -h, --help            show this help message and exit
  -r OBJECT_DETECT_RESULTS, --od_results OBJECT_DETECT_RESULTS
                        Path to the folder where object detection results are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save output
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored.
"""
import ast
import os
import glob
import argparse
import shutil
import easyocr
import numpy as np

from PIL import Image

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-r",
                    "--od_results",
                    help="Path to the folder where the object detection results are stored.",
                    type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path to save output.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the raw image files are stored. ",
                    type=str, default=None)

args = parser.parse_args()

def extract_images(img_path, out_path, od_path):
    reader = easyocr.Reader(["en"])
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    for txt_file in glob.glob(od_path+"/*.txt"):
        img_file = os.path.basename(txt_file)
        img_file_num = img_file[:-4]
        print("OCR For Image ", img_file_num)
        tmp_img_path = img_path+"/"+img_file_num+".png"
        if not os.path.exists(tmp_img_path): tmp_img_path = img_path+"/"+img_file_num+".jpg"
        if not os.path.exists(tmp_img_path): tmp_img_path = img_path+"/"+img_file_num+".jpeg"
        save_path = out_path + "/" + img_file_num + ".txt"


        text_f = open(save_path,"w+")

        with open(txt_file,"r") as f:
            res_list = ast.literal_eval(f.readline())

        img = Image.open(tmp_img_path)

        for idx in range(len(res_list)):
            obj_type, coords = res_list[idx]
            if obj_type == "many" or obj_type == "one":
                continue
            xmin,xmax = min(coords[1],coords[3]), max(coords[1],coords[3])
            ymin,ymax = min(coords[0],coords[2]), max(coords[0],coords[2])
            img2 = img.crop((xmin, ymin, xmax, ymax))
            result = reader.readtext(np.array(img2),paragraph="False")
            ans_arr = []
            for idx2 in result:
              ans_arr.append(idx2[1])

            write_str = "{}_{}: {} {}".format(img_file_num, idx, obj_type, list(map(str, ans_arr)))
            print(write_str)
            text_f.write(write_str+'\n')
        print()
        text_f.close()

def main():
    extract_images(args.image_dir, args.output_path, args.od_results)
    print("Successfully processed and extracted Images!!")


if __name__ == '__main__':
    main()