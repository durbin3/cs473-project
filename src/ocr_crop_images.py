# dividing od output images to coordinates
# outputs directory: image # with object type images and text file with all the types


"""
usage: ocr_crop_images.py [-h] [-r OBJECT_DETECT_RESULTS] [-o OUTPUT_PATH] [-i IMAGE_DIR]
optional arguments:
  -h, --help            show this help message and exit
  -r OBJECT_DETECT_RESULTS, --od_results OBJECT_DETECT_RESULTS
                        Path to the folder where object detection results are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output directory
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored.
"""
import ast
import os
import glob
import argparse
import shutil

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
                    help="Path of output directory.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. ",
                    type=str, default=None)

args = parser.parse_args()

def ocr_resize(size, img):
    side_length = max(img.size[0],img.size[1])
    ratio = side_length/size
    return ratio


def segment_images(img_path, out_path, od_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    for txt_file in glob.glob(od_path+"/*.txt"):
        img_file = os.path.basename(txt_file)
        img_file_num = img_file[:-4]
        tmp_img_path = img_path+"/"+img_file_num+".png"
        img_file_path = out_path + "/" + str(img_file_num)
        img_type_txt = img_file_path + "/types.txt"


        if os.path.exists(img_file_path):
            shutil.rmtree(img_file_path)
        os.mkdir(img_file_path)

        if os.path.exists(img_type_txt):
            os.remove(img_type_txt)
        text_f = open(img_type_txt,"w+")

        with open(txt_file,"r") as f:
            res_list = ast.literal_eval(f.readline())

        img = Image.open(tmp_img_path)
        ratio = ocr_resize(512, img)

        for idx in range(len(res_list)):
            obj_type, coords = res_list[idx]
            if obj_type == "many" or obj_type == "one":
                continue
            img_output_file = img_file_path + "/" + str(idx) + ".png"
            xmin,xmax = min(coords[1],coords[3])*ratio, max(coords[1],coords[3])*ratio
            ymin,ymax = min(coords[0],coords[2])*ratio, max(coords[0],coords[2])*ratio
            img2 = img.crop((xmin, ymin, xmax, ymax))

            img2.save(img_output_file)

            write_str = str(idx) + " " + obj_type + "\n"
            text_f.write(write_str)
        text_f.close()

def main():
    segment_images(args.image_dir, args.output_path, args.od_results)
    print("Successfully processed Images!!")


if __name__ == '__main__':
    main()