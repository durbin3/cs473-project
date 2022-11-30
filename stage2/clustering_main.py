"""
usage: clustering_main.py [-h] [-r OCR_RESULTS_PATH] [-c OCR_RESULTS_PATH] [-o OUTPUT_PATH]
optional arguments:
  -h, --help            show this help message and exit
  -r OCR_RESULTS_PATH, --ocr_results OCR_RESULTS_PATH
                        Path to the folder where ocr results are stored.
  -c OB_RESULTS_PATH, --ob_results OB_RESULTS_PATH
                        Path to the folder where object detection results are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save output
  -k NUM_CLUSTERS --num_clusters NUM_CLUSTERS
                        number of clusters
"""

import argparse
import os
import sys
from clustering_methods import method0_clustering, method1_clustering
from utils import get_rand_score

# from "cs473-project.src.od_inference" import object_detection

path_to_project = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path_to_project)

from src.od_inference import object_detection
from src.ocr_main import extract_images

parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-o",
                    "--output_path",
                    help="Path to save output.", type=str)
parser.add_argument("-p",
                    "--parameters_path",
                    help="Path to the parameters.txt file.",
                    type=str)

parser.add_argument("-r",
                    "--ocr_results",
                    help="[OPTIONAL] Path to the folder where the ocr results are stored.",
                    type=str)
parser.add_argument("-b",
                    "--ob_results",
                    help="[OPTIONAL] Path to the folder where object detection results are stored.",
                    type=str)
parser.add_argument("-k",
                    "--num_clusters",
                    help="[OPTIONAL] Number of clusters used when clustering.",
                    type=int)

SAVED_MODEL_PATH = os.path.join(".", "models", "exported_models", "centernet512_7000_no_aug", "saved_model")
BASE_LINE_CLUSTERING_FILENAME = "base_line_clusters.txt"
ADVANCED_CLUSTERING_FILENAME = "advanced_clusters.txt"

def read_parameters(file_path):
  with open(file_path, "r") as file:
    images_path = os.path.normpath(file.readline().strip())
    k = int(file.readline())

    return images_path, k

def dump_clustering(clustering, output_filepath):
  sorted_clustering = sorted(clustering.items(), key=lambda x: x[1])

  max_k = sorted_clustering[-1][1]
  output = [[] for _ in range(max_k + 1)]

  for img_num, cluster_num in sorted_clustering:
    output[cluster_num].append(img_num)

  with open(output_filepath, "w") as output_file:
    for cluster in output:
      output_file.write(", ".join(cluster) + "\n")

def main():
  args = parser.parse_args()

  ocr_output_path = args.ocr_results
  od_output_path = args.ob_results
  k = args.num_clusters

  if args.parameters_path:
    images_path, num_clusters = read_parameters(args.parameters_path)
    k = num_clusters

    # Obtain Object detection results.
    od_output_path = os.path.join(images_path, "out")
    object_detection(images_path, 512, SAVED_MODEL_PATH)

    # Obtain OCR results.
    ocr_output_path = os.path.join(od_output_path, "ocr")
    extract_images(images_path, ocr_output_path, od_output_path)

  clustering0 = method0_clustering(ocr_output_path, k)
  clustering1 = method1_clustering(ocr_output_path, od_output_path, k)

  clustering0_output_path = os.path.join(args.output_path, BASE_LINE_CLUSTERING_FILENAME)
  clustering1_output_path = os.path.join(args.output_path, ADVANCED_CLUSTERING_FILENAME)

  print(clustering0)
  dump_clustering(clustering0, clustering0_output_path)
  
  print(clustering1)
  dump_clustering(clustering1, clustering1_output_path)

  true_clustering_path = os.path.join("..", "dataset1_K_4", "dataset1_K_4_clustering.txt")
  print(get_rand_score(clustering0_output_path, clustering1_output_path))

if __name__ == "__main__":
  main()