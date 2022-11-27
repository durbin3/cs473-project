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
from clustering_methods import method0_clustering, method1_clustering

parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-r",
                    "--ocr_results",
                    help="Path to the folder where the ocr results are stored.",
                    type=str)
parser.add_argument("-b",
                    "--ob_results",
                    help="Path to the folder where object detection results are stored.",
                    type=str)
parser.add_argument("-k",
                    "--num_clusters",
                    help="number of clusters", type=int)
parser.add_argument("-o",
                    "--output_path",
                    help="Path to save output.", type=str)

def dump_clustering(clustering, output_dir, filename):
  sorted_clustering = sorted(clustering.items(), key=lambda x: x[1])

  max_k = sorted_clustering[-1][1]
  output = [[] for _ in range(max_k + 1)]

  for img_num, cluster_num in sorted_clustering:
    output[cluster_num].append(img_num)

  with open(os.path.join(output_dir, filename), "w") as output_file:
    for cluster in output:
      output_file.write(", ".join(cluster) + "\n")

def main():
  args = parser.parse_args()
  
  clustering0 = method0_clustering(args.ocr_results, args.num_clusters)
  clustering1 = method1_clustering(args.ocr_results, args.ob_results, args.num_clusters)

  print(clustering0)
  dump_clustering(clustering0, args.output_path, "base_line_clusters.txt")

  print(clustering1)
  dump_clustering(clustering1, args.output_path, "advanced_clusters.txt")


if __name__ == "__main__":
  main()