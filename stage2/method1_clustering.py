import argparse
import ast
import glob
import pandas as pd
import os

from baseClustering import base_clustering, doc_vectorize, concat_texts

parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-r",
                    "--ocr_results",
                    help="Path to the folder where the ocr results are stored.",
                    type=str)
parser.add_argument("-b",
                    "--ob_results",
                    help="Path to the folder where the object detection results are stored.",
                    type=str)
parser.add_argument("-k",
                    "--num_clusters",
                    help="number of clusters", type=int)
# parser.add_argument("-o",
#                     "--output_path",
#                     help="Path to save output.", type=str)

# Added "ERD_LABEL_" to avoid key collision with words in ocr results
ERD_LABELS = ["ERD_LABEL_entity", "ERD_LABEL_weak_entity",
              "ERD_LABEL_rel", "ERD_LABEL_ident_rel",
              "ERD_LABEL_rel_attr", "ERD_LABEL_many", "ERD_LABEL_one"]

#######################################################################################
# Reads in the object detection output and returns the counters of each entity,       #
# weak_entity, relationship, etc...                                                   #
#                                                                                     #
# Input:                                                                              #
#       ob_path = path to ocr result text files                                       #
#                                                                                     #
# Output:                                                                             #
#       type pandas.Dataframe: columns = ERD_LABELS                                   #
#######################################################################################

def parse_ob_output(ob_path):
  df_erd_counts = pd.DataFrame([], columns=ERD_LABELS)

  for txt_file in glob.glob(ob_path + "/*.txt"):
    file_num = os.path.basename(txt_file)[:-4]
    label_counts = {label: 0 for label in ERD_LABELS}

    with open(txt_file,"r") as f:
      components = ast.literal_eval("".join(f.readlines()))
      for component in components:
        label_counts["ERD_LABEL_" + component[0]] += 1

    df_erd_counts.loc[file_num] = label_counts

  return df_erd_counts

def main():
  args = parser.parse_args()


  text_dict = concat_texts(args.ocr_results)
  df_ocr_features = doc_vectorize(text_dict)
  df_erd_features = parse_ob_output(args.ob_results)
  
  df_features = df_ocr_features.join(df_erd_features, on=df_ocr_features.index, how='left')
  # df_features = df_ocr_features

  cluster_dict = base_clustering(df_features, args.num_clusters)
  print(cluster_dict)

if __name__ == "__main__":
  main()