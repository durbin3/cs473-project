import argparse
import ast
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import silhouette_score

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

# Added "ERD_LABEL_" to avoid key collision with words in ocr results.
ERD_LABELS = ["ERD_LABEL_entity", "ERD_LABEL_weak_entity",
              "ERD_LABEL_rel", "ERD_LABEL_ident_rel",
              "ERD_LABEL_rel_attr", "ERD_LABEL_many", "ERD_LABEL_one"]

# Reads in the object detection output and returns the counters of each entity,
# weak_entity, relationship, etc...
# Returns a pandas.DataFrame of the ERD_LABEL_label counters.
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

# Plots line graph of different k values against their within-cluster sum square error.
def graph_distortion_WC_SSE(features, min_k=1, max_k=5):
  k_nums = [i for i in range(min_k, max_k)]
  within_cluster_SSEs = []  # Within-cluster sum squared errors

  for k in k_nums:
    k_means = KMeans(n_clusters=k, init="k-means++").fit(features)

    clustering = k_means.predict(features)
    centroids = k_means.cluster_centers_

    sse = 0
    for i in range(features.shape[0]):
      center = centroids[clustering[i]]
      sse += sum([(features.iloc[i][j] - center[j]) ** 2 for j in range(features.shape[1])])

    within_cluster_SSEs.append(sse)

  plt.plot(k_nums, within_cluster_SSEs)
  plt.xlabel("k")
  plt.ylabel("Within-cluster-SSE")
  plt.show()

def graph_distortion_silhouette_scores(features):
  min_k = 2
  max_k = features.shape[0]

  k_nums = [i for i in range(min_k, max_k)]
  sil_scores = []

  for k in k_nums:
    k_means = KMeans(n_clusters=k, init="k-means++").fit(features)
    sil_scores.append(silhouette_score(features, k_means.labels_, metric='euclidean'))

  plt.plot(k_nums, sil_scores)
  plt.xlabel("k")
  plt.ylabel("silhouette_score")
  plt.show()
  

def main():
  args = parser.parse_args()

  text_dict = concat_texts(args.ocr_results)
  df_ocr_features = doc_vectorize(text_dict)

  df_erd_features = parse_ob_output(args.ob_results)
  
  df_features = df_ocr_features.join(df_erd_features, on=df_ocr_features.index, how='left')
  df_features = pd.DataFrame(normalize(df_features), index= df_features.index,columns=df_features.columns)

  # print(df_features)
  # graph_distortion_WC_SSE(df_features, 1, 5)
  graph_distortion_silhouette_scores(df_features)

if __name__ == "__main__":
  main()