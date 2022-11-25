
"""
usage: baseClustering.py [-h] [-r OCR_RESULTS_PATH] [-o OUTPUT_PATH]
optional arguments:
  -h, --help            show this help message and exit
  -r OCR_RESULTS_PATH, --ocr_results OCR_RESULTS_PATH
                        Path to the folder where ocr results are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save output
  -k NUM_CLUSTERS --num_clusters NUM_CLUSTERS
                        number of clusters
"""
import ast
import math
import os
import glob
import argparse
import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Image processing for OCR")
parser.add_argument("-r",
                    "--ocr_results",
                    help="Path to the folder where the ocr results are stored.",
                    type=str)
parser.add_argument("-k",
                    "--num_clusters",
                    help="number of clusters", type=int)
parser.add_argument("-o",
                    "--output_path",
                    help="Path to save output.", type=str)


#######################################################################################
# Input:                                                                              #
#       ocr_path = path to ocr result text files                                      #
#                                                                                     #
# Output:                                                                             #
#       type Dict: key = erd_image_num , value = concatenated string                  #
#######################################################################################

def concat_texts(ocr_path):
    ps = PorterStemmer()
    erd_dict = {}
    for txt_file in glob.glob(ocr_path+"/*.txt"):
        file_num = os.path.basename(txt_file)[:-4]
        with open(txt_file,"r") as f:
            erd_dict[file_num] = []
            for line in f:
                erd_dict[file_num].extend(ast.literal_eval(re.search("\[.*\]",line).group()))

    return dict(dict(sorted(erd_dict.items())))


#######################################################################################
# Input:                                                                              #
#       erd_dict = type Dict:  key = erd_image_num , value = list of words in the img #
#                                                                                     #
# Output:                                                                             #
#       ans_df = type DataFrame: DataFrame of processed texts                         #
#######################################################################################
def doc_vectorize(erd_dict):
    ps = PorterStemmer()
    erd_list = list(erd_dict.values())
    tmp_erd_string = [" ".join(doc) for doc in erd_list]

    ###########stemming###########
    for doc_idx in range(len(tmp_erd_string)):
        tmp_string = [ps.stem(i) for i in tmp_erd_string[doc_idx].split()]
        tmp_erd_string[doc_idx] = " ".join(tmp_string)

    vectorizer = CountVectorizer(dtype=float)
    vectorizer.fit(tmp_erd_string)

    # Unique words along with their indices
    # voc_index = vectorizer.vocabulary_

    # Encode the Document
    vector = vectorizer.transform(tmp_erd_string)
    matrix = vector.toarray()

    ### term frequency ######
    for docidx in range(len(matrix)):
        for inneridx in range(len(matrix[docidx])):
            # if tf = 0 , ans is 0. if tf = 1, logTF(1)+1 = 1 so leave it as it is.
            if matrix[docidx][inneridx] == 0 or matrix[docidx][inneridx] == 1:
                continue
            matrix[docidx][inneridx] = math.log10(matrix[docidx][inneridx])+1


    df = pd.DataFrame(matrix, columns=vectorizer.get_feature_names_out())
    ans_df = df.set_index(pd.Index(erd_dict.keys()))

    return ans_df


#######################################################################################
# Input:                                                                              #
#       dataframe = processed dataframe                                               #
#       k = number of clusters                                                        #
#                                                                                     #
# Output:                                                                             #
#       type Dictionary:   key = erd_image_num ,  value = corresponding cluster num   #
#######################################################################################
def base_clustering(dataframe, k):
    km = KMeans(n_clusters=k, init="k-means++")
    km.fit(dataframe)

    ans_dict = {}
    for k_idx in range(len(dataframe.index)):
        ans_dict[dataframe.index[k_idx]] = km.labels_[k_idx]
    return ans_dict

def main():
    args = parser.parse_args()

    text_dict = concat_texts(args.ocr_results)
    dataframe = doc_vectorize(text_dict)
    cluster_dict = base_clustering(dataframe, args.num_clusters)
    print(cluster_dict)

if __name__ == '__main__':
    main()