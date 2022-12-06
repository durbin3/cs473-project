import ast
import math
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import normalize


# Added "ERD_LABEL_" to avoid key collision with words in ocr results.
ERD_LABELS = ["ERD_LABEL_entity", "ERD_LABEL_weak_entity",
              "ERD_LABEL_rel", "ERD_LABEL_ident_rel",
              "ERD_LABEL_rel_attr", "ERD_LABEL_many", "ERD_LABEL_one"]

RELATIONSHIP_SUFFIX_DELIMETER = "#"

#######################################################################################
# Input:                                                                              #
#       ocr_path = path to ocr result text files                                      #
#                                                                                     #
# Output:                                                                             #
#       type Dict: key = erd_image_num , value = concatenated string                  #
#######################################################################################

def ocr_concat_texts(ocr_path, with_relationship_as_suffix=False):
    ps = PorterStemmer()
    erd_dict = {}
    for txt_file in glob.glob(ocr_path+"/*.txt"):
        file_num = os.path.basename(txt_file)[:-4]
        with open(txt_file,"r") as f:
            erd_dict[file_num] = []
            for line in f:
                word_list = ast.literal_eval(re.search("\[.*\]",line).group())

                if with_relationship_as_suffix:
                    relationship = re.search(": .* \[",line).group()
                    relationship = relationship[2: len(relationship) - 2]
                    
                    # Append relationship suffix to every word in entity's name.
                    for i in range(len(word_list)):
                      word_list[i] = " ".join([relationship + RELATIONSHIP_SUFFIX_DELIMETER + sub_word for sub_word in word_list[i].split()])

                erd_dict[file_num].extend(word_list)

    return dict(dict(sorted(erd_dict.items())))


#######################################################################################
# Input:                                                                              #
#       erd_dict = type Dict:  key = erd_image_num , value = list of words in the img #
#                                                                                     #
# Output:                                                                             #
#       ans_df = type DataFrame: DataFrame of processed texts                         #
#######################################################################################
def ocr_doc_vectorize(erd_dict, with_relationship_as_suffix=False):
    ps = PorterStemmer()
    erd_list = list(erd_dict.values())
    tmp_erd_string = [" ".join(doc) for doc in erd_list]

    ###########stemming###########
    for doc_idx in range(len(tmp_erd_string)):
        
        tmp_string = []
        if with_relationship_as_suffix:
            # Remove the relationship type suffix before stemming. 
            # Example input word with relationship: "entity_hello world" 
            for label in tmp_erd_string[doc_idx].split():
                # print("label", label)
                relationship, word = label.split(RELATIONSHIP_SUFFIX_DELIMETER, 1)
                tmp_string.append(relationship + RELATIONSHIP_SUFFIX_DELIMETER + ps.stem(word))
        else:
            tmp_string = [ps.stem(word) for word in tmp_erd_string[doc_idx].split()]

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

# Returns the global optimal k based on the silhouette scores and saves
# the graph of k against silhouette scores as "graph_silhouette_scores.png".
def get_optimal_k_silhouette(features):
  min_k = 2
  max_k = features.shape[0]

  k_nums = [i for i in range(min_k, max_k)]

  best_k = -1
  best_score = -1

  for k in k_nums:
    k_means = KMeans(n_clusters=k, init="k-means++").fit(features)
    sil_score = silhouette_score(features, k_means.labels_, metric='euclidean')

    if (sil_score > best_score):
      best_k = k
      best_score = sil_score

  return best_k

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

# Module 4 baseline clustering
def method0_clustering(ocr_results_path, k=0):
  text_dict = ocr_concat_texts(ocr_results_path)
  df_ocr_features = ocr_doc_vectorize(text_dict)

  if k == 0:
    k = get_optimal_k_silhouette(df_ocr_features)

  return base_clustering(df_ocr_features, k)

# Method 1: Use the features: number of entities, number of relationships, number of weak
# entities, ... etc. Also, use entity names, attribute names, ... etc.
def method1_clustering(ocr_results_path, ob_results_path, k=0):
  text_dict = ocr_concat_texts(ocr_results_path)
  df_ocr_features = ocr_doc_vectorize(text_dict)
  df_erd_features = parse_ob_output(ob_results_path)

  df_features = df_ocr_features.join(df_erd_features, on=df_ocr_features.index, how='left')
  df_features = pd.DataFrame(normalize(df_features), index= df_features.index,columns=df_features.columns)

  if k == 0:
    k = get_optimal_k_silhouette(df_features)

  return base_clustering(df_features, k)

def method3_clustering(ocr_results_path, ob_results_path, k=0):
    text_dict = ocr_concat_texts(ocr_results_path, with_relationship_as_suffix=True)
    df_ocr_features = ocr_doc_vectorize(text_dict, with_relationship_as_suffix=True)

    df_erd_features = parse_ob_output(ob_results_path)

    df_features = df_ocr_features.join(df_erd_features, on=df_ocr_features.index, how='left')
    df_features = pd.DataFrame(normalize(df_features), index= df_features.index,columns=df_features.columns)

    if k == 0:
      k = get_optimal_k_silhouette(df_features)

    return base_clustering(df_features, k)