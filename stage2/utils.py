import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score

# Saves a line graph of different k values against their within-cluster sum square error
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
  plt.savefig("graph_Within-cluster-SSE.png")

# Returns the global optimal k based on the silhouette scores and saves
# the graph of k against silhouette scores as "graph_silhouette_scores.png".
def graph_distortion_silhouette(features):
  min_k = 2
  max_k = features.shape[0]

  k_nums = [i for i in range(min_k, max_k)]
  sil_scores = []

  for k in k_nums:
    k_means = KMeans(n_clusters=k, init="k-means++").fit(features)
    sil_score = silhouette_score(features, k_means.labels_, metric='euclidean')

    sil_scores.append(sil_score)

  plt.plot(k_nums, sil_scores)
  plt.xlabel("k")
  plt.ylabel("silhouette_score")
  plt.savefig("graph_silhouette_scores.png")