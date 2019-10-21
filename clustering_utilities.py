import rbo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def compute_rbo_matrix(kw_list, p):
    rbo_mat = np.zeros(shape=(len(kw_list), len(kw_list)))
    for i in range(len(kw_list)):
        for j in range(i, len(kw_list)):
            rbo_mat[i][j] = rbo_mat[j][i] = rbo.rbo_ext(kw_list[i], kw_list[j], p)
    return rbo_mat

# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
def plot_elbow(max_k, X):
    wcss = []
    for i in range(1, max_k):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# https://stackoverflow.com/questions/12550929/how-to-make-all-lists-in-a-list-of-lists-the-same-length-by-adding-to-them
def create_clus_output(kw_list, k_means_pred):
    max_len = max(map(len, kw_list))
    for row in kw_list:
        if len(row) < max_len:
            row.extend(['null'] * (max_len - len(row)))
    kw_df = pd.DataFrame(kw_list, columns=range(max_len))
    pred_df = pd.DataFrame(k_means_pred, columns=['cluster'])
    kw_clus = pd.concat([pred_df, kw_df], axis=1)
    return kw_clus