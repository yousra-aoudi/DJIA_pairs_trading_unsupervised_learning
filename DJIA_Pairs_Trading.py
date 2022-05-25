# 1. Problem definition
"""
A pairs trading strategy constructs a portfolio of correlated assets with similar market risk factor exposure.
Temporary price discrepancies in these assets can create opportunities to profit through a long position in one
instrument and a short position in another. A pairs trading strategy is designed to eliminate market risk and exploit
these temporary discrepancies in the relative returns of stocks.

The key to successful pairs trading is the ability to select the right pairs of assets to be used.

Our goal in this case study is to perform clustering analysis on the stocks of DJIA  and come up with pairs for a pairs
trading strategy.

The data of the stocks of DJIA, obtained using pandas_datareader from yahoo finance. It includes price data from
2015 onwards.

"""

# 2. Loading the data and Python packages
# 2.1. Loading the python packages
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import pandas_datareader as dr


# Import Model Packages
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold


# Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 2.2. Loading data
# The data already obtained from yahoo finance is imported.
dataset = read_csv('DJIAData.csv',index_col=0, parse_dates=True)

# 3. Exploratory Data Analysis
# 3.1. Descriptive Statistics
# Data type and description of data
set_option('display.width', 100)
print(type(dataset))
print('dataset \n',dataset.tail(5))

# describe data
set_option('precision', 3)
print('describe data \n',dataset.describe())

# 3.2. Data Visualization
"""
We will take a detailed look into the visualization post clustering.
"""

# 4. Data Preparation
# 4.1. Data Cleaning
# Checking for any null values and removing the null values
print('Null Values =', dataset.isnull().values.any())

"""
Removing the columns with more than 30% missing values.
"""
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
missing_fractions.head(10)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print('dataset shape \n', dataset.shape)

"""
Given that there are null values drop the row containing the null values.
"""
# Fill the missing values with the last value available in the dataset.
dataset = dataset.fillna(method='ffill')
print('dataset \n', dataset.head(10))

# 4.2. Data Transformation
"""
For the purpose of clustering, we will be using annual returns and variance as the variables as they are the indicators 
of the stock performance and its volatility. Let us prepare the return and volatility variables from the data.
"""
# Calculate average annual percentage return and volatilities over a theoretical one-year period
returns = dataset.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
data = returns
print('data \n', data.tail(10))

"""
All the variables should be on the same scale before applying clustering, otherwise a feature with large values will 
dominate the result. We use StandardScaler in sklearn to standardize the dataset’s features onto unit scale (mean = 0 
and variance = 1).
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data)
rescaledDataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
# summarize transformed data
print('rescaled data \n',rescaledDataset.head(2))
X = rescaledDataset

# 5. Evaluate Algorithms and Models
"""
Three models that we will look at for this clustering case study
- KMeans
- Hierarchical Clustering
- Affinity Propagation
"""

# 5.1. K-Means Clustering
print('K-means clustering')
# 5.1.1. Finding optimal number of clusters
"""
We know that k-means initially assigns data points to clusters randomly and then calculates centroids or mean values. 
Further, it calculates the distances within each cluster, squares these, and sums them to get the sum of squared errors.
The basic idea is to define k clusters so that the total within-cluster variation (or error) is minimized. 
The following two methods are useful in finding the number of clusters in k-means:
- Sum of square errors (SSE) within clusters
- Silhouette score
"""

distorsions = []
max_loop = 20
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), distorsions)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors')
plt.grid(True)
plt.title('K-means method: Optimal number of clusters using the Elbow method')
plt.savefig('K-means method: Optimal number of clusters using the Elbow method.png')
plt.show()

# Silhouette score

from sklearn import metrics

silhouette_score = []
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k,  random_state=10, n_init=10, n_jobs=-1)
    kmeans.fit(X)
    silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), silhouette_score)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.grid(True)
plt.title('K-means method: Optimal number of clusters using the Silhouette method')
plt.savefig('K-means method: Optimal number of clusters using the Silhouette method.png')
plt.show()

# 5.1.2. Clustering and Visualisation
"""
Let us build the k-means model with six clusters and visualize the results.
"""
nclust=5
# Fit with k-means
k_means = cluster.KMeans(n_clusters=nclust)
k_means.fit(X)

# Extracting labels
target_labels = k_means.predict(X)

centroids = k_means.cluster_centers_
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", label=X.index)
ax.set_title('k-Means results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
plt.savefig('k-Means results.png')
plt.show()

"""
Let us check the elements of the clusters
"""

# show number of stocks in each cluster
clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
# clustered stock with its cluster label
clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
clustered_series = clustered_series[clustered_series != -1]

plt.figure(figsize=(12, 7))
plt.barh(range(len(clustered_series.value_counts())),clustered_series.value_counts())
plt.title('K-means method: Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number')
plt.savefig('K-means method: Cluster Member Counts.png')
plt.show()

"""
The number of stocks in a cluster range from around 1 to 11. The distribution is not equal, and having only one stock 
per cluster is may not be relevant for the case study. 
"""

# 5.2. Hierarchical Clustering
print('Hierarchical clustering')
"""  
In the first step, we look at the hierarchy graph and check for the number of clusters.
"""
# 5.2.1. Building Hierarchy Graph/ Dendogram
"""
The hierarchy class has a dendrogram method that takes the value returned by the linkage method of the same class. 
The linkage method takes the dataset and the method to minimize distances as parameters. We use ward as the method 
since it minimizes the variance of distances between the clusters:
"""

from scipy.cluster.hierarchy import dendrogram, linkage, ward

#Calulate linkage
Z = linkage(X, method='ward')
print('Z[0]', Z[0])

"""
The best way to visualize an agglomerative clustering algorithm is through a dendrogram, which displays a cluster 
tree, the leaves being the individual stocks and the root being the final single cluster. The distance between each 
cluster is shown on the y- axis. The longer the branches are, the less correlated the two clusters are:
"""

#Plot Dendogram
plt.figure(figsize=(10, 7))
plt.title("Hierarchical method: Stocks Dendrograms")
dendrogram(Z, labels=X.index)
plt.savefig("Hierarchical method: Stocks Dendrograms.png")
plt.show()

"""
This chart can be used to visually inspect the number of clusters that would be created for a selected distance 
threshold (although the names of the stocks on the horizontal axis are not very clear, we can see that they are 
grouped into several clusters). The number of vertical lines a hypothetical straight, horizontal line will pass through 
is the number of clusters created for that distance threshold value. 
Choosing a threshold cut at 3 yields four clusters, as confirmed in the following Python code:
"""

distance_threshold = 3
clusters = fcluster(Z, distance_threshold, criterion='distance')
chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
chosen_clusters['cluster'].unique()
print(chosen_clusters['cluster'].unique())

# 5.2.2. Clustering and visualization
"""
Let us build the hierarchical clustering model with four clusters and visualize the results:
"""
nclust = 6
hc = AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='ward')
clust_labels1 = hc.fit_predict(X)
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow")
ax.set_title('Hierarchical Clustering')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.savefig('Hierarchical Clustering.png')
plt.show()

# 5.3. Affinity Propagation
print('Affinity propagation')
ap = AffinityPropagation()
ap.fit(X)
clust_labels2 = ap.predict(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow")
ax.set_title('Affinity')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
plt.colorbar(scatter)
plt.savefig('Affinity propagation.png')
plt.show()

"""
The affinity propagation model with the chosen hyper-parameters produced many more clusters than k-means and hierarchical
clustering. There is some clear grouping, but also more overlap due to the larger number of clusters.
"""

# 5.3.1 Cluster Visualisation
cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

no_clusters = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % no_clusters)
# Plot exemplars

X_temp = np.asarray(X)
plt.close('all')
plt.figure(1)
plt.clf()

fig = plt.figure(figsize=(8,6))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(no_clusters), colors):
    class_members = labels == k
    cluster_center = X_temp[cluster_centers_indices[k]]
    plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X_temp[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
plt.show()

# show number of stocks in each cluster
clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
# clustered stock with its cluster label
clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
clustered_series_ap = clustered_series_ap[clustered_series != -1]

plt.figure(figsize=(12, 7))
plt.barh(range(len(clustered_series_ap.value_counts())), clustered_series_ap.value_counts())
plt.title('Affinity Propagation: Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number')
plt.savefig('Affinity Propagation: Stocks in cluster.png')
plt.show()

# 5.4. Cluster evaluation
from sklearn import metrics
print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

"""
Results
---------
Estimated number of clusters: 7
km 0.3753473352959341
hc 0.38720151457306523
ap 0.4008625565240457
-----------
Given the km propagation performs the best, we will use it for the next step of our case study, with a number of 
cluster = 7.
"""
# Visualising the return within a cluster
"""
Let us visualize the results of the clusters in order to understand the intuition behind clustering.
"""
# all stock with its cluster label (including -1)
clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
# clustered stock with its cluster label
clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
clustered_series = clustered_series[clustered_series != -1]

# get the number of stocks in each cluster
counts = clustered_series_ap.value_counts()

# let's visualize some clusters
cluster_vis_list = list(counts[(counts<25) & (counts>1)].index)[::-1]
print('cluster visualisation list\n',cluster_vis_list)

CLUSTER_SIZE_LIMIT = 9999
counts = clustered_series.value_counts()
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
print("Clusters formed: %d" % len(ticker_count_reduced))
print("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())
print('Pairs to evaluate list \n',ticker_count_reduced)

# plot a handful of the smallest clusters
plt.figure(figsize=(12, 7))
cluster_vis_list[0:min(len(cluster_vis_list), 4)]
print('cluster visualisation list\n',cluster_vis_list[0:min(len(cluster_vis_list), 4)])


for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(dataset.loc[:"2022-05-20", tickers].mean())
    data = np.log(dataset.loc[:"2022-05-20", tickers]).sub(means)
    data.plot(title='Stock Time Series for Cluster %d' % clust)
    plt.savefig('Stock Time Series for Cluster %d.png' % clust)
    plt.show()

# 6. Pairs Selection
# 6.1. Cointegration and Pair Selection Function


def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

from statsmodels.tsa.stattools import coint
cluster_dict = {}
print('tickers \n',tickers)
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(dataset[tickers].dropna())
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs

pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])
print("Number of pairs found : %d" % len(pairs))
print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))
print('pairs \n',pairs)


# 6.2. Pair Visualization
from sklearn.manifold import TSNE
import matplotlib.cm as cm
stocks = np.unique(pairs)
X_df = pd.DataFrame(index=X.index, data=X).T

in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.T.loc[stocks]

X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)
plt.figure(1, facecolor='white', figsize=(16, 8))
plt.clf()
plt.axis('off')
for pair in pairs:
    # print(pair[0])
    ticker1 = pair[0]
    loc1 = X_pairs.index.get_loc(pair[0])
    x1, y1 = X_tsne[loc1, :]
    # print(ticker1, loc1)

    ticker2 = pair[0]
    loc2 = X_pairs.index.get_loc ( pair[1] )
    x2, y2 = X_tsne[loc2, :]

    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
plt.title('T-SNE Visualization of Validated Pairs')

# zip joins x and y coordinates in pairs
for x, y, name in zip(X_tsne[:, 0], X_tsne[:, 1], X_pairs.index):
    label = name
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
plt.savefig('T-SNE Visualization of Validated Pairs.png')
plt.show()

"""
Analysis - The clustering techniques do not directly help in stock trend prediction. However, they can be effectively 
used in portfolio construction for finding the right pairs, which eventually help in risk mitigation and one can achieve 
superior risk adjusted returns.
"""