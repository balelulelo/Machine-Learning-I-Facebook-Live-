import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go  

# load dataset
df = pd.read_csv('pesbuk_live.csv')

# drop every line that includes Not a Number (NaN) values
df = df.dropna()
# pick numeric features
num_features = ['num_comments', 'num_shares', 'num_likes']
df_train = df[num_features]

# if there are still NaN values after feature selection, fill with mean
df_train = df_train.fillna(df_train.mean())

# store wcss and silhouette score
wcss = []
scores = []

for i in range(2, 10):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(df_train)
    wcss.append(km.inertia_)

    labels = km.labels_
    silhouette_avg = silhouette_score(df_train, labels)
    scores.append(silhouette_avg)
    
    print(f'WCSS score for n_clusters = {i} is {wcss[-1]}')
    print(f'Silhouette score for n_clusters = {i} is {silhouette_avg}')

# combine elbow method & silhouette for better comparison
fig, ax1 = plt.subplots()

ax1.set_xlabel('No. of Clusters')
ax1.set_ylabel('WCSS', color='blue')
ax1.plot(range(2, 10), wcss, marker='o', linestyle='-', color='blue', label='WCSS (Elbow Method)')
ax1.tick_params(axis='y', labelcolor='blue')

# add 2nd Y axis for silhouette score
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='red')
ax2.plot(range(2, 10), scores, marker='s', linestyle='--', color='red', label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle('Elbow Method & Silhouette Score', fontsize=16)
fig.tight_layout()
plt.show()


# based on elbow method and silhouette, pick K = 3
kmeansmodel = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeansmodel.fit_predict(df_train)

# visualize clustering in 3D with plotly
trace1 = go.Scatter3d(
    x=df_train.iloc[y_kmeans == 0, 0],
    y=df_train.iloc[y_kmeans == 0, 1],
    z=df_train.iloc[y_kmeans == 0, 2],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.8),
    name='Cluster 1'
)

trace2 = go.Scatter3d(
    x=df_train.iloc[y_kmeans == 1, 0],
    y=df_train.iloc[y_kmeans == 1, 1],
    z=df_train.iloc[y_kmeans == 1, 2],
    mode='markers',
    marker=dict(size=8, color='blue', opacity=0.8),
    name='Cluster 2'
)

trace3 = go.Scatter3d(
    x=df_train.iloc[y_kmeans == 2, 0],
    y=df_train.iloc[y_kmeans == 2, 1],
    z=df_train.iloc[y_kmeans == 2, 2],
    mode='markers',
    marker=dict(size=8, color='green', opacity=0.8),
    name='Cluster 3'
)

# scatter plot for visualizing centroids
centroids = go.Scatter3d(
    x=kmeansmodel.cluster_centers_[:, 0],
    y=kmeansmodel.cluster_centers_[:, 1],
    z=kmeansmodel.cluster_centers_[:, 2],
    mode='markers',
    marker=dict(size=12, color='black', symbol='diamond', opacity=1),
    name='Centroids'
)

# Layout plot
layout = go.Layout(
    title='KMeans Clustering Results',
    scene=dict(
        xaxis_title='num_comments',
        yaxis_title='num_shares',
        zaxis_title='num_likes'
    ),
    showlegend=True
)

# combine plots
fig = go.Figure(data=[trace1, trace2, trace3, centroids], layout=layout)

# show the plot
fig.show()

from sklearn.cluster import AgglomerativeClustering

linkage_col = ['ward', 'complete', 'average', 'single']
scores_all = [[] for _ in range(len(linkage_col))]
for j in range (len(linkage_col)):
    print('Linkage: ', linkage_col[j])
    scores = []
    for i in range(2, 10):
        AC = AgglomerativeClustering(n_clusters=i, linkage = linkage_col[j]) # --> word method
        AC.fit(df_train)

        labels = AC.labels_
        silhouette_avg = silhouette_score(df_train, labels)
        scores.append(silhouette_avg)
        print('silhoutte score for n_clusters = ' + str(i) + ' is ' + str(silhouette_avg))
    print("\n##########################\n")
    scores_all[j] = scores

colors = sns.color_palette("Set1", len(linkage_col))

for j in range(len(linkage_col)):
    plt.plot(range(2, 10), scores_all[j], marker='o', color=colors[j], label=linkage_col[j])

plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters for Different Linkage Methods')
plt.legend()
plt.grid(True)
plt.show()

# perform agglomerative clustering (3 clusters)

number_of_cluster = 3
AC = AgglomerativeClustering(n_clusters=number_of_cluster, linkage='average')
labels = AC.fit_predict(df_train)

df_train_with_labels = np.column_stack((df_train, labels))

centroids = []
for cluster_label in range(number_of_cluster):
    cluster_data = df_train_with_labels[df_train_with_labels[:, -1] == cluster_label]
    cluster_centroid = np.mean(cluster_data[:, :-1], axis=0)
    centroids.append(cluster_centroid)
centroids = np.array(centroids)

traces = []
colors = ['red', 'blue', 'green']

for cluster_label in range(number_of_cluster):
    cluster_data = df_train_with_labels[df_train_with_labels[:, -1] == cluster_label]
    trace = go.Scatter3d(
        x=cluster_data[:, 0],
        y=cluster_data[:, 1],
        z=cluster_data[:, 2],
        mode='markers',
        marker=dict(size=8, color=colors[cluster_label], opacity=0.8),
        name=f'Cluster {cluster_label + 1}'
    )
    traces.append(trace)

centroid_trace = go.Scatter3d(
    x=centroids[:, 0],
    y=centroids[:, 1],
    z=centroids[:, 2],
    mode='markers',
    marker=dict(size=12, color='black', symbol='diamond', opacity=1),
    name='Centroids'
)

traces.append(centroid_trace)

layout = go.Layout(
    title='Agglomerative Clustering 3D',
    scene=dict(
        xaxis_title='num_comments',
        yaxis_title='num_shares',
        zaxis_title='num_likes'
    ),
    showlegend=True
)

fig = go.Figure(data=traces, layout=layout)
fig.show()

import scipy.cluster.hierarchy as sch

linked = sch.linkage(df_train, 'average')


plt.figure(figsize=(15, 10))
dendrogram = sch.dendrogram(linked, orientation='top',distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Data Point')
plt.ylabel('Distance')
plt.axhline(y=1500, color='black', linestyle='--') 
# --> data will be dividedinto 3 clusters
plt.show()


