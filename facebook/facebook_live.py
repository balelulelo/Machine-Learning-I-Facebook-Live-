import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('./pesbuk_live.csv')

# Selecting relevant columns for clustering
kolom_2 = ['num_reactions', 'num_comments', 'num_shares']
df_train = df[kolom_2]

# Finding the optimal number of clusters using Elbow Method and Silhouette Score
wcss = []
scores = []
for i in range(2, 10):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(df_train)
    wcss.append(km.inertia_)
    labels = km.labels_
    silhouette_avg = silhouette_score(df_train, labels)
    scores.append(silhouette_avg)
    print(f'WCSS score for n_cluster = {i} is {wcss[-1]}')
    print(f'Silhouette score for n_clusters = {i} is {silhouette_avg}')

# Elbow Method Plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), wcss, marker='o', linestyle='-', color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), scores, marker='o', linestyle='-', color='g')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.show()

# Applying KMeans Clustering with K=3
kmeansmodel = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeansmodel.fit_predict(df_train)

# Creating 3D scatter plot with plotly
trace1 = go.Scatter3d(
    x=df_train[y_kmeans == 0].iloc[:, 0],
    y=df_train[y_kmeans == 0].iloc[:, 1],       
    z=df_train[y_kmeans == 0].iloc[:, 2],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.8),
    name='Cluster 1'
)

trace2 = go.Scatter3d(
    x=df_train[y_kmeans == 1].iloc[:, 0],
    y=df_train[y_kmeans == 1].iloc[:, 1],
    z=df_train[y_kmeans == 1].iloc[:, 2],
    mode='markers',
    marker=dict(size=8, color='blue', opacity=0.8),
    name='Cluster 2'
)

trace3 = go.Scatter3d(
    x=df_train[y_kmeans == 2].iloc[:, 0],
    y=df_train[y_kmeans == 2].iloc[:, 1],
    z=df_train[y_kmeans == 2].iloc[:, 2],
    mode='markers',
    marker=dict(size=8, color='green', opacity=0.8),
    name='Cluster 3'
)

# Scatter plot for centroids
centroids = go.Scatter3d(
    x=kmeansmodel.cluster_centers_[:, 0],
    y=kmeansmodel.cluster_centers_[:, 1],
    z=kmeansmodel.cluster_centers_[:, 2],
    mode='markers',
    marker=dict(size=12, color='black', symbol='diamond', opacity=1),
    name='Centroids'
)

# Layout setup
layout = go.Layout(
    title='Hasil KMeans Clustering',
    scene=dict(
        xaxis_title='HP',
        yaxis_title='ATK',
        zaxis_title='DEF'
    ),
    showlegend=True
)

# Combine traces into figure
fig = go.Figure(data=[trace1, trace2, trace3, centroids], layout=layout)
fig.show()

from sklearn.cluster import AgglomerativeClustering

linkage_col = ['ward', 'complete', 'average', 'single']
scores_all = [[] for _ in range(len(linkage_col))]
for j in range(len(linkage_col)):
    print('Linkage: ', linkage_col[j])
    scores = []
    for i in range(2, 10):
        AC = AgglomerativeClustering(n_clusters=i, linkage=linkage_col[j])
        AC.fit(df_train)
        labels = AC.labels_
        silhouette_avg = silhouette_score(df_train, labels)
        scores.append(silhouette_avg)
        print('Silhouette score for n_clusters = ' + str(i) + ' is ' + str(silhouette_avg))
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
    title='Hasil Agglomerative Clustering 3D',
    scene=dict(
        xaxis_title='HP',
        yaxis_title='ATK',
        zaxis_title='DEF'
    ),
    showlegend=True
)

fig = go.Figure(data=traces, layout=layout)
fig.show()
