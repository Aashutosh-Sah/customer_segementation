import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset (Replace with actual dataset)
df = pd.read_csv("customer_data.csv")


# Data Preprocessing
df.dropna(inplace=True)  # Drop missing values
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)  # One-hot encoding

# Feature Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Finding optimal clusters using Silhouette Score
sil_scores = []
k_values = range(2, 11)  # Testing 2 to 10 clusters
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    sil_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, sil_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters')
plt.show()

# Applying K-Means with optimal clusters (assuming best k from silhouette analysis)
best_k = k_values[np.argmax(sil_scores)]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_pca['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualizing Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=df_pca['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation')
plt.show()
print(f"Optimal Number of Clusters: {best_k}")
