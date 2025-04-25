#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[25]:


#read csv file
ingredient = pd.read_csv('C:/Users/User/Desktop/GitHub/Petrol Additives Statistical Analysis.csv')
ingredient.head()


# In[31]:


#data cleaning- to check missing values
print('The total missing values is: ',ingredient.isna().sum().sum())


# # Statistical Analysis

# In[63]:


#1. Summary statistics
sum_stats = ingredient.describe()
sum_stats


# In[65]:


sum_stats.loc['range'] = sum_stats.loc['max'] - sum_stats.loc['min']
sum_stats.loc['IQR'] = sum_stats.loc['75%'] - sum_stats.loc['25%']
sum_stats


# In[27]:


#2. Correlation
correlation = ingredient.corr()
correlation


# In[39]:


#3. ANOVA analysis
anova = stats.f_oneway(ingredient['a'], ingredient['b'], ingredient['c'], ingredient['d'], 
                       ingredient['e'], ingredient['f'], ingredient['g'], ingredient['h'], ingredient['i'])
print("ANOVA:\n", anova)


# ## Data Visualizations
# 

# In[42]:


#Histograms
ingredient.hist(figsize=(10, 8))
plt.show()


# In[74]:


#Box plots-detection of outliers
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
for column in ingredient.columns:
    sns.boxplot(x=ingredient[column], orient="v", width=0.4)
    plt.title(f"Additives {column}", fontsize=14)
    plt.show()


# In[75]:


#Heatmap for correlation
plt.figure(figsize=(14, 8))
plt.title('Heatmap')
sns.heatmap(correlation, annot=True)
plt.show()


# # Unsupervised Learning (Clustering)

# In[114]:


#K-Means Clustering
get_ipython().system('pip install yellowbrick')


# In[8]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


#read csv file
ingredient2 = pd.read_csv('C:/Users/User/Desktop/GitHub/Petrol Additives Statistical Analysis.csv')
ingredient2.head()


# In[13]:


scaler = StandardScaler()

# Fit the scaler to data and transform it
ingredient2_scaled = scaler.fit_transform(ingredient2)


# In[14]:


#Determination of best number of clusters with elbow method
## Instantiate the K-means model
k_mean = KMeans(random_state=42)

# Instantiate the KElbowVisualizer with a range of clusters
visualizer = KElbowVisualizer(k_mean, k=(2,10))

# Fit the visualizer to the data
visualizer.fit(ingredient2_scaled)
visualizer.show() 


# In[15]:


#set the number of cluster or with elbow method 
kmeans = KMeans(n_clusters=5)
#fit the model
kmeans.fit(ingredient2_scaled)


# In[16]:


#get the cluster center
centroids = kmeans.cluster_centers_
centroids


# In[17]:


#label assigned to each data point
kmeans_labels = kmeans.labels_


# In[18]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(ingredient2_scaled)

# Concatenate the PCA-transformed data with the cluster labels
data_with_labels = np.column_stack((X_pca, kmeans_labels))

# Visualize clusters
plt.scatter(data_with_labels[:, 0], data_with_labels[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5)
# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red') 
plt.title('K-Means Clustering')
plt.show()

