import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load the dataset
df1 = pd.read_csv("ml18.csv")
f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values

X = np.column_stack((f1, f2))  # Use np.column_stack to create a 2D array

# Plot the original dataset
plt.subplot(511)
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.ylabel('Speeding Feature')
plt.xlabel('Distance Feature')
plt.scatter(f1, f2)

colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# Plot K-Means results
plt.subplot(513)
kmeans_model = KMeans(n_clusters=3).fit(X)
kmeans_labels = kmeans_model.labels_

for i, l in enumerate(kmeans_labels):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l], markersize=6)

plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('K-Means')
plt.ylabel('Speeding Feature')
plt.xlabel('Distance Feature')

# Plot Gaussian Mixture Model results
plt.subplot(515)
gmm = GaussianMixture(n_components=3).fit(X)
gmm_labels = gmm.predict(X)

for i, l in enumerate(gmm_labels):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l], markersize=6)

plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Gaussian Mixture')
plt.ylabel('Speeding Feature')
plt.xlabel('Distance Feature')

plt.tight_layout()  # Adjust layout for better visualization
plt.show()


# ********************short code********************************
# ********************short code********************************
# ********************short code********************************
# ********************short code********************************
# ********************short code********************************


from sklearn import datasets 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() 
#print(iris)
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target) 
model =KMeans(n_clusters=3)
model.fit(X_train,y_train) 
model.score
print('K-Mean: ',metrics.accuracy_score(y_test,model.predict(X_test)))

#-------Expectation and Maximization----------
from sklearn.mixture import GaussianMixture 
model2 = GaussianMixture(n_components=3) 
model2.fit(X_train,y_train)
model2.score
print('EM Algorithm:',metrics.accuracy_score(y_test,model2.predict(X_test)))