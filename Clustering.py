# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:42:59 2022

@author: Prashant Kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

df = pd.read_csv("Iris.csv")
#print(df.head())

data = df.drop(['Species'], axis = 1)

#Scaling the data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(data)

#Doing PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
prComp = pca.fit_transform(X_scaled)
prDF = pd.DataFrame(data=prComp, columns=['X1', 'X2'])

#Making a copy of dataframe
reduced_data = prDF.copy()
red_df = prDF.copy()


#plotting the dataframe
plt.scatter(prDF['X1'],prDF['X2'])
plt.show()


#Implementing the Kmeans clustering
km = KMeans(n_clusters = 3)
km.fit(prDF)
y_predict = km.predict(prDF)
cluster_center = km.cluster_centers_
# print(cluster_center)

prDF['cluster'] = y_predict

df1 = prDF[prDF['cluster']==0]
df2 = prDF[prDF['cluster']==1]
df3 = prDF[prDF['cluster']==2]

#Plotting the clusters
plt.scatter(df1.X1,df1['X2'],color='green')
plt.scatter(df2.X1,df2['X2'],color='red')
plt.scatter(df3.X1,df3['X2'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()


print("\nSum of squared distances of samples to their closest cluster centre :%.2f"%km.inertia_)
def purity_score(y_true, y_pred):
    # computing contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print the contingency_matrix
    # calculating optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    #Return cluster accuracy

    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

df['cluster']= y_predict

df["Species"].replace({"Iris-setosa": 0,"Iris-versicolor" : 1,"Iris-virginica":2}, inplace=True)
species = list(df["Species"])
purity = purity_score(species, y_predict)
print("\nThe purity score after examples are assigned to clusters : %.2f"%purity)


#Implementing K-means clustering for number of clusters (K) as 2, 3, 4, 5, 6 and 7 :-
sse = []
score = [] #Purity Score
k_rng = [2,3,4,5,6,7]
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(prDF[['X1','X2']])
    sse.append(km.inertia_)
    y_pred = km.predict(prDF[['X1','X2']])
    score.append(purity_score(species,y_pred))

#Plotting of K vs distortion measure :-
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

#Purity score for different clusters :-
plt.xlabel('K')
plt.ylabel('Purity Score')
plt.plot(k_rng,score)
plt.show()

#%%
#GMM
# GMM

from sklearn.mixture import GaussianMixture
K = 3
gmm = GaussianMixture(n_components = K)
gmm.fit(reduced_data)
GMM_prediction = gmm.predict(reduced_data)

print("Total data log likelihood at the last iteration of the GMM : %.2F"%gmm.score(reduced_data[['X1','X2']]))



reduced_data['cluster'] = GMM_prediction


df_1 = reduced_data[reduced_data['cluster']==0]
df_2 = reduced_data[reduced_data['cluster']==1]
df_3 = reduced_data[reduced_data['cluster']==2]

#Plotting the clusters
plt.scatter(df_1.X1,df_1['X2'],color='green')
plt.scatter(df_2.X1,df_2['X2'],color='red')
plt.scatter(df_3.X1,df_3['X2'],color='black')
plt.scatter(gmm.means_[:,0],gmm.means_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

purity_gmm = purity_score(species, GMM_prediction)
print("The purity score after examples are assigned to clusters : %.2f"%(100*purity_gmm))



log_liklihood = []
score_gmm = [] #Purity Score
g_rng = [2,3,4,5,6,7]
for g in g_rng:
    gmm = GaussianMixture(n_components = g)
    gmm.fit(reduced_data[['X1','X2']])
    log_liklihood.append(gmm.score(reduced_data[['X1','X2']]))
    GMM_prediction_ = gmm.predict(reduced_data[['X1','X2']])
    score_gmm.append(purity_score(species,GMM_prediction_))
#Plotting K vs total data log likelihood :-
plt.xlabel('K')
plt.ylabel('log liklihood')
plt.plot(g_rng,log_liklihood)
plt.show()

#Purity score for the different number of clusters :-
plt.xlabel('K')
plt.ylabel('Purity Score')
plt.plot(g_rng,score_gmm)
plt.show()

#%%
#DBSCAN

from sklearn.cluster import DBSCAN
dbscan_model=DBSCAN(eps=1, min_samples=5).fit(red_df)
DBSCAN_predictions = dbscan_model.labels_
print(np.unique(DBSCAN_predictions))

red_df['cluster'] = DBSCAN_predictions

newdf1 = red_df[red_df['cluster']==0]
newdf2 = red_df[red_df['cluster']==1]
plt.scatter(newdf1.X1,newdf1['X2'],color='green')
plt.scatter(newdf2.X1,newdf2['X2'],color='red')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()



