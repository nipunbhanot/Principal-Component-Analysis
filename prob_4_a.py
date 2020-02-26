# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:18:38 2019

@author: nipun
"""


"""
PROBLEM 4, PART A

"""


from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA as RandomizedPCA
import matplotlib.pyplot as plt



faces = fetch_lfw_people(min_faces_per_person=60)

print(faces.target_names)
print(faces.images.shape)
n_samples, h, w = faces.images.shape
print(n_samples)

n_components = 150
pca = RandomizedPCA(n_components=n_components, svd_solver='randomized')  ##Randomized PCA for the the first 150 components
pca.fit(faces.data)

print(pca.components_)  ##These are the first 150 Principal Components

pcacomponents25 = pca.components_[0:25] ##First 25 Principal Components

eigenfaces = pca.components_.reshape((n_components, h, w)) ##Eigenfaces for 150 PCs

eigenfaces25 = pcacomponents25.reshape((25, h, w)) ##Eigenfaces for first 25 PCs


## Plotting EigenFaces for First 25 PCs
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces25[i], cmap='bone')
    
