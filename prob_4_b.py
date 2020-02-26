# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:30:26 2019

@author: nipun
"""

"""
PROBLEM 4, PART B

"""


from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA as RandomizedPCA
import matplotlib.pyplot as plt
import numpy as np



faces = fetch_lfw_people(min_faces_per_person=60)

print(faces.target_names)
print(faces.images.shape) ##1348 images of 62 x 47 pixels each
n_samples, h, w = faces.images.shape
print(n_samples)

n_components = 150
pca = RandomizedPCA(n_components=n_components, svd_solver='randomized')  ##Randomized PCA for the the first 150 components
x_proj = pca.fit_transform(faces.data)


#Reconstruction 
x_inv_proj = pca.inverse_transform(x_proj)
x_proj_img = np.reshape(x_inv_proj,(1348,62,47))


#The first 24 reconstructed images
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_proj_img[i], cmap='bone')

plt.show()



#Original Pictures (The first 24)
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i], cmap='bone')

plt.show()


