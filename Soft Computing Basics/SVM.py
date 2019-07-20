# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 08:47:36 2017

@author: Manas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=20, centers=2, random_state=6)

clf = svm.SVC(kernel='linear')
clf.fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=35)

ax = plt.gca()
xlimit = ax.get_xlim()
ylimit = ax.get_ylim()

xx = np.linspace(xlimit[0], xlimit[1])
yy = np.linspace(ylimit[0], ylimit[1])
YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],linestyles=['dashed', 'solid', 'dashed'])
plt.show()