import numpy as np
import numpy.random as rnd
import os

import matplotlib
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
# 3D dinension dataset
# page 272
#-----------------------------------------------------------------
rnd.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * rnd.randn(m)

X = X - X.mean(axis=0)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
X2D_inv = pca.inverse_transform(X2D)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
     def __init__(self, xs, ys, zs, *args, **kwargs):
         FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
         self._verts3d = xs, ys, zs

     def draw(self, renderer):
         xs3d, ys3d, zs3d = self._verts3d
         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
         FancyArrowPatch.draw(self, renderer)

axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X2D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X2D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
ax.plot([0], [0], [0], "k.")

for i in range(m):
     if X[i, 2] > X2D_inv[i, 2]:
         ax.plot([X[i][0], X2D_inv[i][0]], [X[i][1], X2D_inv[i][1]], [X[i][2], X2D_inv[i][2]], "k-")
     else:
         ax.plot([X[i][0], X2D_inv[i][0]], [X[i][1], X2D_inv[i][1]], [X[i][2], X2D_inv[i][2]], "k-", color="#505050")

ax.plot(X2D_inv[:, 0], X2D_inv[:, 1], X2D_inv[:, 2], "k+")
ax.plot(X2D_inv[:, 0], X2D_inv[:, 1], X2D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

#-----------------------------------------------------------------
# Projection
# page 273
#-----------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)
plt.show()

#-----------------------------------------------------------------
# Swiss Roll
# page 274
#-----------------------------------------------------------------
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

axes = [-11.5, 14, -2, 23, -12, 15]
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

#-----------------------------------------------------------------
# Swiss Roll ????????????
# page 274
#-----------------------------------------------------------------

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)
plt.show()

#-----------------------------------------------------------------
# Swiss Roll ????????????
# page 275 3th ??????
#-----------------------------------------------------------------

from matplotlib import gridspec

axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot_wireframe(5, x2, x3, alpha=0.5)
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()

#-----------------------------------------------------------------
# Swiss Roll ????????????
# page 275 4th ??????
#-----------------------------------------------------------------

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

#-----------------------------------------------------------------
# Swiss Roll ????????????
# page 275 1st ??????
#-----------------------------------------------------------------

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
plt.show()


#-----------------------------------------------------------------
# Swiss Roll ????????????
# page 275 2nd ??????
#-----------------------------------------------------------------
fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.plot([4, 15], [0, 22], "b-", linewidth=2)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

#-----------------------------------------------------------------
# PCA, ????????????
# page 277
#-----------------------------------------------------------------
rnd.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * rnd.randn(m)

print('\nX = \n   {}'.format(X))

X_centered = X - X.mean(axis=0)

angle = np.pi / 5
stretch = 5
m = 200

rnd.seed(3)
X = rnd.randn(m, 2) / 10
X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch
X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate

u1 = np.array([np.cos(angle), np.sin(angle)])
u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])
u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])

X_proj1 = X.dot(u1.reshape(-1, 1))
X_proj2 = X.dot(u2.reshape(-1, 1))
X_proj3 = X.dot(u3.reshape(-1, 1))

plt.figure(figsize=(8,4))
plt.subplot2grid((3,2), (0, 0), rowspan=3)
plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)
plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)
plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
plt.axis([-1.4, 1.4, -1.4, 1.4])
plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot2grid((3,2), (0, 1))
plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (1, 1))
plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticklabels([])
plt.axis([-2, 2, -1, 1])
plt.grid(True)

plt.subplot2grid((3,2), (2, 1))
plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
plt.gca().get_yaxis().set_ticks([])
plt.axis([-2, 2, -1, 1])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)
plt.show()

#-----------------------------------------------------------------
# PCA using SVD decomposition
# page 278
#-----------------------------------------------------------------
m, n = X.shape

X_centered = X - X.mean(axis=0)  # PCA??? ???????????? ????????? 0????????? ???????????? ??????
U, s, V = np.linalg.svd(X_centered)  # covariance matrix??? Decomposition?????? USVt 3??? amtrix???  ????????? ??????

c1 = V.T[:, 0]   # V amtrix??? eigenvextor??? ???????????? ?????? matrix. V??? Transpose?????? ????????? vector??? ??????
c2 = V.T[:, 1]   # V amtrix??? eigenvextor??? ???????????? ?????? matrix. V??? Transpose?????? ????????? vector??? ??????

S = np.zeros(X.shape)
S[:n, :n] = np.diag(s)

np.allclose(X, U.dot(S).dot(V))  # numpy??? np.allclose(a, b)??? ???????????? a??? b??? ?????? ????????? ????????? ????????? ??????
T = X.dot(V.T[:, :2])

#-----------------------------------------------------------------
# PCA using scikit-learn????????????
# page 280
#-----------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D_p = pca.fit_transform(X)
np.allclose(X2D_p, T)
X3D_recover = T.dot(V[:2, :])
np.allclose(X3D_recover, pca.inverse_transform(X2D_p))
# V
print('\nPCA Components = {}\n\n'.format(pca.components_))

R = pca.components_.T.dot(pca.components_)

print('\nS[:3] = {}\n\n'.format(S[:3]))

print('\npca.explained_variance_ratio_ = {}\n\n'.format(pca.explained_variance_ratio_))

print('\n1-pca.explained_variance_ratio_.sum() = {}\n\n'.format(1-pca.explained_variance_ratio_.sum()))

print('\nnp.sqrt((T[:, 1]**2).sum()) = {}\n\n'.format(np.sqrt((T[:, 1]**2).sum())))

#-----------------------------------------------------------------
#Choosing the right number of dimensions
# page 281
#-----------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
X = X_train

pca = PCA()
pca.fit(X)
d = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1  # variance_ratio_) >= 0.95????????? PC?????? ??????
print('\ndimension of principal component = {}\n\n'.format(d))

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print('\npca.n_components_ = {}\n\n'.format(pca.n_components_))

print('\nnp.sum(pca.explained_variance_ratio_) = {}\n\n'.format(np.sum(pca.explained_variance_ratio_)))

#-----------------------------------------------------------------
#PCA for Compression
# page 283
#-----------------------------------------------------------------
X_mnist = X_train

pca = PCA(n_components = 154)
X_mnist_reduced = pca.fit_transform(X_mnist)
X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)

def plot_digits(instances, images_per_row=5, **options):
     size = 28
     images_per_row = min(len(instances), images_per_row)
     images = [instance.reshape(size,size) for instance in instances]
     n_rows = (len(instances) - 1) // images_per_row + 1
     row_images = []
     n_empty = n_rows * images_per_row - len(instances)
     images.append(np.zeros((size, size * n_empty)))
     for row in range(n_rows):
         rimages = images[row * images_per_row : (row + 1) * images_per_row]
         row_images.append(np.concatenate(rimages, axis=1))
     image = np.concatenate(row_images, axis=0)
     plt.imshow(image, cmap = matplotlib.cm.binary, **options)
     plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_mnist[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_mnist_recovered[::2100])
plt.title("Compressed", fontsize=16)
plt.show()

#-----------------------------------------------------------------
# Incremental PCA
# page 283
#-----------------------------------------------------------------
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_mnist, n_batches):
     inc_pca.partial_fit(X_batch)

X_mnist_reduced = inc_pca.transform(X_mnist)

X_mnist_recovered_inc = inc_pca.inverse_transform(X_mnist_reduced)

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_mnist[::2100])
plt.subplot(122)
plot_digits(X_mnist_recovered_inc[::2100])
plt.tight_layout()
plt.show()

#---------------------------------------------------------------
# numpy??? memmap????????? ???????????? ????????????
#---------------------------------------------------------------
np.allclose(pca.mean_, inc_pca.mean_)

np.allclose(X_mnist_reduced, X_mnist_reduced)

filename = "my_mnist.data"

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_mnist.shape)
X_mm[:] = X_mnist
del X_mm

X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=X_mnist.shape)

batch_size = len(X_mnist) // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

#---------------------------------------------------------------
# random PCA and performance comparison
# page 284
#---------------------------------------------------------------
rnd_pca = PCA(n_components=154, random_state=42, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_mnist)

import time

for n_components in (2, 10, 154):
     print('\nn_components = {}\n\n'.format(n_components))
     regular_pca = PCA(n_components=n_components)
     inc_pca = IncrementalPCA(n_components=154, batch_size=500)
     rnd_pca = PCA(n_components=154, random_state=42, svd_solver="randomized")

     for pca in (regular_pca, inc_pca, rnd_pca):
         t1 = time.time()
         pca.fit(X_mnist)
         t2 = time.time()
         print('    {}: {} seconds\n'.format(pca.__class__.__name__, t2 - t1))


#---------------------------------------------------------------
# Kernel PCA
# Page 285
#---------------------------------------------------------------

from sklearn.decomposition import KernelPCA

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
     X_reduced = pca.fit_transform(X)
     if subplot == 132:
         X_reduced_rbf = X_reduced

     plt.subplot(subplot)
     #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
     #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
     plt.title(title, fontsize=14)
     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
     plt.xlabel("$z_1$", fontsize=18)
     if subplot == 131:
         plt.ylabel("$z_2$", fontsize=18, rotation=0)
     plt.grid(True)
plt.show()


#---------------------------------------------------------------
# Kernel ????????? ????????? ???????????? ??????
# Page 286
#---------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf = Pipeline([
         ("kpca", KernelPCA(n_components=2)),
         ("log_reg", LogisticRegression())
     ])

param_grid = [
         {"kpca__gamma": np.linspace(0.03, 0.05, 10), "kpca__kernel": ["rbf", "sigmoid"]}
     ]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print('\ngrid_search.best_params_ = {}\n\n'.format(grid_search.best_params_))

#---------------------------------------------------------------
# Kernel PCA??? ????????? ????????????
# Page 287
#---------------------------------------------------------------

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                     fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

plt.figure(figsize=(6, 5))

X_inverse = pca.inverse_transform(X_reduced)

ax = plt.subplot(111, projection='3d')
ax.view_init(10, -70)
ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()

X_reduced = rbf_pca.fit_transform(X)

plt.figure(figsize=(11, 4))
plt.subplot(132)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error
print('\nmean_squared_error(X, X_preimage) = {}\n\n'.format(mean_squared_error(X, X_preimage)))


#---------------------------------------------------------------
# Locally-Linear Embedding
# Page 289
#---------------------------------------------------------------

from sklearn.manifold import LocallyLinearEmbedding

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
X_reduced = lle.fit_transform(X)

plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
plt.show()
