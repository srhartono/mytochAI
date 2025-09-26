import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# Graphing function to plot data and principal components and save to PNG
def plot_pca(X, pca, filename='pca_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], label='Data Points')
    mean = np.mean(X, axis=0)
    for length, vector in zip(pca.singular_values_, pca.components_):
        v = vector * length / 2  # scale for visualization
        plt.arrow(mean[0], mean[1], v[0], v[1], color='r', width=0.05, head_width=0.2, label='Principal Component')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('PCA of Data')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as {filename}")

# Save the plot as a PNG file
plot_pca(X, pca)

