import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def embed_visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def loss_visualize(losses):
    losses_float = [float(loss) for loss in losses] 
    loss_indices = [i for i,l in enumerate(losses_float)] 
    plt = sns.lineplot(loss_indices, losses_float);