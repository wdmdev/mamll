import torch
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_posterior(samples: Tuple[torch.Tensor, torch.Tensor], save_plot: str) -> None:
    """Plot samples from the approximate posterior and colour them by their correct class label. 
    
    Args:
    -----
        model (VAE): The trained VAE model.
        data_loader (torch.utils.data.DataLoader): Data loader for the dataset.
    """
    for sample, y in samples: 
        #Do PCA if the latent space is greater than 2
        if sample.shape[1] > 2:
            pca = PCA(n_components=2)
            sample = pca.fit_transform(sample.detach().cpu().numpy())
        
        plt.scatter(sample[:, 0], sample[:, 1], c=y, cmap='tab10')
        plt.colorbar()

        #save the plot
        plt.savefig(save_plot, dpi=300)