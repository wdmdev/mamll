# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from mamll.plot.vae import plot_posterior
from mamll.dgm.models.vae import VAE, GaussianPrior, BernoulliDecoder, GaussianEncoder

def get_next_batch(data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """ Get the next batch of data from a data loader.
    
    Args:
    ----------
        data_loader (torch.utils.data.DataLoader): The data loader to get the next batch from.
        
    Returns:
    ----------
        torch.Tensor: The next batch of data.
    """
    return next(iter(data_loader))[0]

def train(model: VAE, optimizer: torch.optim.Optimizer, 
          data_loader: torch.utils.data.DataLoader, epochs: int, device: torch.device) -> None:
    """ Train a VAE model.

    Args:
    ----------
        model (VAE): The VAE model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        data_loader (torch.utils.data.DataLoader): The data loader to use for training.
        epochs (int): Number of epochs to train for.
        device (torch.device): The device to use for training.
    """
    model.train()
    num_steps = len(data_loader) * epochs

    with tqdm(range(num_steps), desc="Training progress") as pbar:
        for step in pbar:
            batch = get_next_batch(data_loader)
            batch = batch.to(device)

            optimizer.zero_grad()

            loss = model(batch)
            loss.backward()

            optimizer.step()

            # Update progress bar every 5 steps
            if step % 5 == 0: 
                loss_value = loss.detach().cpu()
                pbar.set_description(f"Epoch={step // len(data_loader)}, Step={step}, Loss={loss_value:.1f}")

def test(model: VAE, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """ Test a VAE model.

    Args:
    ----------
        model (VAE): The VAE model to test.
        data_loader (torch.utils.data.DataLoader): The data loader to use for testing.
        device (torch.device): The device to use for testing.

    Returns:
    ----------
        float: The average negative ELBO over the test set.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        with tqdm(range(len(data_loader)), desc="Test progress") as pbar:
            for step in pbar:
                batch = get_next_batch(data_loader)
                batch = batch.to(device)

                test_loss += model(batch).detach().cpu()

                # Update progress bar every 5 steps
                if step % 5 == 0: 
                    pbar.set_description(f"Epoch={step // len(data_loader)}, Step={step}, Loss={test_loss:.1f}")

    return test_loss / len(data_loader)


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'posterior', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--sample_posterior', type=str, default='posterior_samples.png', help='file to save posterior sample plot in (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)
    
    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Test model
        test_loss = test(model, mnist_test_loader, args.device)
        print(f"Test set average negative ELBO: {test_loss:.1f}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'posterior':
        model.eval()
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        data_loader = mnist_test_loader

        with torch.no_grad():
            x, y = next(iter(data_loader))
            x = x.to(device)
            y = y.to(device)
            z = model.sample_posterior(x)

            plt.figure(figsize=(10, 10))
            for sample in z:
                if sample.shape[1] > 2:
                    pca = PCA(n_components=2)
                    pca_sample = pca.fit_transform(sample.detach().cpu().numpy())

                    plt.scatter(sample[:, 0], sample[:, 1], c=y, cmap='tab10')
                    plt.colorbar()
                    plt.title(f'{sample.shape[0]} samples from the approximate posterior. {z.shape[2]} dimensions reduced to 2 using PCA')

            plt.savefig(args.sample_posterior, dpi=300)

