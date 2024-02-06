# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm


class GaussianPrior(nn.Module):
    """Gaussian prior distribution."""
    def __init__(self, M:int) -> None:
        """ Define a Gaussian prior distribution with zero mean and unit variance.

        Args:
        ----------
            M (int): Dimension of the latent space (number of latent variables).
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self) -> torch.distributions.Distribution:
        """ Return the prior distribution.

        Returns:
        ----------
            torch.distributions.Distribution: The prior distribution.

        """
        prior = td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
        return prior

class MixtureGaussianPrior(nn.Module):
    """Mixture Gaussian prior distribution."""
    def __init__(self, M:int, K:int) -> None:
        """ Define a mixture Gaussian prior distribution with K components.

        Args:
        ----------
            M (int): Dimension of the latent space (number of latent variables).
            K (int): Number of components in the mixture.
        """
        super(MixtureGaussianPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.zeros(self.M, self.K), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M, self.K), requires_grad=False)

    def forward(self) -> torch.distributions.Distribution:
        """ Return the prior distribution.

        Returns:
        ----------
            torch.distributions.Distribution: The prior distribution.
        """
        prior = td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
        return prior


class GaussianEncoder(nn.Module):
    """Gaussian encoder distribution."""
    def __init__(self, encoder_net: torch.nn.Module) -> None:
        """ Define a Gaussian encoder distribution based on a given encoder network.

        Args:
        ----------
            encoder_net (torch.nn.Module): The encoder network that takes as a tensor of dim 
                                            `(batch_size, feature_dim1, feature_dim2)` and output 
                                            a tensor of dimension `(batch_size, 2M)`, 
                                            where M is the dimension of the latent space.
        """

        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """ Given a batch of data, return a Gaussian distribution over the latent space.

        Args:
        ----------
            x (torch.Tensor): A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`

        Returns:
        ----------
            torch.distributions.Distribution: The Gaussian distribution over the latent space.
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    """Bernoulli decoder distribution."""
    def __init__(self, decoder_net: torch.nn.Module) -> None:
        """ Define a Bernoulli decoder distribution based on a given decoder network.

        Args:
        ----------
            decoder_net (torch.nn.Module): A decoder network that transforms an input tensor of 
                                             shape `(batch_size, M)` to an output tensor of shape 
                                             `(batch_size, feature_dim1, feature_dim2)`. M is the 
                                             latent space dimension.
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z: torch.Tensor) -> torch.distributions.Distribution:
        """ Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Args:
        ----------
            z (torch.Tensor): A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.

        Returns:
        ----------
            torch.distributions.Distribution: The Bernoulli distribution over the data space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """Variational Autoencoder (VAE) model."""
    def __init__(self, prior: torch.nn.Module, decoder: torch.nn.Module, encoder: torch.nn.Module) -> None:
        """ VAE model.

        Args:
        ----------
            prior (torch.nn.Module): The prior distribution over the latent space.
            decoder (torch.nn.Module): The decoder distribution over the data space.
            encoder (torch.nn.Module): The encoder distribution over the latent space.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the ELBO for the given batch of data.

        Args:
        ----------
            x (torch.Tensor): A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`

        Returns:
        ----------
            torch.Tensor: The ELBO for the given batch of data.
        """
        q = self.encoder(x)
        z = q.rsample() #reparameterization trick, under the hood
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples:int=1) -> torch.Tensor:
        """ Sample from the model.

        Args:
        ----------
            n_samples (int): Number of samples to generate.

        Returns:
        ----------
            torch.Tensor: A tensor of dimension `(n_samples, feature_dim1, feature_dim2, ...)`
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Compute the negative ELBO for the given batch of data.

        Args:
        ----------
            x (torch.Tensor): A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`

        Returns:
        ----------
            torch.Tensor: The negative ELBO for the given batch of data.
        """
        return -self.elbo(x)


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
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

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
    # prior = MixtureGaussianPrior(M, 10) # Mixture of 10 Gaussians

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
