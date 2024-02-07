import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data

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
    
    def sample_posterior(self, x: torch.Tensor, n_samples:int=1) -> torch.Tensor:
        """ Sample from the approximate posterior.

        Args:
        ----------
            x (torch.Tensor): A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
            n_samples (int): Number of samples to generate.

        Returns:
        ----------
            torch.Tensor: A tensor of dimension `(n_samples, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample(torch.Size([n_samples]))
        return z

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