import argparse
from tqdm import tqdm 

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from mamll.dgm.models.flow import GaussianBase, RandomMask, ChequerboardMask, Flow

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x, y in data_iter:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "sample"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/flow.pt",
        help="file to save model to or load model from (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples/flow.png",
        help="file to save samples in (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="V",
        help="learning rate for training (default: %(default)s)",
    )
    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    train_dataset = datasets.MNIST('data/dequantized/', train = True, download = True,
                    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x + torch.rand(x.shape)/254),
                    transforms.Lambda(lambda x: x.flatten())
                    ]))

    train_loader = torch.utils.data.DataLoader( #type: ignore
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Define prior distribution
    D = next(iter(train_loader))[0].shape[1]
    base = GaussianBase(D)

    # Define transformations
    transformations = []
    num_transformations = 5
    num_hidden = 8

    for i in range(num_transformations):
        scale_net = nn.Sequential(
            nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D)
            , nn.Tanh() #added for stability
        )
        translation_net = nn.Sequential(
            nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D)
        )
        transformations.append(RandomMask(scale_net, translation_net, D))

    # Define flow model
    model = Flow(base, transformations).to(args.device)
    

    if args.mode == "train":
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == "sample":
        import matplotlib.pyplot as plt

        # Load the trained model
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = model.sample((10000,)).cpu()

        # Reshape the samples to 28x28 (the size of MNIST images)
        samples = samples.view(-1, 28, 28)

        # Plot the samples
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axs.flat):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')

        plt.savefig(args.samples)
        plt.close()
