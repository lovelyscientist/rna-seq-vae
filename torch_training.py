import os
import numpy as np
import time
import torch
import argparse
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from torch_model import VAE
from gtex_loader import get_gtex_dataset


def main(args):
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = time.time()

    (X_train, Y_train), (X_test, Y_test), scaled_df_values, gene_names, Y = get_gtex_dataset()

    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    train_targets = le.transform(Y_train)
    test_targets = le.transform(Y_test)

    train_target = torch.as_tensor(train_targets)
    train = torch.tensor(X_train.astype(np.float32))
    train_tensor = TensorDataset(train, train_target)
    data_loader = DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)

    test_target = torch.as_tensor(test_targets)
    test = torch.tensor(X_test.astype(np.float32))
    test_tensor = TensorDataset(test, test_target)
    test_loader = DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=True)


    def loss_fn(recon_x, x, mean, log_var):
        view_size = 1000
        ENTROPY = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, view_size), x.view(-1, view_size), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (ENTROPY + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 6).long().unsqueeze(1)
                    x = vae.inference(n=c.size(0), c=c)
                else:
                    x = vae.inference(n=6)


    with torch.no_grad():
        for epoch in range(args.epochs):
            test_loss = 0
            for iteration, (x, y) in enumerate(test_loader):
                recon_x, mean, log_var, z = vae(x, y)
                test_loss += loss_fn(recon_x, x, mean, log_var)

            print('====> Test set loss: {:.4f}'.format(test_loss))

    '''with torch.no_grad():
        sample = torch.randn(64, args.latent_size).to(device)
        sample = model.decode(sample).cpu()'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1000, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 1000])
    parser.add_argument("--latent_size", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)