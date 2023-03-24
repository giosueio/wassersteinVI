import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.distributions import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from sinkhorn import .
# from wvi_autoencoder import .

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
help='how many batches to wait before logging training status')
args, unknown = parser.parse_known_args()

torch.manual_seed(args.seed)
dataset_dimension = 28

device = torch.device("cpu")


train_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=True, download=True,
transform=transforms.Compose([
    transforms.Resize((dataset_dimension,dataset_dimension)),transforms.ToTensor(),
                   ])),batch_size=args.batch_size, shuffle=False)

if __name__ == "__main__":
    latent_dim = 10
    image_size = 784
    num_hidden = 100
    load_previous = False

    encoder = A_Encoder(latent_dim=latent_dim,image_size=image_size,hidden_dim=num_hidden).to(device)
    decoder = A_Decoder(latent_dim=latent_dim,image_size=image_size,hidden_dim=num_hidden).to(device)
    model = Autoencoder(encoder, decoder)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=20, verbose=True)
    previous_epoch = 0
    training_loss = []
    for epoch in range(1, args.epochs + 1):
        training_loss.append(train(epoch,model,image_size=image_size,latent_dim=latent_dim,prior='Gauss'))
        if epoch % 50 == 0:
            torch.save(model.state_dict(),'models/mnist_wvi_' + str(epoch+previous_epoch) + '_' + str(args.batch_size) + '_' + str(latent_dim) +  '.model')
            torch.save(encoder.state_dict(), 'models/mnist_wvi_encoder_' + str(epoch+previous_epoch) + '_' + str(args.batch_size)+ '_' + str(latent_dim)+ '.model')
            torch.save(decoder.state_dict(), 'models/mnist_wvi_decoder_'+ str(epoch+previous_epoch) + '_' + str(args.batch_size)+ '_' + str(latent_dim)+ '.model')
        scheduler.step(training_loss[-1])

    plt.plot(np.array(training_loss))
    plt.savefig('losses.PNG')
    plt.close()
