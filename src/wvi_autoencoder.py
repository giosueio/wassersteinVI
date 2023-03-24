class A_Encoder(nn.Module):
    def __init__(self,image_size=784,latent_dim=10,hidden_dim=500):
        super(A_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.a_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2*latent_dim))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self,x):
        q = self.a_encoder(x)
        mu = q[:,:self.latent_dim]
        log_var = q[:,self.latent_dim:]
        z = self.reparameterize(mu,log_var)
        return mu,log_var,z

class A_Decoder(nn.Module):
    def __init__(self,image_size=784,latent_dim=10,hidden_dim=500):
        super(A_Decoder, self).__init__()
        self.a_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.a_decoder(z)

class Autoencoder(nn.Module):
    def __init__(self,a_encoder,a_decoder):
        super(Autoencoder, self).__init__()

        self.a_encoder = a_encoder
        self.a_decoder = a_decoder

    def forward(self, input):
        _,_,z = self.a_encoder(input)
        x = self.a_decoder(z)
        return z,x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train(epoch,model,image_size,latent_dim,prior='Gauss'):
    model.train()
    train_loss = list(range(len(train_loader)))
    train_model_kernel = 0
    train_encoder_kernel = 0
    train_cross_kernel = 0
    recon_loss = list(range(len(train_loader)))
    sinkhorn_solver = SinkhornSolver(epsilon=0.01,iterations=20)

    start_time = time.time()
    for batch_idx, (real_data, _) in enumerate(train_loader):
        real_data = real_data.to(device)
        real_data = real_data.type(torch.float32)
        optimizer.zero_grad()
        if prior == 'Gauss':
            latent_priors = multivariate_normal.MultivariateNormal(loc=torch.zeros(latent_dim),
                                                                   covariance_matrix=torch.eye(latent_dim)).\
                sample(sample_shape=(real_data.size()[0],)).to(device)
        else:
            latent_priors = Variable(
                -2*torch.rand(real_data.size()[0], latent_dim) + 1,
                requires_grad=False
            ).to(device)

        mu, logvar, latent_encoded = model.a_encoder(real_data) 
        decoded_data = model.a_decoder(latent_priors) # decoded data is x1
        latent_decoded_data,_,_ = model.a_encoder(decoded_data.unsqueeze(1).reshape((args.batch_size,1,28,28))) 
        reconstructed_data = model.a_decoder(latent_encoded)
        observable_error = ((real_data - reconstructed_data).pow(2).mean(-1)).mean()
        C1 = compute_cost(decoded_data.view(-1,image_size),real_data.view(-1,image_size))
        C2 = compute_cost(decoded_data.view(-1,image_size),reconstructed_data.view(-1,image_size))
        C3 = compute_cost(latent_priors - latent_decoded_data, latent_encoded - mu)
        C4 = compute_cost(torch.zeros_like(reconstructed_data.view(-1,image_size)),
                          real_data.view(-1,image_size) - reconstructed_data.view(-1,image_size))
        loss,_ = sinkhorn_solver(decoded_data.view(-1,image_size),real_data.view(-1,image_size),C=C1+C2+C3+C4)
        loss.backward()
        train_loss[batch_idx] = loss.item()
        recon_loss[batch_idx] = observable_error.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(real_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))
    end_time = time.time()
    end_time = end_time - start_time
    print('====> Epoch: {} Average training loss: {:.4f}, Model Kernel: {:.6f},Encoder Kernel: {:.6f}, Cross Kernel: {:.6f}, Observable Error: {:.6f} Time: {:.6f}'.format(
        epoch, np.array(train_loss).mean(0),train_model_kernel/batch_idx, 
        train_encoder_kernel/batch_idx,train_cross_kernel/batch_idx,np.array(recon_loss).mean(0),end_time))
    plt.imshow(reconstructed_data[0].detach().squeeze(), cmap='gist_gray')
    plt.show()
    return np.array(train_loss).mean(0)
