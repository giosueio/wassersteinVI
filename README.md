# Wasserstein variational inference
Unofficial implementation of the paper [Wasserstein Variational Inference](https://papers.nips.cc/paper_files/paper/2018/hash/2c89109d42178de8a367c0228f169bf8-Abstract.html) by Ambrogioni et al. (2018).

The implementation follows a different neural architecture from that of the original paper. The original considers the following architectures: for the encoder, a multilayer perceptron with three fully connected layers (100-300-500-1568) and ReLu nonlinearities in the hidden layers, and for the decoder, a three-layered ReLu networks (784-500-300-100). In our implementation, the encoder was parametrized by three convolutional layers of increasing size (24-384-1536) with leaky ReLu activation functions (0.2), followed by two perceptron layers (288-128) with the same activation functions. The decoder is instead parametrized by the same architecture in reverse. Since the dataset where the model was applied is MNIST, the dimensionality of the latent space was set to be $d=10$, hoping that the mapping $q_\phi(z|x)$ would learn the actual labels in an unsupervised way. Neither this nor other hyperparameters were tuned, although we strongly encourage practitioners to do so. The stochastic gradient-based optimizer we used is Adam (Kingma and Ba, 2014), with a learning rate $\epsilon = 0.001$ and weight decay set to 0.00005.

The VAE was trained for 10 epochs, considering an extremely small minibatch of 20 samples per time in order to speed up the lengthy computations of Sinkhorn divergences. As per the Sinkhorn divergence algorithm, the implementation was translated to the log domain in order to avoid perpetually occurring numerical errors. The metric used in all the components of $C(x_1,z_1;x_2,z_2)$ was the L2 distance. For every call to the Sinkhorn algorithm, $L=20$ iterations were performed and the regularization term was chosen to be $\epsilon = 0.01$. 

![image](https://user-images.githubusercontent.com/77994290/227616561-24d7116c-c273-4be6-b000-c2b9d0627427.png)


## Credits:
- general structure: https://github.com/zqkhan/WVI_pytorch
- neural architecture and plots: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
