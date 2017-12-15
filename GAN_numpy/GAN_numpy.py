import numpy as np
from torchvision import datasets, transforms

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../MNIST_data', train=True, download=True,
    	transform=transforms.Compose([
    		transforms.Scale(28),
    		transforms.ToTensor()
    	])),
    batch_size=mb_size, shuffle=True
)

def sigmoid(x, deriv=False):
	if deriv:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(x, 0, x)

mb_size = 64
Z_dim = 100
X_dim = 28*28
h_dim = 128
lr = 1e-3


# define Generator
G_W0 = np.random.random((Z_dim, h_dim)) - 1
G_W1 = np.random.random((h_dim, X_dim)) - 1

def generator_fwd(x):
	l0 = x
	l1 = relu(l0.dot(G_W0))
	l2 = sigmoid(l1.dot(G_W1))
	return l2

# define Discriminator
D_W0 = np.random.random((X_dim, h_dim)) - 1
D_W1 = np.random.random((h_dim, 1)) - 1

def discriminator_fwd(x):
	l0 = x
	l1 = relu(l0.dot(D_W0))
	l2 = sigmoid(l1.dot(D_W1))
	return l2
