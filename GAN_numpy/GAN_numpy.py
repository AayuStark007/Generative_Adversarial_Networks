import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import torch
from torchvision import datasets, transforms


mb_size = 64
Z_dim = 100
X_dim = 28*28
h_dim = 128
lr = 1e-3
m_epoch = 1

try:
	os.makedirs('out-np-gan')
except:
	pass

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../MNIST_data', train=True, download=True,
    	transform=transforms.Compose([
    		transforms.Scale(28),
    		transforms.ToTensor()
    	])),
    batch_size=mb_size, shuffle=True
)

#### UTILS ####
def sigmoid(x, deriv=False):
	if deriv:
		return x * (1 - x)
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(x, 0, x)

def BCELoss(x, y):
	dx = 1e-12
	return -np.mean((x*np.log(y+dx)) + ((1-x) * np.log(1-y+dx))) 




# define Generator
G_W0 = np.random.randn(Z_dim, h_dim).astype(np.float32) * np.sqrt(2.0/(Z_dim))
G_b0 = np.zeros([h_dim]).astype(np.float32)
G_W1 = np.random.randn(h_dim, X_dim).astype(np.float32) * np.sqrt(2.0/(h_dim))
G_b1 = np.zeros([X_dim]).astype(np.float32)

def generator_fwd(x):
	l0 = x
	l1 = relu(l0.dot(G_W0) + G_b0)
	l2 = sigmoid(l1.dot(G_W1) + G_b1)
	return l2

# define Discriminator
D_W0 = np.random.randn(X_dim, h_dim).astype(np.float32) * np.sqrt(2.0/(X_dim))
D_b0 = np.zeros([h_dim]).astype(np.float32)
D_W1 = np.random.randn(h_dim, 1).astype(np.float32) * np.sqrt(2.0/(h_dim))
D_b1 = np.zeros([1]).astype(np.float32)

def discriminator_fwd(x):
	l0 = x
	l1 = relu(l0.dot(D_W0) + D_b0)
	l2 = sigmoid(l1.dot(D_W1) + D_b1)
	return l2

ones_label = np.ones((mb_size, 1))
zeros_label = np.zeros((mb_size, 1))

for epoch in range(m_epoch):
	for i, (data, _) in enumerate(dataloader):

		# train discriminator
		z = np.random.random((mb_size, Z_dim))
		X = data.numpy()
		
		X = np.resize(X, (mb_size, X_dim))

		G_sample = generator_fwd(z)
		D_real = discriminator_fwd(X)
		D_fake = discriminator_fwd(G_sample)

		D_loss_real = BCELoss(D_real, ones_label)
		D_loss_fake = BCELoss(D_fake, zeros_label)
		D_loss = D_loss_real + D_loss_fake

		# backprop D
		#######

		# train generator
		z = np.random.random((mb_size, Z_dim))
		G_sample = generator_fwd(z)
		D_fake = discriminator_fwd(G_sample)

		G_loss = BCELoss(D_fake, ones_label)

		# backprop G
		########

		# Print and plot every now and then
		if i % 1000 == 0:
			print('Epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss, G_loss))
			#dis_loss.append(D_loss)
			#gen_loss.append(G_loss)

			samples = generator_fwd(z)[:16]

			fig = plt.figure(figsize=(4, 4))
			gs = gridspec.GridSpec(4, 4)
			gs.update(wspace=0.05, hspace=0.05)

			for i, sample in enumerate(samples):
				ax = plt.subplot(gs[i])
				plt.axis('off')
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('equal')
				plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

			plt.savefig('{}/epoch-{}.png'.format('out-np-gan', str(epoch).zfill(3)), bbox_inches='tight')
			plt.close(fig)

            #plot_loss(gen_loss, dis_loss)
