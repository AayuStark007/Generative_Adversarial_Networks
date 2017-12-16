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
lr = 0.0002
m_epoch = 5

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

def relu(x, deriv=False):
	if deriv:
		x[x > 0] = 1
		return x	
	return np.maximum(x, 0, x)

def BCELoss(x, y):
	dx = 1e-12

	loss_v = (y*np.log(x)) + ((1-y)*np.log(1-x))
	return np.abs(loss_v)

class Generator():

	def __init__(self):
		self.W0 = np.random.randn(Z_dim, h_dim).astype(np.float32) * np.sqrt(2.0/(Z_dim))
		self.b0 = np.zeros([h_dim]).astype(np.float32)
		self.W1 = np.random.randn(h_dim, X_dim).astype(np.float32) * np.sqrt(2.0/(h_dim))
		self.b1 = np.zeros([X_dim]).astype(np.float32)

		self.l0 = np.ndarray(shape=(mb_size, Z_dim), dtype=np.float32)
		self.l1 = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)
		self.l2 = np.ndarray(shape=(mb_size, X_dim), dtype=np.float32)

		self.l2_delta = np.ndarray(shape=(mb_size, X_dim), dtype=np.float32)
		self.l1_error = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)
		self.l1_delta = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)

	def forward(self, x):
		self.l0 = x
		self.l1 = relu(np.dot(self.l0, self.W0) + self.b0)
		self.l2 = sigmoid(self.l1.dot(self.W1) + self.b1)

		return self.l2

	def backward(self, l2_error):
		self.l2_delta += np.multiply(l2_error, sigmoid(self.l2, deriv=True))
		self.l1_error += self.l2_delta.dot(self.W1.T)
		self.l1_delta += np.multiply(self.l1_error, relu(self.l1, deriv=True))

	def step(self):
		self.W1 += -1.0 * lr * self.l1.T.dot(self.l2_delta)
		self.W0 += -1.0 * lr * self.l0.T.dot(self.l1_delta) 
		self.reset_grad()

	def reset_grad(self):
		self.l2_delta.fill(0.0)
		self.l1_error.fill(0.0)
		self.l1_delta.fill(0.0)

class Discriminator():

	def __init__(self):
		self.W0 = np.random.randn(X_dim, h_dim).astype(np.float32) * np.sqrt(2.0/(X_dim))
		self.b0 = np.zeros([h_dim]).astype(np.float32)
		self.W1 = np.random.randn(h_dim, 1).astype(np.float32) * np.sqrt(2.0/(h_dim))
		self.b1 = np.zeros([1]).astype(np.float32)

		self.l0 = np.ndarray(shape=(mb_size, X_dim), dtype=np.float32)
		self.l1 = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)
		self.l2 = np.ndarray(shape=(mb_size, 1), dtype=np.float32)

		self.l2_delta = np.ndarray(shape=(mb_size, 1), dtype=np.float32)
		self.l1_error = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)
		self.l1_delta = np.ndarray(shape=(mb_size, h_dim), dtype=np.float32)

	def forward(self, x):
		self.l0 = x
		self.l1 = relu(self.l0.dot(self.W0) + self.b0)
		self.l2 = sigmoid(self.l1.dot(self.W1) + self.b1)

		return self.l2

	def backward(self, l2_error):
		self.l2_delta += np.multiply(l2_error, sigmoid(self.l2, deriv=True))
		self.l1_error += self.l2_delta.dot(self.W1.T)
		self.l1_delta += np.multiply(self.l1_error, relu(self.l1, deriv=True))

	def step(self):
		self.W1 += -1.0 * lr * self.l1.T.dot(self.l2_delta)
		self.W0 += -1.0 * lr * self.l0.T.dot(self.l1_delta) 
		self.reset_grad()

	def reset_grad(self):
		self.l2_delta.fill(0.0)
		self.l1_error.fill(0.0)
		self.l1_delta.fill(0.0)

ones_label = np.ones((mb_size, 1))
zeros_label = np.zeros((mb_size, 1))

G = Generator()
D = Discriminator()

G.reset_grad()
D.reset_grad()

for epoch in range(m_epoch):
	for i, (data, _) in enumerate(dataloader):

		# train discriminator
		z = np.random.random((mb_size, Z_dim))
		X = data.numpy()
		
		X = np.resize(X, (mb_size, X_dim))

		G_sample = G.forward(z)
		D_real = D.forward(X)
		D_fake = D.forward(G_sample)

		D_loss_real = BCELoss(D_real, ones_label)
		D_loss_fake = BCELoss(D_fake, zeros_label)
		D_loss = D_loss_real + D_loss_fake

		#print('backprop D')

		# backprop D
		#######
		D.backward(D_loss_real)
		D.backward(D_loss_fake)
		D.step()

		# train generator
		z = np.random.random((mb_size, Z_dim))
		G_sample = G.forward(z)
		D_fake = D.forward(G_sample)

		G_loss = BCELoss(D_fake, ones_label)

		#print('backprop G')

		# backprop G
		########
		G.backward(G_loss)
		G.step()

		# Print and plot every now and then
		if i % 1000 == 0:
			print('Epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, np.mean(D_loss), np.mean(G_loss)))
			#dis_loss.append(D_loss)
			#gen_loss.append(G_loss)

			samples = G.forward(z)[:16]

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
