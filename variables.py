## Variables (Physical / NN)


import torch
from torch import nn
import numpy as np


# Physics settings

dim = 3 # Dimension of field (3: rho, m, E)
L = 2*np.pi
Re = 100
Pr = 1
gamma = 1.4


# Numerics settings

Nx = 32
dx = L/Nx
t_final = 1
rx = 8 # Ratio of No. of grids in DNS / LES
rt = rx

# Timestep
dt = min(dx, dx**2*Re)*0.5
Nt = int(t_final/dt)
dt = t_final/Nt

Nx_dns = Nx*rx
dx_dns = dx/rx
Nt_dns = Nt*rt
dt_dns = t_final/Nt_dns

k_dns = np.fft.fftfreq(Nx_dns) * 2*np.pi / dx_dns # Wavenumber


# Fields

x = np.linspace(0, L, Nx, endpoint=False)
x_dns = np.linspace(0, L, Nx_dns, endpoint=False)

field_init_dns = np.zeros([dim, Nx_dns]) # Initial condition (DNS)
field_init = np.zeros([dim, Nx]) # Initial condition (downsampled)

field = np.zeros([Nt, 4, dim, Nx]) # LES data
field_exact_dns = np.zeros([Nt_dns, 4, dim, Nx_dns]) # Exact data (DNS)
field_exact = np.zeros([Nt, 4, dim, Nx]) # Exact data (downsampled)
field_iles = np.zeros([Nt, 4, dim, Nx]) # ILES data (LES w/ no model)

adj = np.zeros([Nt, 4, dim, Nx]) # Adjoints


# Neural network settings
torch.set_default_dtype(torch.float64) # Set torch precision

# Default NN settings
numInputUnits = 5*dim # Input: 5-point stencil for all 3 physical variables
numHiddenLayers = 3
numHiddenUnits = 100

numIter = 500 # No. of iteration (== No. of training set)
learningRate = 0.001 # Learning rate
beta_rmsprop = 0.9 # Moving average parameter (RMSprop)
epsilon = 1e-8 # Prevents division by zero

loss_list = np.zeros(numIter) # List of loss


# NN class
class Model(nn.Module):
	def __init__(self):

		super().__init__()

		self.numInputUnits = numInputUnits # No. of units in input layer
		self.numHiddenLayers = numHiddenLayers # No. of hidden layers
		self.numHiddenUnits = numHiddenUnits # No. of units in hidden layer
		# Output layer has only 1 unit

		# Weights

		self.weights = [] # List of weights between each layers

		for i in range(numHiddenLayers+1):

			if numHiddenLayers == 0: # No hidden layer
				numPreUnits = self.numInputUnits
				numNextUnits = 1

			elif i == 0: # Parameters for Input -> Hidden
				numPreUnits = self.numInputUnits
				numNextUnits = self.numHiddenUnits

			elif i == numHiddenLayers: # Parameters for Hidden -> Output
				numPreUnits = self.numHiddenUnits
				numNextUnits = 1

			else:
				numPreUnits = self.numHiddenUnits
				numNextUnits = self.numHiddenUnits

			# Xavier initialization
			self.weights.append(nn.Parameter(torch.randn(numPreUnits,\
							numNextUnits)/np.sqrt(numPreUnits)/2))


	def forward(self, inputs):
		# Forward propagation
		a = inputs
		for i in range(len(self.weights)-1):
			z = a.mm(self.weights[i])
			a = torch.tanh(z)
			# a = z
		output = a.mm(self.weights[-1])
		return output


	def zero_grad(self):
		# Zeroes gradients
		for weight in self.weights:
			weight.grad.data.zero_()


# Create NNs
net_list = [Model() for i in range(dim)]
# net_list[0] : net_rho
# net_list[1] : net_m
# net_list[2] : net_E


# Define gradient arrays
# - h_grad_param[dim][numLayers][Nt,4,Nx,weight.shape]
# - h_grad_field[Nt,4,dim,Nx,Nx]

h_grad_param = [[] for i in range(dim)]
expect = [[] for i in range(dim)] # Expectations (RMSprop)

for i in range(dim):
	for weight in net_list[i].weights:
		h_grad_param[i].append(np.zeros([Nt, 4, Nx, weight.shape[0], weight.shape[1]]))

for i in range(dim):
		for weight in net_list[i].weights:
			expect[i].append(np.zeros([weight.shape[0], weight.shape[1]]))

h_grad_field = np.zeros([Nt, 4, dim, dim, Nx, Nx])



