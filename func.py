
## Functions


from variables import *
import mms


def get_filtered_noise():

	# Generate random spatial noise

	# Random seed
	u = np.random.randn(Nx_dns)

	# Filtering
	k_max = np.pi / (dx*4) # dx_filter = 2 * dx_LES
	k = np.fft.fftfreq(Nx_dns) * 2*np.pi / dx_dns
	uk = np.fft.fft(u)
	for i in range(Nx_dns): # Fourier cutoff filter
		if abs(k[i]) > k_max:
			uk[i] = 0
	u = np.real(np.fft.ifft(uk)) # Inverse DFT
	u = u - np.average(u) # Remove DC component

	return u


def init_field_rand():

	# Randomly initialize fields

	rho = get_filtered_noise()*0.5 + 1
	u = get_filtered_noise()
	p = get_filtered_noise()*0.5 + 1

	field_init_dns[0,:] = rho # Density
	field_init_dns[1,:] = rho*u # Momentum
	field_init_dns[2,:] = p/(gamma-1) + 0.5*rho*u**2 # Energy

	field_init[:,:] = field_init_dns[:,::rx] # Downsampling


def get_exact_dns():

	# Solve DNS
	# Data saved at field_exact_dns, field_exact

	t = 0

	# Initial condition
	field_4 = field_init_dns.copy()
	# field_k4 = np.fft.fft(field_init_dns)

	print('Solving DNS...')

	# RK4 Loop
	for n in range(Nt_dns):

		print('Time: {0}'.format(t), end="\r")

		# RK4 substep 0 = 4 in previous timestep
		field_0 = field_4.copy()
		# field_k0 = field_k4.copy()

		# RK4 substep 0 -> 1
		k1 = navier_stokes_dns(field_0, t)
		field_1 = field_0 + 0.5*dt_dns*k1

		# RK4 substep 0 -> 2
		k2 = navier_stokes_dns(field_1, t+0.5*dt_dns)
		field_2 = field_0 + 0.5*dt_dns*k2

		# RK4 substep 0 -> 3
		k3 = navier_stokes_dns(field_2, t+0.5*dt_dns)
		field_3 = field_0 + dt_dns*k3

		# RK4 substep 0 -> 4
		k4 = navier_stokes_dns(field_3, t+dt_dns)
		field_4 = field_0 + (dt_dns/6)*(k1+2*(k2+k3)+k4)

		field_exact_dns[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# March in time
		t += dt_dns

	print()

	for n in range(Nt):
		# Downsampling
		field_exact[n,0,:,:] = field_exact_dns[int((n+0.5)*rt)-1,-1,:,::rx]
		field_exact[n,1,:,:] = field_exact_dns[int((n+0.5)*rt)-1,-1,:,::rx]
		field_exact[n,2,:,:] = field_exact_dns[int((n+1)*rt)-1,-1,:,::rx]
		field_exact[n,3,:,:] = field_exact_dns[int((n+1)*rt)-1,-1,:,::rx]


def user_force(field):

	# User-defined force

	# return 0
	# h = field.copy()
	# h[0,:] = ddx(field[0,:],dx)
	# h[1,:] = ddx(field[1,:],dx)
	# h[2,:] = ddx(field[2,:],dx)

	# return h*dx
	return field


def get_coarse_solution():

	# Solve ILES (LES w/ no model)
	# Data saved at field_iles

	t = 0

	# Initial condition
	field_4 = field_init.copy()

	print('Prediction')

	# RK4 Loop
	for n in range(Nt):

		print('Time: {0}'.format(t), end="\r")

		# RK4 substep 0 = 4 in previous timestep
		field_0 = field_4.copy()

		# RK4 substep 0 -> 1
		k1 = navier_stokes(field_0, t)
		field_1 = field_0 + 0.5*dt*k1

		# RK4 substep 0 -> 2
		k2 = navier_stokes(field_1, t+0.5*dt)
		field_2 = field_0 + 0.5*dt*k2

		# RK4 substep 0 -> 3
		k3 = navier_stokes(field_2, t+0.5*dt)
		field_3 = field_0 + dt*k3

		# RK4 substep 0 -> 4
		k4 = navier_stokes(field_3, t+dt)
		field_4 = field_0 + (dt/6)*(k1+2*(k2+k3)+k4)

		field_iles[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# March in time
		t += dt

	print()


def navier_stokes(field, t):

	# Returns RHS of Navier-Stokes equations

	p = eos(field) # Get pressure

	NaSt = np.zeros([dim, Nx]) # Return value: d/dt {rho, m, E}

	rho = field[0,:]
	m = field[1,:]
	E = field[2,:]
	u = m/rho
	e = E/rho

	NaSt[0,:] = -ddx(m,dx)
	NaSt[1,:] = -ddx(rho*u**2+p,dx) + 1/Re*d2dx2(u,dx)
	NaSt[2,:] = -ddx(u*(E+p),dx) + 1/Re*d2dx2(u**2/2,dx) \
				+ gamma/(gamma-1)/Re/Pr*d2dx2(p/rho,dx)

	return NaSt


def navier_stokes_dns(field, t):

	# Returns RHS of Navier-Stokes equations (for DNS only)

	p = eos(field)

	NaSt = np.zeros([dim, Nx_dns]) # Return value: d/dt {rho, m, E}

	rho = field[0,:]
	m = field[1,:]
	E = field[2,:]
	u = m/rho
	e = E/rho

	NaSt[0,:] = -ddx(m,dx_dns)
	NaSt[1,:] = -ddx(rho*u**2+p,dx_dns) + 1/Re*d2dx2(u,dx_dns)
	NaSt[2,:] = -ddx(u*(E+p),dx_dns) + 1/Re*d2dx2(u**2/2,dx_dns) \
				+ gamma/(gamma-1)/Re/Pr*d2dx2(p/rho,dx_dns)

	return NaSt


def eos(field):

	# Estimate and return pressure value given field variables

	rho = field[0,:]
	m = field[1,:]
	E = field[2,:]
	u = m/rho
	e = E/rho

	p = (e-u**2/2)*rho*(gamma-1)

	return p


def nn_force(field, net_list):

	# Forward propagate & Backpropagate NN
	# Return values:
	# (1) NN-predicted force (used in original PDE)
	# 		h: (dim, Nx) numpy array
	# (2) Gradient of force w.r.t. NN parameters (used when computing loss gradient)
	# 		h_grad_param: [[(Nx,weight.shape)*numLayers]*dim] list of list of numpy array
	# (3) Gradient of force w.r.t. field variables (used in adjoint PDE)
	# 		h_grad_field: (dim, dim, Nx, Nx) numpy array

	h = np.zeros([dim, Nx])
	h_grad_param = [[] for i in range(dim)]
	h_grad_field = np.zeros([dim, dim, Nx, Nx])

	for i in range(dim):

		for weight in net_list[i].weights:
			h_grad_param[i].append(np.zeros([Nx, weight.shape[0], weight.shape[1]]))

		numLayers = len(net_list[i].weights)

		for j in range(Nx):

			# Stencil of field variables as NN input
			index_stencil = (np.arange(numInputUnits//dim) \
							- (numInputUnits//dim-1)//2 + j) % Nx

			inputs = torch.from_numpy(field[:,index_stencil]).reshape(1,numInputUnits)

			# Turn on gradient tracking
			inputs.requires_grad = True
			for k in range(numLayers):
				net_list[i].weights[k].requries_grad = True

			h_tmp = net_list[i].forward(inputs) # Forward-propagate
			h_tmp.backward() # Backpropagate

			# Save data
			h[i,j] = h_tmp.detach().clone().numpy()

			for k in range(numLayers):
				h_grad_param[i][k][j,:,:] = net_list[i].weights[k].grad.detach().clone().numpy()

			h_grad_field[i,:,j,index_stencil] = inputs.grad.detach().clone().numpy().reshape(dim,numInputUnits//dim).T

			# Make gradients zero
			net_list[i].zero_grad()
			inputs.grad.data.zero_()

	

	return h, h_grad_param, h_grad_field


def nn_force_no_grad(field, net_list):

	# Forward propagate NN ONLY (No BP)
	# Return values:
	# (1) NN-predicted force (used in original PDE)
	# 		h: (dim, Nx) numpy array

	h = np.zeros([dim, Nx])

	for i in range(dim):

		numLayers = len(net_list[i].weights)

		for j in range(Nx):

			# Stencil of field variables as NN input
			index_stencil = (np.arange(numInputUnits//dim) \
							- (numInputUnits//dim-1)//2 + j) % Nx

			inputs = torch.from_numpy(field[:,index_stencil]).reshape(1,numInputUnits)

			# Turn off gradient tracking
			inputs.requires_grad = False
			for k in range(numLayers):
				net_list[i].weights[k].requires_grad = False

			with torch.no_grad():
				h_tmp = net_list[i].forward(inputs) # Forward-propagate

			# Save data
			h[i,j] = h_tmp.detach().clone().numpy()

	return h


def get_loss():

	# Calculate current loss (cost)

	beta = np.array([1/6, 1/3, 1/3, 1/6]) # RK4 quadrature weights

	loss = 0

	for k in range(4):
		loss += beta[k] * dx * dt * np.sum((field[:,k,:,:]-field_exact[:,k,:,:])**2)

	return loss


def adjoint(field, adj, t):

	# Returns RHS of adjoint PDE

	rho = field[0,:]
	m = field[1,:]
	E = field[2,:]
	u = m/rho
	e = E/rho
	p = eos(field)

	rho_adj = adj[0,:]
	m_adj = adj[1,:]
	E_adj = adj[2,:]

	R_rho = (gamma-3)/2*u**2*ddx(m_adj,dx) + ((gamma-1)/2*u**3 - u/rho*(E+p))*ddx(E_adj,dx) \
			- 1/Re*(u/rho)*d2dx2(m_adj,dx) - 1/Re*u**2/rho*d2dx2(E_adj,dx) \
			+ gamma/Re/Pr/rho**2*(0.5*rho*u**2-p/(gamma-1))*d2dx2(E_adj,dx)
	R_m = ddx(rho_adj,dx) - (gamma-3)*u*ddx(m_adj,dx) \
			+ ((E+p)/rho-(gamma-1)*u**2)*ddx(E_adj,dx) + 1/Re/rho*d2dx2(m_adj,dx) \
			+ 1/Re*u/rho*d2dx2(E_adj,dx) - gamma/Re/Pr*(u/rho)*d2dx2(E_adj,dx)
	R_E = (gamma-1)*ddx(m_adj,dx) + gamma*u*ddx(E_adj,dx) + gamma/Re/Pr/rho*d2dx2(E_adj,dx)

	R = np.vstack((R_rho,R_m,R_E))

	return R


def get_loss_gradient():

	# Calculate d(loss)/d(parameters) (Gradient of loss w.r.t. NN parameters)
	# Return value: loss_gradient (shape == net_list.shape)

	loss_gradient = [[] for i in range(dim)]

	for i in range(dim):
		for weight in net_list[i].weights:
			loss_gradient[i].append(np.zeros([weight.shape[0], weight.shape[1]]))

	beta = np.array([1/6, 1/3, 1/3, 1/6])

	for i in range(dim):
		for j in range(len(net_list[i].weights)):
			for k in range(4):
				loss_gradient[i][j] += beta[k] * dx * dt \
										* np.tensordot(adj[:,k,i,:], \
												h_grad_param[i][j][:,k,:,:,:], \
												axes=([0,1],[0,1]))

	return loss_gradient


def update_param(loss_gradient, alpha):

	# Simplest gradient descent

	for i in range(dim):
		for j in range(len(net_list[i].weights)):
			with torch.no_grad():
				net_list[i].weights[j].data -= alpha * loss_gradient[i][j]


def rmsprop(loss_gradient):

	# RMSprop (for stochastic gradient descent)

	for i in range(dim):
		for j in range(len(net_list[i].weights)):
			expect[i][j] = beta_rmsprop*expect[i][j] \
							+ (1-beta_rmsprop)*loss_gradient[i][j]**2
			with torch.no_grad():
				net_list[i].weights[j].data -= learningRate \
												/ (np.sqrt(expect[i][j])+epsilon) \
												* loss_gradient[i][j]
	

def save_nn():

	# Save NN

	for i in range(dim):
		torch.save(net_list[i], 'post/nnModel_{0}.pt'.format(i))


def ddx(u, dx):

	# 1st-order derivative in space


	# Periodic B.C.

	a = np.array([-1, 0., 1]) # CD2
	N = len(u)
	dudx = np.zeros(N)
	# dudx[0] = (-u[2]+4*u[1]-3*u[0])/2/dx # For Dirichlet B.C.
	# dudx[-1] = (3*u[-1]-4*u[-2]+u[-3])/2/dx

	for i in range(N):
	# for i in range(1,N-1): # Uncomment if Dirichlet
		u_stencil = np.roll(u, 1-i)[:3]
		dudx[i] = np.dot(a/2/dx, u_stencil)
	return dudx


def d2dx2(u, dx):

	# 2nd-order derivative in space


	# Periodic B.C.

	b = np.array([1, -2, 1]) # CD2
	N = len(u)
	d2udx2 = np.zeros(N)
	for i in range(N):
		u_stencil = np.roll(u, 1-i)[:3]
		d2udx2[i] = np.dot(b/dx**2, u_stencil)
	return d2udx2




## Special functions


# MMS (Method of Manufactured Solutions)

def init_field_mms():

	# Used for MMS only

	field_init[0,:] = mms.rho(x,0)
	field_init[1,:] = mms.rho(x,0) * mms.u(x,0)
	field_init[2,:] = mms.rho(x,0) * mms.e(x,0)

def get_exact_mms():

	# Used for MMS only

	k = np.array([0, 0.5, 0.5, 1]) # RK4 substep

	for n in range(Nt):

		t = n*dt

		for s in range(4):

			field_exact[n,s,0,:] = mms.rho(x,t+k[s]*dt)
			field_exact[n,s,1,:] = mms.rho(x,t+k[s]*dt) * mms.u(x,t+k[s]*dt)
			field_exact[n,s,2,:] = mms.rho(x,t+k[s]*dt) * mms.e(x,t+k[s]*dt)


# Spectral Methods

def conv_dealias(f, g):

	# Used for Spectral Methods only

	N = len(f) # No. of pts
	K = N * 3 // 2 # Padding (2/3 rule)

	f_pad = np.zeros(K, dtype='complex_') # Padded functions
	g_pad = np.zeros(K, dtype='complex_')

	ind_pad = np.append(np.arange(N//2), np.arange(N//2)+K-N//2) # Indices of original functions
	
	f_pad[ind_pad] = f
	g_pad[ind_pad] = g

	tmp = np.fft.fft(K * np.fft.ifft(f_pad) * np.fft.ifft(g_pad)) # Direct multiplication in x-space
	
	fg = tmp[ind_pad] / N # Normalize
	fg[N//2+1] = 0

	return fg

def div_dealias(f, g):

	# Used for Spectral methods only

	N = len(f) # No. of pts
	K = N * 3 // 2 # Padding (2/3 rule)

	f_pad = np.zeros(K, dtype='complex_') # Padded functions
	g_pad = np.zeros(K, dtype='complex_')

	ind_pad = np.append(np.arange(N//2), np.arange(N//2)+K-N//2) # Indices of original functions
	
	f_pad[ind_pad] = f
	g_pad[ind_pad] = g

	tmp = np.fft.fft(np.fft.ifft(f_pad) / np.fft.ifft(g_pad)) # Direct multiplication in x-space
	
	fg = tmp[ind_pad] # Normalize
	fg[N//2+1] = 0

	return fg

def navier_stokes_dns_spectral(field, t):

	# Spectral Method

	NaSt = np.zeros([dim, Nx_dns], dtype='complex_') # Return value: d/dt {rho, m, E}

	rho_k = field[0,:]
	m_k = field[1,:]
	E_k = field[2,:]

	# rho_inv_k = inv_dealias(rho_k)
	u_k = div_dealias(m_k,rho_k)
	p_k = (gamma-1)*(E_k + 0.5*conv_dealias(m_k,u_k))

	Uk_rho = -1j*k_dns*m_k
	Uk_m = -1j*k_dns*(conv_dealias(m_k,u_k)+p_k) - 1/Re*k_dns**2*u_k
	Uk_E = -1j*k_dns*conv_dealias(u_k,E_k+p_k) - 1/Re*k_dns**2*conv_dealias(u_k,u_k)/2 \
			- gamma/(gamma-1)/Re/Pr*k_dns**2*div_dealias(m_k,rho_k)

	NaSt[0,:] = Uk_rho
	NaSt[1,:] = Uk_m
	NaSt[2,:] = Uk_E

	return NaSt


