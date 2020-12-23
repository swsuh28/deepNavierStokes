## Grid-testing module

import numpy as np
import matplotlib.pyplot as plt


def get_filtered_noise():

	# Random seed
	u = np.random.randn(Nx_max)
	dx_dns = L/Nx_max

	# Filtering
	k_max = np.pi / (dx_bl*2) # dx_filter = 4 * dx_LES
	k = np.fft.fftfreq(Nx_max) * 2*np.pi / dx_dns
	uk = np.fft.fft(u)
	for i in range(Nx_max): # Fourier cutoff filter
		if abs(k[i]) > k_max:
			uk[i] = 0
	u = np.real(np.fft.ifft(uk)) # Inverse DFT
	u = u - np.average(u) # Remove DC component

	return u

def init_field_rand():

	rx_max = max(rx_list)

	rho = get_filtered_noise()*0.5 + 1
	u = get_filtered_noise()
	p = get_filtered_noise()*0.5 + 1

	field_init_finest[0,:] = rho
	field_init_finest[1,:] = rho*u
	field_init_finest[2,:] = p/(gamma-1) + 0.5*rho*u**2

	# field_init[:,:] = field_init_dns[:,::rx_max]

def eos(field):

	# Estimate and return pressure value given field variables

	rho = field[0,:]
	m = field[1,:]
	E = field[2,:]
	u = m/rho
	e = E/rho

	p = (e-u**2/2)*rho*(gamma-1)

	return p

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

def navier_stokes(field, t):

	# Returns RHS of Navier-Stokes equations

	p = eos(field)

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



dim = 3 # Dimension of field (3: rho, m, E)
L = 2*np.pi
Re = 100
Pr = 1
gamma = 1.4

Nx_bl = 32
dx_bl = L/Nx_bl
t_final = 1
# Timestep
dt_bl = min(dx_bl, dx_bl**2*Re)*0.5
Nt_bl = int(t_final/dt_bl)
dt_bl = t_final/Nt_bl

rx_list = np.array([1,2,4,8]) # Grid size ratios

# Init field
rx_max = max(rx_list)
Nx_max = Nx_bl*rx_max
field_init_finest = np.zeros([dim, Nx_max])
init_field_rand()

x_finest = np.linspace(0, L, Nx_max, endpoint=False)
plt.plot(x_finest, field_init_finest[0,:])
plt.show()

for rx in rx_list:

	rt = rx
	Nx = Nx_bl*rx
	dx = dx_bl/rx
	Nt = Nt_bl*rt
	dt = t_final/Nt
	field_exact_dns = np.zeros([Nt, 4, dim, Nx])
	x = np.linspace(0, L, Nx, endpoint=False)

	t = 0
	field_4 = field_init_finest[:,::rx_max//rx]

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

		# Save field data
		# field[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# March in time
		t += dt

	print()

	plt.plot(x, field_4[2,:])


plt.legend(['N=32','N=64','N=128','N=256','N=512'])
plt.xlabel('x')
plt.ylabel('Energy')
plt.show()




