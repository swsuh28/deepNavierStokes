## Module for plotting E spectrum


from func import *
import matplotlib.pyplot as plt


def get_energy_spectrum(E, N):

	k = np.fft.fftfreq(N) * N
	Ek = np.abs(np.fft.fft(E))/N

	return Ek[:N//2-1], k[:N//2-1]


net_list = [Model() for i in range(dim)]


rlz = 100 # No. of realizations

E_dns_avg = np.zeros(Nx_dns//2-1)
E_iles_avg = np.zeros(Nx//2-1)
E_les_avg = np.zeros(Nx//2-1)


# Before training
for j in range(rlz):

	print('Realization: {0}'.format(j+1))

	init_field_rand()
	get_exact_dns()
	get_coarse_solution()

	t = 0 # Set time
	field_4 = field_init.copy()

	print('Prediction')

	for n in range(Nt):

		print('Time: {0}'.format(t), end="\r")

		# RK4 substep 0 = 4 in previous timestep
		field_0 = field_4.copy()


		# RK4 substep 0 -> 1
		h1, h1_grad_param, h1_grad_field = nn_force(field_0, net_list) # FP + BP
		k1 = navier_stokes(field_0, t) + h1
		field_1 = field_0 + 0.5*dt*k1


		# RK4 substep 0 -> 2
		h2, h2_grad_param, h2_grad_field = nn_force(field_1, net_list)
		k2 = navier_stokes(field_1, t+0.5*dt) + h2
		field_2 = field_0 + 0.5*dt*k2


		# RK4 substep 0 -> 3
		h3, h3_grad_param, h3_grad_field = nn_force(field_2, net_list)
		k3 = navier_stokes(field_2, t+0.5*dt) + h3
		field_3 = field_0 + dt*k3


		# RK4 substep 0 -> 4
		h4, h4_grad_param, h4_grad_field = nn_force(field_3, net_list)
		k4 = navier_stokes(field_3, t+dt) + h4
		field_4 = field_0 + (dt/6)*(k1+2*(k2+k3)+k4)


		# Save field data
		field[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# Save gradient data
		for i in range(dim):
			for k in range(len(net_list[i].weights)):
				h_grad_param[i][k][n,:,:,:,:] = np.array([h1_grad_param[i][k], \
														  h2_grad_param[i][k], \
														  h3_grad_param[i][k], \
														  h4_grad_param[i][k]])
			h_grad_field[n,:,i,:,:,:] = np.array([h1_grad_field[i], \
												  h2_grad_field[i], \
												  h3_grad_field[i], \
												  h4_grad_field[i]])

		# March in time
		t += dt

	print()


	E_dns, k_dns = get_energy_spectrum(field_exact_dns[-1,-1,2,:], Nx_dns)
	E_iles, k_iles = get_energy_spectrum(field_iles[-1,-1,2,:], Nx)
	E_les, k_les = get_energy_spectrum(field[-1,-1,2,:], Nx)

	E_dns_avg += E_dns/rlz
	E_iles_avg += E_iles/rlz
	E_les_avg += E_les/rlz

plt.plot(np.log10(k_dns), np.log10(E_dns_avg))
plt.plot(np.log10(k_iles), np.log10(E_iles_avg))
plt.plot(np.log10(k_les), np.log10(E_les_avg))
plt.xlabel('log10(k)')
plt.ylabel('log10(E)')
plt.legend(['DNS', 'ILES', 'LES'])
plt.savefig('post/Ek_before.png')
plt.close()
np.savetxt('post/Ek_before_dns.txt', np.vstack((k_dns, E_dns_avg)).T)
np.savetxt('post/Ek_before_iles.txt', np.vstack((k_iles, E_iles_avg)).T)
np.savetxt('post/Ek_before_les.txt', np.vstack((k_les, E_les_avg)).T)


# Load NN
for i in range(dim):
	net_list[i] = torch.load('post/nnModel_{0}.pt'.format(i))
	print(net_list[i].weights)

E_dns_avg = np.zeros(Nx_dns//2-1)
E_iles_avg = np.zeros(Nx//2-1)
E_les_avg = np.zeros(Nx//2-1)


# After training
for j in range(rlz):

	print('Realization: {0}'.format(j+1))

	init_field_rand()
	get_exact_dns()
	get_coarse_solution()

	t = 0 # Set time
	field_4 = field_init.copy()

	print('Prediction')

	for n in range(Nt):

		print('Time: {0}'.format(t), end="\r")

		# RK4 substep 0 = 4 in previous timestep
		field_0 = field_4.copy()


		# RK4 substep 0 -> 1
		h1, h1_grad_param, h1_grad_field = nn_force(field_0, net_list) # FP + BP
		k1 = navier_stokes(field_0, t) + h1
		field_1 = field_0 + 0.5*dt*k1


		# RK4 substep 0 -> 2
		h2, h2_grad_param, h2_grad_field = nn_force(field_1, net_list)
		k2 = navier_stokes(field_1, t+0.5*dt) + h2
		field_2 = field_0 + 0.5*dt*k2


		# RK4 substep 0 -> 3
		h3, h3_grad_param, h3_grad_field = nn_force(field_2, net_list)
		k3 = navier_stokes(field_2, t+0.5*dt) + h3
		field_3 = field_0 + dt*k3


		# RK4 substep 0 -> 4
		h4, h4_grad_param, h4_grad_field = nn_force(field_3, net_list)
		k4 = navier_stokes(field_3, t+dt) + h4
		field_4 = field_0 + (dt/6)*(k1+2*(k2+k3)+k4)


		# Save field data
		field[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# Save gradient data
		for i in range(dim):
			for k in range(len(net_list[i].weights)):
				h_grad_param[i][k][n,:,:,:,:] = np.array([h1_grad_param[i][k], \
														  h2_grad_param[i][k], \
														  h3_grad_param[i][k], \
														  h4_grad_param[i][k]])
			h_grad_field[n,:,i,:,:,:] = np.array([h1_grad_field[i], \
												  h2_grad_field[i], \
												  h3_grad_field[i], \
												  h4_grad_field[i]])

		# March in time
		t += dt

	print()

	# plt.plot(x_dns, field_exact_dns[-1,-1,2,:])
	# plt.plot(x, field_iles[-1,-1,2,:])
	# plt.plot(x, field[-1,-1,2,:])
	# plt.show()


	E_dns, k_dns = get_energy_spectrum(field_exact_dns[-1,-1,2,:], Nx_dns)
	E_iles, k_iles = get_energy_spectrum(field_iles[-1,-1,2,:], Nx)
	E_les, k_les = get_energy_spectrum(field[-1,-1,2,:], Nx)

	E_dns_avg += E_dns/rlz
	E_iles_avg += E_iles/rlz
	E_les_avg += E_les/rlz


plt.plot(np.log10(k_dns), np.log10(E_dns_avg))
plt.plot(np.log10(k_iles), np.log10(E_iles_avg))
plt.plot(np.log10(k_les), np.log10(E_les_avg))
plt.xlabel('log10(k)')
plt.ylabel('log10(E)')
plt.legend(['DNS', 'ILES', 'LES'])
plt.savefig('post/Ek_after.png')
plt.close()
np.savetxt('post/Ek_after_dns.txt', np.vstack((k_dns, E_dns_avg)).T)
np.savetxt('post/Ek_after_iles.txt', np.vstack((k_iles, E_iles_avg)).T)
np.savetxt('post/Ek_after_les.txt', np.vstack((k_les, E_les_avg)).T)


