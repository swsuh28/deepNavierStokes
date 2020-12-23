############################################
#
# Module for validating adjoint formulation
#
############################################


from func import *
import matplotlib.pyplot as plt


# Initialize field variables
# init_field_mms()
init_field_rand()

plt.plot(x, field_init[1,:])
plt.show()



# Generate training dataset : DNS / MMS
get_exact_dns()
# get_exact_mms()
# -> Saved to field_exact


# Predict : Solve PDE
t = 0
field_4 = field_init.copy() # Initial condition

print('Prediction')


# RK4 Loop
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

	field[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

	# Save gradient data
	for i in range(dim):
		for k in range(len(net_list[i].weights)):
			h_grad_param[i][k][n,:,:,:,:] = np.array([h1_grad_param[i][k], \
													  h2_grad_param[i][k], \
													  h3_grad_param[i][k], \
													  h4_grad_param[i][k]])
		h_grad_field[n,:,i,:,:] = np.array([h1_grad_field[i], \
											   h2_grad_field[i], \
											   h3_grad_field[i], \
											   h4_grad_field[i]])

	# March in time
	t += dt

print()


plt.plot(x,field_exact[-1,-1,0,:])
plt.plot(x,field[-1,-1,0,:])
plt.show()


# Estimate current loss
loss = get_loss()
print('Loss: {0}'.format(loss))


# Solve Adjoint PDE
adj_0 = (dt/6) * 2*(field[-1,-1,:,:]-field_exact[-1,-1,:,:]) # Final condition (Discrete-adjoint)

print('Adjoint')


# RK4 Loop (Backward in time)
for n in reversed(range(Nt)):

	print('Time: {0}'.format(t), end="\r")

	# RK4 substep 4 = 0 in previous timestep
	adj_4 = adj_0.copy()

	# RK4 substep 4 -> 3
	field_3 = field[n,2,:,:]
	h_grad_field_T_3 = np.transpose(h_grad_field[n,3,:,:,:,:],(0,1,3,2))

	flux_grad_3 = np.zeros([dim, Nx])
	for i in range(dim):
		for j in range(dim):
			flux_grad_3[i,:] += h_grad_field_T_3[j,i,:,:] @ adj_4[j,:]

	k3 = 0.5*adjoint(field_3, adj_4, t) + 0.5*flux_grad_3 \
			+ 2*(field_3-field_exact[n,2,:,:])
	adj_3 = adj_4 + dt*k3

	# RK4 substep 4 -> 2
	field_2 = field[n,1,:,:]
	h_grad_field_T_2 = np.transpose(h_grad_field[n,2,:,:,:,:],(0,1,3,2))

	flux_grad_2 = np.zeros([dim, Nx])
	for i in range(dim):
		for j in range(dim):
			flux_grad_2[i,:] += h_grad_field_T_2[j,i,:,:] @ adj_3[j,:]

	k2 = adjoint(field_2, adj_3, t-0.5*dt) + flux_grad_2 \
			+ 2*(field_2-field_exact[n,1,:,:])
	adj_2 = adj_4 + 0.5*dt*k2

	# RK4 substep 4 -> 1
	field_1 = field[n,0,:,:]
	h_grad_field_T_1 = np.transpose(h_grad_field[n,1,:,:,:,:],(0,1,3,2))

	flux_grad_1 = np.zeros([dim, Nx])
	for i in range(dim):
		for j in range(dim):
			flux_grad_1[i,:] += h_grad_field_T_1[j,i,:,:] @ adj_2[j,:]

	k1 = 2*adjoint(field_1, adj_2, t-0.5*dt) + 2*flux_grad_1 \
			+ 2*(field_1-field_exact[n,0,:,:])
	adj_1 = adj_4 + 0.5*dt*k1

	# RK4 substep 4 -> 0
	if n == 0:
		field_4 = field_init
		field_exact_4 = field_init
	else:
		field_4 = field[n-1,3,:,:]
		field_exact_4 = field_exact[n-1,3,:,:]

	h_grad_field_T_4 = np.transpose(h_grad_field[n,0,:,:,:,:],(0,1,3,2))

	flux_grad_4 = np.zeros([dim, Nx])
	for i in range(dim):
		for j in range(dim):
			flux_grad_4[i,:] += h_grad_field_T_4[j,i,:,:] @ adj_1[j,:]

	k4 = adjoint(field_4, adj_1, t-dt) + flux_grad_4 \
			+ 2*(field_4-field_exact_4)
	adj_0 += dt/6 * (k1+2*(k2+k3)+k4)

	# Save adjoints
	adj[n,:,:,:] = [adj_1, adj_2, adj_3, adj_4]
	
	# March backward in time
	t -= dt

print()



# Compute gradient of loss
loss_gradient = get_loss_gradient()
# -> grad_loss_param


# Error plot
numExp = 16 # number of points to plot
error_list = np.zeros(numExp)


# Update NN parameters
for exp in range(numExp):

	alpha = pow(10, -exp) # Rate of change of NN parameters
	update_param(loss_gradient, alpha) # Update


	# Prediction
	t = 0
	field_4 = field_init.copy() # Initial condition

	print('Prediction')


	# RK4 Loop
	for n in range(Nt):

		print('Time: {0}'.format(t), end="\r")

		# RK4 substep 0 = 4 in previous timestep
		field_0 = field_4

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

		field[n,:,:,:] = np.array([field_1, field_2, field_3, field_4])

		# Save gradient data
		for i in range(dim):
			for k in range(len(net_list[i].weights)):
				h_grad_param[i][k][n,:,:,:,:] = np.array([h1_grad_param[i][k], \
														  h2_grad_param[i][k], \
														  h3_grad_param[i][k], \
														  h4_grad_param[i][k]])
			h_grad_field[n,:,i,:,:] = np.array([h1_grad_field[i], \
												   h2_grad_field[i], \
												   h3_grad_field[i], \
												   h4_grad_field[i]])

		# March in time
		t += dt

	print()



	# Estimate loss
	loss_new = get_loss()
	print('Loss: {0}'.format(loss_new))


	# Estimate error
	grad_square = 0 # Squared sum of all gradients
	for i in range(dim):
		for j in range(len(net_list[i].weights)):
			grad_square += np.sum(loss_gradient[i][j]**2)

	error_list[exp] = abs((loss_new-loss)/alpha + grad_square)
	print(error_list[exp])


	# Initialize parameters
	update_param(loss_gradient, -alpha)


# Print & Plot errors
print(error_list)

plt.plot(np.arange(numExp), np.log10(error_list), '.-')
plt.show()


