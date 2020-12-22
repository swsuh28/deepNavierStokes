## NN trainig module

from func import *
import matplotlib.pyplot as plt


# Main loop (training iteration)
for iter_step in range(numIter+1):

	print('Iteration step: {0}'.format(iter_step+1))

	init_field_rand() # Randomly initialize field
	get_exact_dns() # Solve DNS -> Training data (reference)

	# Prediction
	t = 0 # Set time

	# Initial condition
	field_4 = field_init.copy()

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


	if iter_step%100 == 0: # Plot prediction for every N iteration steps

		get_coarse_solution() # Solve ILES (LES w/ no model)

		plt.plot(x_dns, field_exact_dns[-1,-1,2,:])
		plt.plot(x, field_iles[-1,-1,2,:])
		plt.plot(x, field[-1,-1,2,:])
		plt.xlabel('x')
		plt.ylabel('Energy')
		plt.legend(['Exact', 'No model', 'ML'])
		plt.title('Iteration step: {0}'.format(iter_step))
		# plt.show()
		plt.savefig('post/field_iter={0}.png'.format(iter_step))
		plt.close()

		np.savetxt('post/field_iter={0}.txt'.format(iter_step),\
					np.vstack((x, \
							field_exact[-1,-1,0,:], field_iles[-1,-1,0,:], field[-1,-1,0,:],\
							field_exact[-1,-1,1,:], field_iles[-1,-1,1,:], field[-1,-1,1,:],\
							field_exact[-1,-1,2,:], field_iles[-1,-1,2,:], field[-1,-1,2,:])).T)


	if iter_step == numIter:
		break

	# Estimate current loss
	loss = get_loss() # Loss
	print('Loss: {0}'.format(loss))
	loss_list[iter_step] = loss # Add to loss list (learning curve)


	# Solve adjoint PDE

	# Final condition (Discrete-adjoint)
	adj_0 = (dt/6) * 2*(field[-1,-1,:,:]-field_exact[-1,-1,:,:])

	print('Solving adjoint PDE...')

	for n in reversed(range(Nt)): # March backward in time

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


	# Compute gradient
	loss_gradient = get_loss_gradient()

	# Update NN parameters

	# update_param(loss_gradient, learningRate)
	rmsprop(loss_gradient) # RMSprop


# Plot data (optional)

# Compare No-model vs. ML-prediction
plt.plot(x,field_exact[-1,-1,0,:])
plt.plot(x,field_iles[-1,-1,0,:])
plt.plot(x,field[-1,-1,0,:])
plt.show()

plt.plot(x,field_exact[-1,-1,1,:])
plt.plot(x,field_iles[-1,-1,1,:])
plt.plot(x,field[-1,-1,1,:])
plt.show()

plt.plot(x,field_exact[-1,-1,2,:])
plt.plot(x,field_iles[-1,-1,2,:])
plt.plot(x,field[-1,-1,2,:])
plt.show()

# Plot learning curve
plt.plot(np.arange(numIter), loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.ylim(bottom=0)
plt.show()
np.savetxt('post/loss.txt', loss_list.reshape(numIter,1))


# Save NN data
save_nn()

