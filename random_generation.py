import numpy as np
import matplotlib.pyplot as plt

def get_ap(shape):

	N = shape[-1]
	M = shape[-2]

	ret = np.ones(shape)

	for s in range(shape[0]):
		ap = np.ones((M, N))
		nb_square = np.random.randint(6, 13)
		last = 0
		mod = np.random.randint(2)
		for i in range(nb_square):
			#dur = int(np.random.normal(N/nb_square, N/(nb_square*2)))
			dur = int(np.random.uniform(0.05*(N/nb_square), 2.05*(N/nb_square)))
			ap[last:last+dur, :] = mod
			last = last+dur

			mod += 1
			mod = mod % 2

		X = np.arange(N)
		weights = np.random.normal(200, 15, size=M)
		Y = 500*np.exp(-0.011*X)

		b = np.repeat(Y[:, np.newaxis], M, axis=1)
		# print (M//1-1, 'M//1-1')
		# print (b.shape, 'b.shape')
		# print (Y.shape, 'Y.shape')
		# # print (weights, 'weights')
		# print (weights.shape, 'weights.shape')




		for dim in range(M//1 -1):
			# print (weights[dim], weights[dim])
			b[:,dim*1:(dim+1)*1] /= Y[int(weights[dim])]

		b[b>1] = 1

		ap = 1-(np.rot90(b,1)*(1-ap))

		if len(shape) == 3:
			ret[s,:,:] = ap
		elif len(shape) == 2:
			return ap
		else:
			print("Data shape not understood")

	return ret


def get_f0(shape):

	ret = np.zeros(shape)
	N = shape[-1]
	
	for i in range(shape[0]): 

		centroid = np.random.randint(50, 200)

		STD = np.random.randint(100, 500)

		f0 = (np.random.rand(N*2)-0.5)*STD + centroid

		f0 = np.convolve(f0, np.ones(30))[15:-15] / 30
		f0 = np.convolve(f0, np.ones(10))[5:-5] / 10

		f0 = f0[N//2:N + N//2]

		if len(shape) == 3:
			ret[i,0,:] = f0
		elif len(shape) == 2:
			ret[i,:] = f0
		elif len(shape) == 1:
			return f0
		else:
			print("Data shape not understood ...")
			quit()

	return ret

def get_random_h(shape_f0, shape_ap):
	"""
	Return f0 and ap at random but with good properties
	"""
	f0 = get_f0(shape_f0)
	ap = get_ap(shape_ap)

	#for i in range(len(f0)):
	#	plt.plot(f0[i,0,:])
	#	plt.show()

	#f0 = f0*(1-ap[:,:,-1])
	if len(shape_f0) == 3:
		f0[:,0,:] = f0[:,0,:]*(1-ap[:,:,0])
	elif len(shape_f0) == 1:
		f0[:] = f0[:]*(1-ap[:,0])

	f0[f0<50] = 0


	# for i in range(len(f0)):
	# 	plt.subplot(211)
	# 	plt.plot(f0[i,0,:])
	# 	plt.subplot(212)
	# 	plt.imshow(np.rot90(ap[i,:,:]), aspect="auto")
	# 	plt.show()


	return f0, ap


#for i in range(10):
	#plt.plot(get_f0((64,1,401))[0,0,:])
	#plt.show()
	#plt.imshow(get_ap((64,600,400))[0,:,:])
	#plt.show()

# N = 401
# M = 600

# ### AP GENERATION
# for i in range(10):

# 	plt.imshow(getAP(N, M), cmap=plt.cm.BuPu_r)
# 	plt.colorbar()
# 	plt.show()

# 	plt.plot(getF0(N))
# 	plt.show()
