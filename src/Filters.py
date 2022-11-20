'''
@File    :   Filters.py
@Time    :   2022/08/20 15:39:48
@Author  :   Marco Sento 
@Version :   1.0
@Desc    :   None
'''

import sys
import numpy as np
import pandas #for debug purpose

ALPHA = 0.001

def ReLu(input_volume):
	input_volume[ input_volume <0 ] *= ALPHA  
	return input_volume

def deLeReLu(input_volume):
    #Same of doing:
    input_volume[ input_volume < 0 ] = ALPHA #faster
    return input_volume


	
class Convolutional:
	
	#parameter:
		#image_dim = touple of dimension of the image and depth ( H x W x depth)
		#kernels = touple of number of kernels x dimension of the kernel x depth ( n_kern x H x W x depth)
	def __init__(self, image_dim=(1,16,16), kernels=(2,3,3,1), padding=1, stride=1, bias=0.1, eta=0.01): #eta=learning rate
		
		if (image_dim[0]!=kernels[3]): 
			print(sys.stderr,"Error: depth of the filter must match the depth of the image.")

		self.image_dim=( image_dim[0], image_dim[1] + 2*padding , image_dim[2] + 2*padding )
		self.specs=kernels 		# Only a tuple with the specifications
		self.padding=padding
		self.stride=stride
		self.eta = eta
		self.iteration = 0 		# To update the gradient descent 
		self.bias = []   		# List of bias, same for each kernel so one value for each of it (kernels[0])		
		self.cache = 0			# Useful to save a padded image

		self.filters = np.random.rand(*kernels)*0.1 # The parameters of the neurons aka kernels

		for i in range(kernels[0]): self.bias.append(bias)

		
			


	def pad(self,image):
		
		new_img=np.zeros( (self.image_dim[0], self.image_dim[1] , self.image_dim[2] ) )
		y, x = 0, 0
		for depth in range( self.image_dim[0] ):
		
			for y in range( self.image_dim[1] - 2*self.padding):
				
				for x in range(  self.image_dim[2] - 2*self.padding):
					
					new_img[depth][x+self.padding][y+self.padding] = image[depth][x][y]
		
		return new_img
	



	def out_dimension(self, f_y, f_x):


		out_H = (self.image_dim[1] - f_y + 2*self.padding)/self.stride +1
		out_W = (self.image_dim[2] - f_x + 2*self.padding)/self.stride +1	
		depth = self.image_dim[0]

		#if(out_W%1!=0 and out_H%1!=0): print(sys.stderr,"\n\nWarning: padding and stride combination is not integer.\n\n")
		out_W=int(out_W)
		out_H=int(out_H)
	
		return out_H, out_W, depth
	


	
	def fwd(self, image):
		
		#Produces a volume of size D2xH2xW2 where:
			#W2=(W1−F+2P)/S+1
			#H2=(H1−F+2P)/S+1
			#D2= kernel number
		
		'''print("Immagine prima della convoluzione")
		print(pandas.DataFrame(image[0,:,:]))
		'''

		if(self.padding!=0): 
			image=self.pad(image)
			
		self.cache=image	#save for backpropagation

		f_y= self.specs[1]
		f_x= self.specs[2]
		f_d= self.specs[3]
		
		out_H, out_W, depth = self.out_dimension(f_y,f_x)
		n_kernel= self.specs[0]

		

		out = np.zeros( ( n_kernel, out_H, out_W ))

		y_out, x_out = 0,0

		#convolution algorithm:
			# each element of the kernel is superimposed to the one of the image
			# if the pixel of the image is valid, then they are multiplied
			# in output will be present the sum of the 3 level (R G B) convolutions 


		for kernel in range(n_kernel):

			'''print("Immagine del kernel " + str(kernel))
			print(pandas.DataFrame(self.filters[kernel,:,:,0]))
			'''

			for layer in range( depth ): #each kernel has n (3) layers, one for each of the n (3) layers of the image, the depth.
				
				y_out, x_out = 0,0

				for y in range( 0, self.image_dim[1] - f_y, self.stride):		# image = ( depth x H x W )
					x_out=0

					for x in range( 0,  self.image_dim[2] - f_x, self.stride ):

						for f_y_it in range(f_y):
							for f_x_it in range(f_x):
								out[kernel][y_out][x_out] += (image[layer][y + f_y_it][x + f_x_it]*self.filters[kernel][f_y_it][f_x_it][layer]) 
								
						x_out+=1
					y_out+=1
			
			#out[kernel]+=self.bias[kernel]

		return ReLu(out)	
		
	

	def bp(self, d_out_vol):

		#INPUT:
			# d_out_vol = d(out_volume) = loss gradient of the output of this conv layer  ( out_W, out_H, out_depth ) 
		#RETURN:
			#d_input 		= loss gradient of the input image received during the convolution (np.array)
			#self.d_filters	= gradient of the filter (np.array)
			#self.d_bias	= gradient of the bias   (list)


		d_out_vol=deLeReLu(d_out_vol)

		image=self.cache

		# Compute the effective dimensions of the image

		f_y= self.specs[1]
		f_x= self.specs[2]
		f_d= self.specs[3]
		
		#out_H, out_W, depth = self.out_dimension(f_y,f_x)
		n_kernel= self.specs[0]


		d_input = np.zeros( (  self.image_dim[0], self.image_dim[1], self.image_dim[2]) )
		d_filters = np.zeros(self.specs)   # The list of lists of error terms (lowercase deltas) 
		d_bias=[]

		for kernel in range(n_kernel):
		
			y_out, x_out = 0,0

			for y in range( 2*self.padding, self.image_dim[1] - f_y - 2* self.padding, self.stride):		# image = ( H x W x depth )
				
				for x in range(  2*self.padding,  self.image_dim[2] - f_x - 2* self.padding, self.stride ):
					# loss gradient of the input passed in the convolution operation

					for layer in range( f_d ):
						
						for f_y_it in range(f_y):
							for f_x_it in range(f_x):
								
								d_filters[kernel, f_y_it, f_x_it, layer] +=  image[layer, y + f_y_it, x + f_x_it ] * d_out_vol[kernel, y_out, x_out ]
								d_input[layer, y + f_y_it, x + f_x_it ] += d_out_vol[kernel,  y_out, x_out ] * self.filters[kernel,f_y_it,f_x_it, layer]
					
					x_out+=1

				x_out=0
				y_out+=1

			# loss gradient of the bias
			d_bias.append( np.sum(d_out_vol[kernel]) )

		self.gd(d_filters, d_bias)

		return d_input
		
		
	
	def gd(self, d_filters, d_bias):
	
		self.eta = self.eta * np.exp(-self.iteration/20000)
				
		self.filters -= self.eta * d_filters 
		for i in range(len(self.bias)): self.bias[i] -= self.eta * d_bias[i]
		self.iteration +=1


class Pooling: 

	def __init__(self, image_dim=(1, 16,16), mode='avg', size=2, stride=2):
		self.cache=None
		self.image_dim=image_dim
		self.size=size
		self.stride=stride
		self.mode=mode

	def fwd(self,images): #For Pooling layers, it is not common to pad the input using zero-padding.
		
		self.cache=images

		layers, w_in, h_in = self.image_dim

		w_out = int((w_in - self.size)/self.stride)+1
		h_out = int((h_in - self.size)/self.stride)+1

		out=np.zeros((layers, w_out, h_out))

		for layer in range(layers):
			y_out, x_out = 0,0

			for y in range(0, h_in - self.size, self.stride):
				x_out=0

				for x in range(0, w_in - self.size, self.stride):

					if self.mode=='avg':
						out[layer, y_out, x_out] = np.average(images[layer,  y:y+self.size, x:x+self.size])
					else: #max pooling is applied
						out[layer, y_out, x_out] = np.max(images[ layer, y:y+self.size, x:x+self.size])

					x_out+=1
				y_out+=1
		
		return out


	def bp(self, d_out): #d_out is like the derivative of the pooling output, image is the input of the pooling layer
		
		layers, w_in, h_in = self.image_dim

		w_out = int((w_in - self.size)/self.stride)+1
		h_out = int((h_in - self.size)/self.stride)+1

		out = np.zeros((layers, w_in, h_in))

		for layer in range(layers):
			
			y_out, x_out = 0,0
			for y in range(0, h_out, self.stride):
				x_out=0

				for x in range(0, w_out, self.stride):

					if self.mode=='avg':	#not sure about that

						average_dout=d_out[layer,y_out,x_out]/(self.size*2)
						out[layer, y:(y+self.size), x:(x+self.size)] += np.ones((self.size,self.size))*average_dout

					else: #max pooling is applied

						area = self.cache[layer, y:y+self.size, x:x+self.size]
						index = np.nanargmax(area)
						(y_i, x_i) = np.unravel_index(index, area.shape)
						out[layer, y + y_i, x + x_i] += d_out[layer, y_out, x_out]
				
					x_out+=1

				y_out+=1

		return 	out


