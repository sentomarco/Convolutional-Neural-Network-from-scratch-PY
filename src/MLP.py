import numpy as np
import sys

BETA1=0.9
BETA2=0.999
EPS=1e-8


#Perceptron: a neuron with:
# n input sample [0 - n-1]
# 1 bias point (offset for the neural decision threshold)
# 1 neuron able to sum the weigthed input
# 1 output non linear function to betterer distinguish between data (sigmoid activation function)
class Perceptron: 


	def __init__(self,inputs, bias=1.0):
		self.weights = (np.random.rand(inputs+1))*0.1 
		#should be one couple (m,v) for each parameter that undergoes to Adam
		self.m, self.v = np.zeros(inputs+1), np.zeros(inputs+1)		
		self.bias = bias
		self.n_input = inputs
		
	def run(self, x): 
		# x is the list of inputs
		sum = np.dot(np.append(x, self.bias),self.weights)
		
		#the weighted inputs are passed to the activation function
		return self.sigmoid(sum)
		
	def set_weights(self, w_init):
		#specify the values of the weights, they cover also the bias
		if len(w_init) == (self.n_input + 1) :  
			self.weights = np.array(w_init)	

		else:
			print("Error: A neuron has received a worng number of input", sys.stderr)
			
			if len(w_init) < (self.n_input + 1) :  
				base=np.zeros(self.n_input + 1)
				self.weigths = np.array(w_init+base)	
			else:   
				self.weigths = np.array(w_init[:(self.n_input + 1)])	
			
	
	def sigmoid(self, x):
		# normalizing the output of the neuron with the sigmoid function
		return 1/(1+np.exp(-x))



#A multilayer perceptron class that uses the Perceptron class above.   
#Attributes:
#  layers:  A python list with elements the number of neurons per layer.
#  bias:    The bias term. The same bias is used for all neurons.
#  eta:     The learning rate.
class MultiLayerPerceptron:      

	def __init__(self, layers, bias = 1.0, adam=True,eta = 0.5):

		self.layers = np.array(layers,dtype=object)
		self.eta = eta
		self.bias = bias	
		self.network = [] # The list of lists of neurons
		self.values = []  # The list of lists of output values
		self.dactiv = []       # The list of lists of error terms (delta)
		self.loss_gradient = [] # The list of list of loss gradient of each neuron
		self.iteration=0
		self.b_adam=adam

		for i in range(len(self.layers)):
		
			self.loss_gradient.append([])
			self.dactiv.append([])
			self.values.append([])
			self.network.append([])
			
			self.values[i] = [0.0 for j in range(self.layers[i])]
			self.dactiv[i] = [0.0 for j in range(self.layers[i])]
			self.loss_gradient[i] = [0.0 for j in range(self.layers[i])]
			
			#network[0] is the input, it has no neurons
			if i > 0:      
				for j in range(self.layers[i]): 
					self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
		
		self.network = np.array([np.array(x) for x in self.network],dtype=object)
		self.values = np.array([np.array(x) for x in self.values],dtype=object)
		self.dactiv = np.array([np.array(x) for x in self.dactiv],dtype=object)
		self.loss_gradient = np.array([np.array(x) for x in self.loss_gradient],dtype=object)

	def set_weights(self, w_init):
		
		for i in range(len(w_init)):
			for j in range(len(w_init[i])):
				self.network[i+1][j].set_weights(w_init[i][j])
				



	def fwd(self, x): 
		
		self.iteration+=1

		x = np.array(x,dtype=object)
		self.values[0] = x
		
		for i in range(1,len(self.network)):
			for j in range(self.layers[i]): 

				self.values[i][j] = self.network[i][j].run(self.values[i-1]) 
				#for each neuron execute a vectorial product: (inputs of the neuron and bias) * (weights)
				#The values are the result of the sigmoid activation function
				
		return self.values[-1] #output layer of the network


	def Adam(self, m, v, derivative ):

		t=self.iteration
		dx=derivative
		m=BETA1*m+(1-BETA1)*dx
		mt=m/(1-BETA1**t)
		v=BETA2*v + (1-BETA2)*(dx**2)
		vt = v / (1-BETA2**t)
		delta = self.eta * mt / (np.sqrt(vt) + EPS)

		return m,v,delta
			

	#Calculate the derivatives and update the weights 			
	def gd(self):
		
		for i in range(1,len(self.network)): 
		
			for j in range(self.layers[i]):	 
			#j iterates over the number of perceptrons in the layer[i]
				for k in range(self.layers[i-1]+1): 
				#k iterates on the number of the previous output
				#since each previous neuron is connected to the present one 
				
					if k==self.layers[i-1]:
						db=self.dactiv[i][j]
						delta = self.eta * db
					else:
						dw = self.dactiv[i][j] * self.values[i-1][k]		# dw = dscore * X

						if not self.b_adam: delta = self.eta * dw			# learning rate * dw
						else: self.network[i][j].m[k], self.network[i][j].v[k], delta = self.Adam( self.network[i][j].m[k], self.network[i][j].v[k], dw)
						
					self.network[i][j].weights[k] += delta 	



	def bp(self,error):

		outputs = self.values[-1]
		
		#Calculate the output error terms
		#error propagated back through the derivative of the sigmoid function:
		self.dactiv[-1] = error * outputs * (1 - outputs)

		#Calculate the error term of each unit on each layer
		for i in reversed(range(1,len(self.network)-1)): 
		
			for h in range(len(self.network[i])):	
			#h iterates over the number of perceptrons in the layer[i]
			
				fwd_error = 0.0
				
				for k in range(self.layers[i+1]): 
				#k iterates the neuron in the i+1 layer
					fwd_error += self.network[i+1][k].weights[h] * self.dactiv[i+1][k]

				self.loss_gradient[i][h]=fwd_error # d_input
				self.dactiv[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error # dscore  
				

		self.gd()        
		
		return  self.loss_gradient[1]


