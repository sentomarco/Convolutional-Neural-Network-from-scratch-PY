'''
@File    :   CNN.py
@Time    :   2022/08/20 15:40:47
@Author  :   Marco Sento 
@Version :   1.0
@Desc    :   None
'''
import time, sys
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import Filters, MLP

TAB="          "


class CNN:



	def __init__(self):

		self.layers=[]
		self.DenseInputShape=()
		self.numClasses=0

		self.train, self.valid, self.test = [], [], []
		self.train_acc, self.valid_acc, self.test_acc = [], [], []
		self.train_loss, self.valid_loss, self.test_loss = [], [], []



	def add_conv(self, image_dim=(1,16,16), kernels = (2,3,3,1), padding=1, stride=1, bias=0.1, eta=0.01):
		self.layers.append(Filters.Convolutional(image_dim,kernels,padding,stride,bias,eta))

			

	def add_pooling(self, image_dim=(1,16,16), mode='avg', size=2, stride=2):
		self.layers.append(Filters.Pooling(image_dim, mode, size, stride))



	def add_dense(self, input=2*6*6, hidden=[72], num_classes=10, adam=True, eta = 0.5):
		self.layers.append(MLP.MultiLayerPerceptron(layers=[input, *hidden, num_classes], adam=adam, eta=eta))
		self.numClasses=num_classes



	def load_dataset(self, dataset):

		print("\no Datasets loaded\n")

		self.train.append(dataset[0])
		self.train.append(dataset[1])
		self.valid.append(dataset[2])
		self.valid.append(dataset[3])
		self.test.append(dataset[4])
		self.test.append(dataset[5])


	def __forward(self, image, b_plot=False ):

		layerNumber=len(self.layers) 
		plotList=[]
		if(b_plot): plotList.append(image)

		#Forward propagation: 
		for i in range(layerNumber):	

			if ( i == ( layerNumber - 1 ) ):	#dense layer	

				#Saving the dimension of the dense layer input
				if not self.DenseInputShape: 
					self.DenseInputShape=image.shape

				image=image.flatten()	#fitting the dense's input

			image=self.layers[i].fwd(image)	

			if(b_plot): plotList.append(image)

		if(b_plot): self.__plot_preview(plotList)	
				
		return image



	def __backward(self, gradient):
		
		layerNumber=len(self.layers) 

		#Back propagation
		for i in range(layerNumber):	
				
			gradient=self.layers[ layerNumber - 1 - i ].bp(gradient)	

			if ( ( layerNumber - 1 - i ) == ( layerNumber - 1 ) ):
				gradient=gradient.reshape(self.DenseInputShape)



	def __iterate(self, dataset, loss_list, acc_list ,b_training=False, preview_ratio=0):

		accuracy, loss, correctAnswer = 0, 0, 0 
		b_init_prew, b_plot = True, False

		t_start=time.time()

		bar=tqdm(range(len(dataset[1])), ncols=150, colour= "#9dff00" )	#It is an iterator

		for i in bar:
			
			b_plot=False
			if( preview_ratio != 0 and (i+1) % (len(dataset[1])*preview_ratio) ==0 and i!=0): b_plot = True

			image = dataset[0][i]	
			label = dataset[1][i]

			# Feed the sample into the network 
			result=self.__forward(image, b_plot)

			# Error evaluation:
			y=np.zeros(self.numClasses)
			y[label]=1

			error = y - result #Evaluation error
			
			# Update MSE loss function
			loss = sum( error ** 2) / self.numClasses
			loss_list.append( loss )

			if np.argmax(result) == label: correctAnswer += 1

			# Update accuracy
			accuracy = correctAnswer * 100 / ( i + 1 )
			acc_list.append( accuracy )

			if b_training: 
				# Adjust the weight
				self.__backward(error)
			
			
			# Plot progress
			if(b_init_prew and b_plot):
				b_init_prew=False
				bar.write("\t  Classification preview:")
				bar.write(" ")
			if(b_plot):
				bar.write("\t  [ Iteration %d ] Expected class : %d  ||  Predicted class: %d ||  See potted graphs" % (i+1, label, np.argmax(result)) )
				bar.write(" ")

			bar.set_description(TAB + "Accuracy: %02.2f  ||  Loss: %02.2f  ||  Samples" % (accuracy, loss))



	# prew_ratio express the % of iterations elapsed between previews of the evolution of the filters
	def training(self, epochs, preview_ratio=0):
		
		if not self.layers:
			print(sys.stderr,"Error: the network has no layers.")

		else:
			
			t_start=time.time()
			
			print("\no Traininig:")

			#different epochs works on the same training dataset
			for epoch in range(epochs) :
				
				print("\n\to Epoch %d" % (epoch+1))
				#The batch size defines the number of samples to work through before updating the internal model parameters.
				#Batch Size = 1 => stochastic gradient descent learning algorithm
				self.__iterate(self.train, self.train_loss, self.train_acc, True, preview_ratio)
				
				print("\n\to Validation")
				#the model evaluation is performed on the validation set after every epoch	
				self.__iterate(self.valid, self.valid_loss, self.valid_acc, False, preview_ratio)



	# prew_ratio express the % of iterations elapsed between previews of the evolution of the filters
	def testing(self, preview_ratio=0 ):
		
		if not self.layers:
			print(sys.stderr,"Error: the network has no layers.")

		else:
			print("\no Testing:\n")
			#evaluate the performances on the test dataset
			self.__iterate(self.test, self.test_loss, self.test_acc, False, preview_ratio)



	def sanity_check(self, set_size=50 , epochs=200, preview_ratio=0 ):

		if not self.layers:
			print(sys.stderr,"Error: the network has no layers.")

		else:
			t_start=time.time()
			print("\no Performing sanity check:")

			check_list, check_loss, check_acc = [], [], []
			check_list.append(self.train[0][:set_size])
			check_list.append(self.train[1][:set_size])

			for epoch in range(epochs) :
				check_loss.clear()
				check_acc.clear()
				print("\n\to Epoch %d" % (epoch+1))
				self.__iterate(check_list, check_loss, check_acc, True, preview_ratio)

			print("\n\tFinal losses: %02.2f" % (np.mean(check_loss)) )


	def plot_results(self):
		
		if self.test_acc: 
			len_test=len(self.test_acc)
			delta_test=int(len_test/1000)
			_test_acc=self.test_acc[::delta_test]
			_test_loss=self.test_loss[::delta_test]
		
		len_train=len(self.train_acc)
		delta_train=int(len_train/1000)
		_train_acc=self.train_acc[::delta_train]
		_train_loss=self.train_loss[::delta_train]

		len_valid=len(self.valid_acc)
		delta_valid=int(len_valid/1000)
		_valid_acc=self.valid_acc[::delta_valid]
		_valid_loss=self.valid_loss[::delta_valid]

		print("\n\no Results plotted\n")

		fig=plt.figure(figsize=(8, 10))
		ax = []
		ax.append( fig.add_subplot(2, 1, 1) )

		x_axe=np.array(range(1000))/10

		plt.plot(x_axe, _train_acc, 'b', label='Training set - {0} samples'.format(len_train), linewidth=2.0)
		plt.plot(x_axe, _valid_acc, 'r', label='Validation set - {0} samples'.format(len_valid), linewidth=2.0)

		if self.test_acc: 
			plt.plot(x_axe, _test_acc, 'g', label='Test set - {0} samples'.format(len_test), linewidth=2.0)
			ax[-1].set_title('Test accuracy')
		else: ax[-1].set_title('Training accuracy')

		plt.ylabel('Accuracy %')
		plt.xlabel('Iteration %')
		plt.legend()

		ax.append( fig.add_subplot(2, 1, 2) )
		
		plt.plot(x_axe, _train_loss, 'b', label='Training set - {0} samples'.format(len_train), linewidth=2.0)
		plt.plot(x_axe, _valid_loss, 'r', label='Validation set - {0} samples'.format(len_valid), linewidth=2.0)

		if self.test_loss: 
			plt.plot(x_axe, _test_loss, 'g', label='Test set - {0} samples'.format(len_test), linewidth=2.0)
			ax[-1].set_title('Test losses')
		else: ax[-1].set_title('Training losses')

		plt.ylabel('Loss')
		plt.xlabel('Iteration %')
		plt.legend()
		
		if self.test_loss: plt.savefig('TestResults.jpg')
		else:  plt.savefig('TrainingResults.jpg')

		plt.show()

		
	def __plot_preview(self, images_list):
		
		rows, columns = int(len(images_list)/2+0.5), int(len(images_list)/2+0.5)
		fig = plt.figure(figsize=(10, 10))
		ax = []

		ax.append( fig.add_subplot(rows, columns, 1) )
		ax[-1].set_title("Input ")  # set title
		plt.imshow(images_list[0][0, :, :])

		for j,img in enumerate(images_list[1:-1]):
			
			# create subplot and append to ax
			ax.append( fig.add_subplot(rows, columns, j+2) )
			ax[-1].set_title("Filter "+str(j+1))  # set title
			plt.imshow(img[0, :, :])


		ax.append( fig.add_subplot(rows, columns, len(images_list)) )
		plt.bar(np.arange(10),images_list[-1])
		ax[-1].set_title('Predicted classes')
		plt.ylabel('Confidence')
		plt.xlabel('Number')
		
		
		plt.show(block=False)
		plt.pause(0.001)
		
		



	def __del__(self):
		print("\no Done\n\n")

