'''
@File    :   main.py
@Time    :   2022/08/22 11:17:09
@Author  :   Marco Sento 
@Version :   1.0
@Desc    :   None
'''

import CNN, Datasets


if __name__ == '__main__':

    #network istantiation

    net=CNN.CNN()

    #build the network 

    net.add_conv(image_dim = (1,28,28), kernels = (8,3,3,1), padding=0 ,stride=2 ,bias=0.1 ,eta=0.01)
    net.add_conv(image_dim = (8,13,13), kernels = (2,3,3,8), padding=0 ,stride=2 ,bias=0.1 ,eta=0.01)
    net.add_dense( input = 2*6*6 , hidden = [72], num_classes=10, adam=True , eta=0.01)
    
    #load the wanted dataset

    data=Datasets.get_mnist(b_random=False)

    net.load_dataset(data)

    #sanity check

    #net.sanity_check()

    #train the network (Batch Size = 1)

    net.training( epochs=1, preview_ratio=1 )

    net.plot_results()

    #evaluate new samples 

    net.testing(preview_ratio=0)

    net.plot_results()

