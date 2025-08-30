import torch
import numpy as np
#import torchvision
#from torchvision import transforms #will break somehow. Why?
from torch.nn import functional as F
from torch.utils import data

#Create a vector
x = torch.arange(12)
#reshape it to a 3x4 matrix
x = x.reshape(3,4)
#create a tensor
y = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
#element-wise addition
x + y, x - y, x * y, x / y, x ** y # ** is power
torch.exp(x) # e^x
#concatenate two tensors
X = torch.arange(12, dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=torch.float32)
torch.cat((X,Y), dim=0), torch.cat((X, Y), dim=1) # concatenate along rows, dim=0, along columns, dim=1 means horizontal
torch.zeros((2,3,4))#2 layers of 3x4 matrices
torch.ones((2,3,4))#2 layers of 3x4 matrices
torch.randn((2,3,4))#2 layers of 3x4 matrices with random values
len(X), X.shape, X.size()#len() gives the size of the first dimension, shape and size() give the shape of the tensor
#broadcasting mechanism
#When operating on two tensors, if their shapes are different, 
#PyTorch will automatically expand the smaller tensor to match the shape of the larger tensor.
#This is called broadcasting. 
#broadcasting mechanism transfrorms the smaller tensor to the same shape as the larger tensor without actually copying the data.
a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
a, b
#index and cutting
X[1], X[-1], X[1:2], X[::2] # get the first row, last row, second row, every other row
X[0:2, :] = 12 # get the first two rows, all columns, and set them to 12
X[0:2, :] = torch.tensor([[1,2,3,4],[5,6,7,8]]) # set the first two rows to specific values
X[(X < 5) | (X > 10)] # get elements less than 5 or greater than 10
X[X > 5] = 12 # set elements greater than 5 to 12  

#transform tensor to numpy array or other python objects
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)#type() used to check the type of a variable
a = torch.tensor([3.5])
a, a.item(),float(a),int(a)#transform a one-element tensor to a python number, .item() only works for one-element tensors
#transform numpy array to tensor
a = np.array([3.5])
b = torch.from_numpy(a)
a, b
#convert a list to a tensor
a = [3.5, 4, 5]
b = torch.tensor(a(dtype=np.float32))#torch.tensor() only accepts list of numbers, dtype make sure the data type is float32.
#dimensionallity decrease and increase
a = torch.arange(12).reshape(3,4)
a, a.shape  #3 rows, 4 columns
a.sum(), a.sum(dim=0), a.sum(dim=1) #sum of all elements, sum of each column, sum of each row
a.mean(), a.mean(dim=0), a.mean(dim=1) #mean of all elements, mean of each column, mean of each row
a.sum().item() #sum of all elements as a python number 
A_sum_axis0 = a.sum(dim=0)
A_sum_axis1 = a.sum(dim=1)
#dot product
y = torch.ones(4, dtype=torch.float32)
x = torch.arange(12, dtype=torch.float32).reshape(3,4)
x, y, torch.dot(x, y) # dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
torch.sum(x * y) # equivalent to dot product

#use matrix-vector product to calculate the dot product of each row of x and y
torch.matmul(x, y) # matrix-vector product

#caculate norm
torch.norm(x), torch.norm(y) # Frobenius norm for matrices, Euclidean norm for vectors
