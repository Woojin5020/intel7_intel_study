import numpy as np
import torch

array = np.random.rand(3,4) # Creates 3x4 array with random values

print(f"Shape of  array: {array.shape}")
print(f"Number of dimensions: {array.ndim}")
print(f"Total elements: {array.size}")
print(f"Datatype of array: {array.dtype}")
print(f"Memory usage (bytes): {array.nbytes}")
print(f"Item size (bytes): {array.itemsize}")
print(f"Strides: {array.strides}")
print(f"Is C-contiguous: {array.flags.c_contiguous}")
print(f"Is Fortran-contiguous: {array.flags.f_contiguous}")

tensor = torch.from_numpy(array) # Convert numpy array to tensor

print(f"Shape of  tensor: {tensor.shape}")
print(f"Size of  tensor: {tensor.size()}")
print(f"Number of dimensions: {tensor.ndim}")
