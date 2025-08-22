# import torch

# # GPU Information
# if torch.cuda.is.available():
#     print(f"현재 사용 가능: {torch.cuda.get_device_name(0)}")
#     print(f"현재 선택된 디바이스 : {device}")

# Scalar (0D tensor)
scalar = torch.tensor(5)
print("Scalar:", scalar)
print("Shape:", scalar.shape)
print("Dimenstions:", scalar.dim())

# Vector (1D tensor)
vector = torch.tensor([1,2,3,4])
print("Vector:", vector)
print("Shape:", vector.shape)
print("Dimensions:",vector.dim())

# Matrix (2D tensor)
matrix = torch.tensor([1,2], [3,4], [5,6])
print("Matrix:", matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.dim())

# Tensor (3D tensor)
tensor_3d = torch.tensor([[1,2], [3,4], [4,5]])
print("Matrix:", tensor_3d)
print("Shape:", tensor_3d.shape)
print("Dimensions:", tensor_3d.dim())

