import torch
import numpy as np

print("=" * 50)
print("Part One: Create Tensor (5 most common ways)")
print("=" * 50)

# 1. From Python list
print("\n1. creating list：")
shopping_list = [1, 2, 3, 4, 5]
tensor_from_list = torch.tensor(shopping_list)
print(f"Shoping list tensor: {tensor_from_list}")

# 2. All 0/1 (initialization)
print("\n2. All 0 or 1:")
zeros_tensor = torch.zeros(3,4) #3 row, 4 col of all 0 list
ones_tensor = torch.ones(2,3) #2 row, 3 col of all 1 list
print(f"ALL ZERO:\n {zeros_tensor}")
print(f"ALL ONE:\n {ones_tensor}")

# 3. Random numbers (initialize neural network)
print("\n3. Random tensor: ")
random_tensor = torch.rand(2, 3) # 0~1 equally distributed
print(f"Random list:\n {random_tensor}") # you can dirctly do +-*/ operations on rand()

# 4. Similar tensor
print("\n4. Similar tensor: ")
like_zeros = torch.zeros_like(random_tensor)
print(f"Have same size as random_tensor with all 0:\n {like_zeros}")

# 5. From NumPy
print("\n5. NumPy switch: ")
numpy_array = np.array([[1,2],[3,4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"NumPy list:\n {numpy_array}")
print(f"After switch to tensor:\n {tensor_from_numpy}")

print("\n" + "=" * 50)
print("Part Two: Operation on Tensor's Shape")
print("=" * 50)

matrix = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
])

print(f"Original shape: {matrix.shape}")
print(matrix)

# 1. Check Matrix Info
print(f"\nShape Info: {matrix.shape}")
print(f"Dim: {matrix.dim()}")
print(f"Number of elements: {matrix.numel()}")

# 2. View/Reshape
print("\nChange Shape: ")
reshaped = matrix.view(2,6)
print(f"Change to:\n {reshaped}")

flattened = matrix.view(-1) # Change to dim 1
print(f"Reshape to dim 1: {flattened}")

# 3. Transpose
print("\nTransposed: ")
transposed = matrix.T
print(f"After transpose shape={transposed.shape}:\n{transposed}")

# 4. Increase/Decrease Dim
print("\nDim Operation: ")
# Add a batch dimension
# From (3,4) to (1,3,4)
with_batch = matrix.unsqueeze(0)
print(f"After add dim shape={with_batch.shape}")

# Decrease Dim
without_batch = with_batch.squeeze(0)
print(f"After reduce dim shape={without_batch.shape}")

print("\n" + "=" * 50)
print("Part Three: Tensor Operations")
print("=" * 50)

a = torch.tensor([1,2,3], dtype=torch.float32)
b = torch.tensor([4,5,6], dtype=torch.float32)

print(f"a = {a}")
print(f"b = {b}")

# Basic operations
print(f"\nBasic operations:")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# Matrix multipulication
print(f"\nMatrix multipulication: ")
matrix_a = torch.tensor([[1,2],[3,4]])
matrix_b = torch.tensor([[5,6],[7,8]])
print(f"matrix_a:\n{matrix_a}")
print(f"matrix_b:\n{matrix_b}")
print(f"Result of multipulication:\n{torch.matmul(matrix_a, matrix_b)}")
print(f"Shorter version of multipulication:\n{matrix_a @ matrix_b}")

print("\n" + "=" * 50)
print("Part Four: Real life scenario")
print("=" * 50)

# Scenario one: Process image
print("\nScenario one: Process image")
image = torch.randn(1,28,28) # one 28x28 image
print(f"image tensor shape: {image.shape}")
print(f"image data type: {image.dtype}")

# Scenario two: Process a batch of data
print("\nScenario two: Process a batch")
batch_size = 32
batch_images = torch.randn(batch_size, 3, 224, 224)
print(f"batch shape: {batch_images.shape}")
print(f"Meaning: {batch_size} images, 3 colour channal, 224x224 resolution")

# Scenario three: A simple neural network
print("\nScenario three: Single-layer neural network")
input_data = torch.randn(10,5) # 10 samples, each with 5 features
weight = torch.randn(5,3) # weight matrix: 5 input features -> 3 output features 
bias = torch.randn(3) # bias

output = input_data @ weight + bias
print(f"input: {input_data.shape}")
print(f"weight: {weight.shape}")
print(f"output: {output.shape}")
print("This is neural network linear calculation")
