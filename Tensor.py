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