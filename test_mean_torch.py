import torch
import numpy as np



vec = np.array([[[1., 2.], 
                 [3., 4.]], 
                [[1., 2.],
                 [3., 4.]]
                ])
vec_tensor = torch.tensor(vec)

print(torch.mean(vec_tensor, dim = 0))
print(torch.mean(vec_tensor, dim = 1))
