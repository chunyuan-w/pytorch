import torch
import torch._inductor.config

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._inductor import config as inductor_config


def fn(qBlockSize, rkvBlockSize, i, j, k, n, qk_data):
    for row in range(0, qBlockSize):
        for col in range(0, rkvBlockSize):
            b_ = i
            h_ = j
            q_ = k * qBlockSize + row
            k_ = n + col

            # in_ptr0 = qk_data + row * rkvBlockSize + col
            in_ptr0 = qk_data[row][col]
            in_ptr1 = b_
            in_ptr2 = h_
            in_ptr3 = q_
            in_ptr4 = k_
            out_ptr0 = in_ptr0

            tmp0 = in_ptr0
            tmp1 = in_ptr3
            tmp2 = in_ptr4
            tmp3 = tmp1 - tmp2
            tmp4 = tmp3
            tmp5 = tmp0 + tmp4
            qk_data[row][col] = tmp5

def fn_tensor(qBlockSize, rkvBlockSize, i, j, k, n, qk_data, row_indices, col_indices):
    # Compute q_ and k_
    q_ = k * qBlockSize + row_indices  # Shape (qBlockSize, 1), broadcasts along columns
    k_ = n + col_indices  # Shape (1, rkvBlockSize), broadcasts along rows

    # Perform the operations
    tmp3 = q_ - k_  # Broadcasting subtraction, shape (qBlockSize, rkvBlockSize)
    tmp4 = tmp3.float()  # Convert to float if needed
    qk_data += tmp4  # Add the result to qk_data in place    
    return qk_data

qBlockSize = 32
rkvBlockSize = 16
i = 4
j = 6
k = 2
n = 0

qk = torch.randn(qBlockSize, rkvBlockSize)
qk_data = qk


# Create tensor representations of indices
row_indices = torch.arange(qBlockSize).unsqueeze(1)  # Shape (qBlockSize, 1)
col_indices = torch.arange(rkvBlockSize).unsqueeze(0)  # Shape (1, rkvBlockSize)

# print(qk_data)
# fn(qBlockSize, rkvBlockSize, i, j, k, n, qk_data)
# print(qk_data)

# with torch.no_grad():
#     compiled = torch.compile(fn)
#     compiled(qBlockSize, rkvBlockSize, i, j, k, n, qk_data)


print(qk_data)
fn_tensor(qBlockSize, rkvBlockSize, i, j, k, n, qk_data, row_indices, col_indices)
print(qk_data)

with torch.no_grad():
    compiled = torch.compile(fn_tensor)
    compiled(qBlockSize, rkvBlockSize, i, j, k, n, qk_data, row_indices, col_indices)


