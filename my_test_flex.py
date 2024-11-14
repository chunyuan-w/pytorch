import torch

import torch._inductor.config



from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from torch._inductor import config as inductor_config

torch.manual_seed(122)

import functools



if __name__ == "__main__":



    seq_lens= {

        "o96" : 128,

        "o32" : 5248,

    }

    # inductor_config.cpu_backend="triton"

    num_channels=256 #1024 works, 256 doesnt at fp16 on 16 heads

    num_heads=4

    head_dim= num_channels // num_heads



    B, H, SEQ_LEN, HEAD_DIM = 1, num_heads, seq_lens['o96'], head_dim

    WINDOW_SIZE = 512

    PRECISION=torch.float32

    DEVICE="cpu"



    FORWARD_ONLY=True



    def make_tensor():

        return torch.randn(B, SEQ_LEN, H,  HEAD_DIM, device=DEVICE, dtype=PRECISION, requires_grad=False)

    # TODO: This is the score function
    # score, q_idx, kv_idx are all scalars
    def relative_positional(score, b, h, q_idx, kv_idx):

        return score + (q_idx - kv_idx)

    q, k, v = make_tensor(), make_tensor(), make_tensor()

    q = q.transpose(1,2)

    import copy

    q_ = copy.deepcopy(q)

    k = k.transpose(1,2)

    k_ = copy.deepcopy(k)

    v = v.transpose(1,2)

    v_ = copy.deepcopy(v)

    gradOut = torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=PRECISION)



    # def sliding_window(b, h, q_idx, kv_idx):

    #     return torch.abs(q_idx - kv_idx) <= WINDOW_SIZE

    def causal_mask(b, h, q_idx, kv_idx):

        return q_idx >= kv_idx



    block_mask = create_block_mask(causal_mask, B, H, seq_lens['o96'], seq_lens['o96'], device="cpu")

    # breakpoint()

    # block_mask = create_block_mask(

    #     sliding_window, B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, _compile=True, device=DEVICE



    # )

    # res_ref = q @ k

    attention = functools.partial(flex_attention, block_mask=block_mask, score_mod = relative_positional) #cache the block mask so its not remade

    # print(torch._dynamo.list_backends())

    with torch.no_grad(), torch.cpu.amp.autocast():

        attention = torch.compile(attention)

        print(f"Compiled correctly")



        # errors here, during initial autotuning

        out = attention(q, k, v)

        out = attention(q, k, v)

        # out = attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))#block_mask=block_mask)



        # out = attention(q, k, v)

        print(f"Shape of output tensor: {list(out.shape)}")



        attn_weights = torch.matmul(q_, k_.transpose(2, 3))  * 0.125

        # upcast attention to fp32

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_output = torch.matmul(attn_weights, v_)

        # breakpoint()

    # if (not FORWARD_ONLY):

    #     out.backward(gradOut, retain_graph=True)

    #     print(f"Shape of output tensor after bw: {list(out.shape)}")
print("#" * 50)












#   for (int64_t n = 0; n < num_keys; n += kvSplitSize) {
#     int64_t kvBlockSize = std::min(kvSplitSize, kvSize - n);
#     int64_t ekvBlockSize = kvBlockSize;
#     int64_t rkvBlockSize = kvBlockSize == kvSplitSize ? rkvSplitSize : rkvTail;
#     // Calculate scale * q @ k.T
#     at::native::cpublas::gemm(
#       at::native::TransposeType::Transpose,
#       at::native::TransposeType::NoTranspose,
#       kvBlockSize,
#       qBlockSize,
#       headSize,
#       static_cast<accum_t>(1),
#       k_data + i * kStrideB + j * kStrideH +
#           n * kStrideN,
#       kStrideN,
#       q_data + i * qStrideB + j * qStrideH +
#           m * qStrideM,
#       qStrideM,
#       static_cast<accum_t>(0),
#       qk_data,
#       kvBlockSize);
    

# TODO: for qk_data, we will pick up the right position
#       to get a scalar value and apply the score func.


#     qk_data_scalar = qk_data[i * exxx+ j]


# TODO: the input is qk_data_scalar and we 
#       return it after having applied the score func

#     qk_data_scalar = qk_data_scalar + (a - b)
#     return qk_data_scalar


#     // Apply causal mask, fill unused with -inf
#     if (is_causal && num_keys - n <= kvSplitSize) {
#       for (const auto row : c10::irange(qBlockSize)) {
#         int64_t last_col = m + row - n;
#         accum_t* row_ptr = qk_data + row * rkvBlockSize;
#         fill_stub(row_ptr + last_col + 1,
#             -std::numeric_limits<accum_t>::infinity(),
#             kvBlockSize - last_col - 1);
#       }
#     }