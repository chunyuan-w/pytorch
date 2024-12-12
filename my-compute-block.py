def compute_qSplitSize(qSize):
    qSplitSize = 32
    if qSize >= 768:
        qSplitSize = 256
    elif qSize >= 192:
        qSplitSize = 64
    return min(qSplitSize, qSize)

def get_cur_kvSplitSize(kvSplitSize, kvSize, n, kvTail):
    cur_kvSplitSize = min(kvSplitSize, kvSize - n)
    return cur_kvSplitSize if cur_kvSplitSize == kvSplitSize else kvTail


query = torch.randn(10, 100, 20, 30)  # Example tensor
# TODO: how to get k and qSize?
k = 2
qSize = query.shape[1]
def get_cur_qSplitSize(qSize, k):
    qSplitSize = compute_qSplitSize(qSize)
    m = k * qSplitSize
    cur_qSplitSize = min(qSplitSize, qSize - m)
    return cur_qSplitSize






def compute_kvBlockSize(provided_kvBlockSize, key):
    return min(provided_kvBlockSize, key.shape[1])

def compute_kvSplitSize(qSize, kvBlockSize):
    kvSplitSize = 512
    if qSize >= 768:
        kvSplitSize = 512
    elif qSize >= 192:
        kvSplitSize = 512
    if kvBlockSize < kvSplitSize:
        kvSplitSize = kvBlockSize
    return kvSplitSize

def compute_use_kv_indice(kv_indices, block_num_kvi, batchSize_k):
    """
    Compute the value of use_kv_indice based on kv_indices, block_num_kvi, and batchSize_k.

    Parameters:
    kv_indices (numpy.ndarray): The kv indices tensor.
    block_num_kvi (int): The number of kv blocks.
    batchSize_k (int): The batch size for the key tensor.

    Returns:
    bool: The value of use_kv_indice.
    """
    block_num_kv_count, _ = compute_block_num_kv_count_and_zero_flag(kv_indices, block_num_kvi)

    use_kv_indice = False
    if block_num_kvi != block_num_kv_count and batchSize_k == 1:
        use_kv_indice = True

    return use_kv_indice

def compute_block_num_kv_count(kv_indices, block_num_kvi):
    """
    Compute the value of block_num_kv_count based on kv_indices and block_num_kvi.

    Parameters:
    kv_indices (numpy.ndarray): The kv indices tensor.
    block_num_kvi (int): The number of kv blocks.

    Returns:
    int: The value of block_num_kv_count.
    """
    block_num_kv_count, _ = compute_block_num_kv_count_and_zero_flag(kv_indices, block_num_kvi)
    return block_num_kv_count

def compute_kvSize(use_kv_indice, block_num_kv_count, kvBlockSize, kSize):
    if use_kv_indice:
        return block_num_kv_count * kvBlockSize
    else:
        return kSize

key = torch.randn(10, 512, 20, 30)  # Example tensor
batchSize_k = key.shape[0]
kSize = key.shape[1]
qSize = query.shape[1]
kv_indices = np.array([0, 1, 2, 0, 3, 4])  # Example kv indices tensor
def get_cur_kvSplitSize(kvBlockSize, qSize, kSize, batchSize_k, kv_indices, n):
    kvBlockSize = min(kvBlockSize, kSize)
    kvSplitSize = compute_kvSplitSize(qSize, kvBlockSize)
        
    block_num_kvi = len(kv_indices)
    block_num_kv_count = compute_block_num_kv_count(kv_indices, block_num_kvi)
    use_kv_indice = compute_use_kv_indice(kv_indices, block_num_kvi, batchSize_k)    
    
    kvSize = compute_kvSize(use_kv_indice, block_num_kv_count, kvBlockSize, kSize)
    kvTail = (kvSize - 1) % kvSplitSize + 1
    cur_kvSplitSize = min(kvSplitSize, kvSize - n)
    return cur_kvSplitSize if cur_kvSplitSize == kvSplitSize else kvTail
