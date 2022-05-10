# export OMP_NUM_THREADS=56

export LD_PRELOAD="/home_local/chunyuan/code/jemalloc/lib/libjemalloc.so" 
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

python -m benchmarks.tensorexpr conv_eltwise  --device cpu --mode fwd --jit_mode trace --cpu_fusion
