export LD_PRELOAD="/home/chunyuan/TE/jemalloc/lib/libjemalloc.so" 
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

if [ $# -lt 1 ]; then
  echo "usage: ./conv_eltwise_op_benchmark.sh [output_suffix_name]"
  exit
fi

FILE_SUFFIX=$1

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

### single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=$CORES $PREFIX python -m benchmarks.tensorexpr conv_eltwise  --device cpu --mode fwd --jit_mode trace --cpu_fusion 2>&1 | tee "1socket_$FILE_SUFFIX.log"

### single thread test
echo -e "\n### using OMP_NUM_THREADS=1"
PREFIX="numactl --physcpubind=0 --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=1 $PREFIX python -m benchmarks.tensorexpr conv_eltwise  --device cpu --mode fwd --jit_mode trace --cpu_fusion 2>&1 | tee "1thread_$FILE_SUFFIX.log"

### 4 thread test
echo -e "\n### using OMP_NUM_THREADS=4"
PREFIX="numactl --physcpubind=0-3 --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=4 $PREFIX python -m benchmarks.tensorexpr conv_eltwise  --device cpu --mode fwd --jit_mode trace --cpu_fusion 2>&1 | tee "4thread_$FILE_SUFFIX.log"
