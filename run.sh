# ./run.sh <cuda_id:0,1,...> <py_module> <args...>
args="${@:3}"

export PYTHONPATH=./src/py
export PYTHONHASHSEED=6993

if [ -e "$2" ]; then
  echo 'Running script'
  CUDA_VISIBLE_DEVICES=$1 python $2 $args
else
  echo 'Running module'
  CUDA_VISIBLE_DEVICES=$1 python -m $2 $args
fi
