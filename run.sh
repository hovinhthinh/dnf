# ./run.sh <py_module> <args...>
args="${@:2}"

export PYTHONPATH=./src/py
export PYTHONHASHSEED=6993

if [ -e "$1" ]; then
  echo 'Running script'
  python $1 $args
else
  echo 'Running module'
  python -m $1 $args
fi
