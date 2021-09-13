# ./run.sh <py_module> <args...>
args="${@:2}"

if [ -e "$1" ]; then
  echo 'Running script'
  export PYTHONPATH=./src/py && python $1 $args
else
  echo 'Running module'
  export PYTHONPATH=./src/py && python -m $1 $args
fi
