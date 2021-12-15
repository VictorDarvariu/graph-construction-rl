#! /bin/bash
source activate ucfadar-relnet

if [[ $# -eq 1 && $1 -eq "gpu" ]]; then
  # Need to build extensions on GPU machine as dependant on CUDA
  # as well as partition being mounted.
  cd /relnet/relnet/common && make
  cd /relnet/relnet/objective_functions && make
  cd /relnet
fi

celery -A tasks worker -Ofair --loglevel=debug --without-gossip --without-mingle &
tail -f /dev/null
