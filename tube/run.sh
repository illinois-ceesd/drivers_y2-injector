#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py tube.py -i run_params.yaml --log --lazy
#mpirun -n 1 python -u -O -m mpi4py tube.py -i run_params.yaml --log
