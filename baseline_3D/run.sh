#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py injector.py -i run_params.yaml --log --lazy
#mpirun -n 1 python -u -O -m mpi4py injector.py -i run_params.yaml --log --lazy --overintegration
#mpirun -n 1 python -u -O -m mpi4py injector.py -i run_params.yaml --log