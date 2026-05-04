#!/bin/bash
FAMILY=$1
FILE=$2
python3.11 scripts/SBM.py "${FAMILY}" data/MSA_array/${FILE} --TestTrain 0 --m 20 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero --lambdJ 0.001 --lambdh 0.001 --N_chains 150