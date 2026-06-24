#!/bin/bash
MODEL=$1
TEMP=$2
N_SEQS=$3
RUN=$4
NAME="$(basename "${MODEL%%_ModelSBM*}")"   
OUTDIR="$(dirname "$MODEL")${RUN:+/$RUN}"
OUT="$OUTDIR/${NAME}_T${TEMP}.fasta"
python3.11 -c "
import numpy as np, SBM.utils.utils as ut 
a = np.load('$MODEL', allow_pickle=True)[()]
seqs = ut.Create_modAlign(a, $N_SEQS, delta_t=100000, temperature = $TEMP)
CODE = '-ACDEFGHIKLMNPQRSTVWY'
with open('$OUT', 'w') as f:
        for i, s in enumerate(seqs):
                f.write(f'>seq{i+1}\n' + ''.join(CODE[int(j)] for j in s) + '\n')
"