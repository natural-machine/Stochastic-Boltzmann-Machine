import SBM.utils.utils as ut
import numpy as np
import shutil
import sys

fasta_file = sys.argv[1] 
output_name = sys.argv[2] 

shutil.copy(fasta_file, 'data/fasta/')
MSA = ut.load_fasta(fasta_file)
np.save(f'data/MSA_array/{output_name}.npy', MSA)

