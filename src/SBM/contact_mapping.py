import numpy as np 
from Bio.PDB import PDBParser, MMCIFParser
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from mysca.results import PreprocessingResults
from Bio.SeqUtils import seq1



def contact_indices_union(pdb_path, preproc_path, ref_id, chain="A", msa_name="msa_orig.fasta-aln"):
    """
    map processed sequence (typically from sca) to sequence from structure via the full reference sequence 
    """
    
    #processed sequence --> reference residue positions 
    h = next(str(r.seq) for r in SeqIO.parse(
        f'{preproc_path}/{msa_name}','fasta') if ref_id in r.id) #original seq (with gaps)
    
    rp = np.asarray(PreprocessingResults.load(preproc_path).retained_positions) #retained pos
    
    isres, ref_seq = np.array(list(h)) != '-', h.replace("-", "")
    pos_at = np.cumsum(isres)
    proc_to_ref = np.array([pos_at[c] if isres[c] else -1 for c in rp])   # maps residue indices from processed 

    #structure seq --> reference seq 
    parser = MMCIFParser() if pdb_path.endswith((".cif", ".mmcif")) else PDBParser()
    struct = [(int(r.id[1]), seq1(r.resname, undef_code="X"))
              for r in parser.get_structure('F', pdb_path)[0][chain] if r.has_id('CA')]

    alignment_hits = [ref_seq[n - 1] == a for n, a in struct if 1 <= n <= len(ref_seq)] #check author residue information 
    if alignment_hits and sum(alignment_hits) / len(alignment_hits) >= 0.95: #use author reference is they line up >95% 
        to_pdb = {n: i for i, (n, a) in enumerate(struct)}
    else: 
        aln = PairwiseAligner().align(ref_seq, "".join(a for _, a in struct))[0]
        to_pdb = {r + 1:s 
                    for (rs, re), (ss, se) in zip(*aln.aligned) 
                    for r, s in zip(range(rs, re), range(ss, se))}


    to_dca = {int(p): c for c, p in enumerate(proc_to_ref) if p > 0}
    union  = np.array(sorted(set(to_pdb) | set(to_dca)))
    pdb_idx  = np.array([to_pdb.get(p, -1)  for p in union])
    dca_idx = np.array([to_dca.get(p, -1) for p in union])
    
    return union, ref_seq, pdb_idx, dca_idx

def display_alignment(ref_seq, union, pdb_idx, dca_idx):
    """
    visualize alignment between reference sequence, processed sequence, and sequence pbd structure 
    """
    struct_pos = set(union[pdb_idx  >= 0])      # positions resolved in the structure
    proc_pos   = set(union[dca_idx >= 0])      # positions carried by a DCA column
    struct = "".join(a if (i+1) in struct_pos else "-" for i, a in enumerate(ref_seq))
    proc   = "".join(a if (i+1) in proc_pos   else "-" for i, a in enumerate(ref_seq))

    for b in range(0, len(ref_seq), 60):
        print(f"{b+1:>4} orig   {ref_seq[b:b+60]}")
        print(f"     struct {struct[b:b+60]}")
        print(f"     proc   {proc[b:b+60]}\n")
