from collections import defaultdict
from kit.bioinf import AA1_FULL, NEXT_CHAIN

def calc_seq_aa_cnts(seqs, standardize=False, aggregate=False):
    if isinstance(seqs, str):
        seqs = [seqs]

    if aggregate:
        seqs = [''.join([s for s in seqs])]

    aa_cnts = defaultdict(lambda: [0 for _ in seqs])
    for i, seq in enumerate(seqs):
        length = len(seq)
        for s in seq:
            assert s in AA1_FULL or s == NEXT_CHAIN
            aa_cnts[s][i] += 1

        if standardize:
            for aa, cnts in aa_cnts.items():
                aa_cnts[aa][i] = aa_cnts[aa][i]/length

    return aa_cnts