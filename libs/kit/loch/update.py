import os
import shutil

from kit.loch.utils import get_seq_hash, chains_to_seq
from kit.bioinf.fasta import seqs_to_fasta
from kit.loch.path import get_fasta_file_path, get_pdb_file_path, get_function_path, get_md_path

from kit.bioinf.pdb import pdb_to_seqs
from kit.bioinf.fasta import fastas_to_seqs


PREDICTOR_STRUCTURE_NAMES = ["AF", "exp"]


class Loch:
    def __init__(self, loch_path=None):
        self.loch_path = loch_path

    def add_seq(self, seq):
        return add_seq(seq, loch_path=self.loch_path)
    
    def add_structure(self, seq_hash, pdb_file_path, predictor_structure_name):
        loch_pdb_file_path = get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)
        if not os.path.exists(loch_pdb_file_path):
            shutil.copy(pdb_file_path, loch_pdb_file_path)
        return loch_pdb_file_path

    def add_entry(self, seq=None, 
                  pdb_file_path=None, model_nr=0, predictor_structure_name='exp', pdb_to_seqs_kwargs=None):
        if seq is not None:
            seq_hash = add_seq(seq, loch_path=self.loch_path)

        if pdb_file_path is not None:
            if pdb_to_seqs_kwargs is not None:
                models = pdb_to_seqs(pdb_file_path, return_full=True, **pdb_to_seqs_kwargs)
            else:
                models = pdb_to_seqs(pdb_file_path, return_full=True)
            chains = models[model_nr]
            pdb_seq = chains_to_seq(chains)
            if seq is not None:
                assert seq == pdb_seq or seq == pdb_seq.replace("-", "X")
            seq_hash = get_seq_hash(seq)
            loch_pdb_file_path = get_pdb_file_path(seq_hash, predictor_structure_name=predictor_structure_name, loch_path=self.loch_path)
            if not os.path.exists(loch_pdb_file_path):
                shutil.copy(pdb_file_path, loch_pdb_file_path)

        return seq_hash

    def rm_entry(self, seq_hash):
        rm_seq(seq_hash, loch_path=self.loch_path)

    def get_seq(self, seq_hash):
        fasta_file_path = get_fasta_file_path(seq_hash, loch_path=self.loch_path)
        return fastas_to_seqs(fasta_file_path, stop_token=False)


def add_seq(seq, loch_path=None):
    seq_hash = get_seq_hash(seq)
    fasta_file_path = get_fasta_file_path(seq_hash, loch_path=loch_path)
    seqs_to_fasta(seq, fasta_file_path)
    return seq_hash

def rm_seq(seq_hash, loch_path=None):
    fasta_file_path = get_fasta_file_path(seq_hash, loch_path=loch_path)
    if os.path.exists(fasta_file_path):
        os.remove(fasta_file_path)

    for predictor_structure_name in PREDICTOR_STRUCTURE_NAMES:
        pdb_file_path = get_pdb_file_path(seq_hash, loch_path=loch_path, predictor_structure_name=predictor_structure_name)
        if os.path.exists(pdb_file_path):
            os.remove(pdb_file_path)

    function_path = get_function_path(seq_hash, loch_path=loch_path)
    if os.path.exists(function_path):
        os.remove(function_path)

    md_path = get_md_path(seq_hash, loch_path=loch_path)
    if os.path.exists(md_path):
        shutil.rmtree(md_path)
