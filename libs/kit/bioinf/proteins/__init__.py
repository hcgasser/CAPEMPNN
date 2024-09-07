from enum import Enum
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder
import numpy as np

from kit.path import join


class Protein:
    def __init__(self):
        self.chains = None
        self.structure = None
        self.phi_psi = None

    def from_pdb(self, pdb_file_path):
        pdb_parser = PDBParser(QUIET=True)
        self.structure = pdb_parser.get_structure('protein', pdb_file_path)
        seq = structure_to_seq(self.structure)
        assert len(seq) == 1
        self.chains = seq[0]
        self.phi_psi = calc_phi_psi(self.structure)

    def plot_ramachandran(self):
        assert self.phi_psi is not None
        phi = [np.degrees(angle[0]) for angle in self.phi_psi if angle[0] is not None]
        psi = [np.degrees(angle[1]) for angle in self.phi_psi if angle[1] is not None]

        plt.figure()
        plt.scatter(phi, psi, marker='.', color='blue')
        plt.xlabel('Phi')
        plt.ylabel('Psi')
        plt.title('Ramachandran Plot')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.show()


def calc_phi_psi(structure):
    phi_psi = []
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        phi_psi.extend(pp.get_phi_psi_list())
    return phi_psi


def structure_to_seq(structure, return_full=True, gaps='-', aa3_replace=None, aa_ids=[' ']):
    seqs_mod = []

    model = structure[0]
    for model in structure:  # each model could be a different confirmation of the molecule
        # (also NMR Nuclear Magnetic Resonance produces multiple models)

        residues = {}
        for chain in model:
            seq = ""
            residues[chain.id] = {}
            for residue in chain:
                if residue.id[0] in aa_ids:  # ignore heteroatoms
                    residues[chain.id][residue.id[1]] = residue.get_resname()

        seqs = {}
        for chain_id, chain_dict in residues.items():
            seq = ""
            chain_dict = dict(sorted(chain_dict.items()))
            _tmp = list(chain_dict)
            rmin, rmax = _tmp[0], _tmp[-1]
            seq_all = [gaps] * (rmax - rmin + 1)
            for pos, res3 in chain_dict.items():
                if aa3_replace is not None and res3 in aa3_replace:
                    res3 = aa3_replace[res3]
                if res3 in PDB.Polypeptide.standard_aa_names:
                    res1 = PDB.Polypeptide.protein_letters_3to1[res3]
                    seq += res1
                    seq_all[pos - rmin] = res1
                elif res3 in ['UNK']:
                    seq_all[pos - rmin] = 'X'
                elif seq not in ("", None):
                    raise Exception(f"unknown residue {res3}")
                else:
                    seq = None

            if seq is not None:
                seqs[chain_id] = ''.join(seq_all) if return_full else seq
        seqs_mod.append(seqs)
    return seqs_mod


class ProteinType(Enum):
    MONOMER = 0
    HOMOOLIGOMER = 1
    COMPLEX = 2

    def pdb_file_path(self, pdb_dir_path, pdb_id=None, ckpt_id=None):
        sub_dir = ""
        if self == self.MONOMER:
            sub_dir = "monomers"
        elif self == self.HOMOOLIGOMER:
            sub_dir = "homooligomers"
        elif self == self.COMPLEX:
            sub_dir = "complexes"
        type_dir_path = join(pdb_dir_path, sub_dir)
        ckpt_dir_path = join(type_dir_path, ckpt_id) if ckpt_id is not None else type_dir_path
        return join(ckpt_dir_path, f"{pdb_id}.pdb") if pdb_id is not None else ckpt_dir_path

    def __str__(self):
        return self.to_text()

    def to_text(self, singular=True):
        if self == self.MONOMER:
            return "monomer" if singular else "monomers"
        elif self == self.HOMOOLIGOMER:
            return "homooligomer" if singular else "homooligomers"
        elif self == self.COMPLEX:
            return "complex" if singular else "complexes"