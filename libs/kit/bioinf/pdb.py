"""This module contains functions for working with PDB files"""

import os
import requests
from Bio import PDB
import py3Dmol

from kit.path import join
from kit.log import log_info
from kit.bioinf.proteins import structure_to_seq


def pdb_to_seqs(file_path, return_full=True, gaps='-', aa3_replace=None, aa_ids=[' ']):
    """reads a PDB file and returns the sequence"""
    parser = PDB.PDBParser()
    structure = parser.get_structure("X", file_path)
    return structure_to_seq(structure, return_full=return_full, gaps=gaps, aa3_replace=aa3_replace, aa_ids=aa_ids)


def download_pdb(pdb_id, output_dir, overwrite=False):
    """Downloads a PDB file from the RCSB PDB database"""
    pdbl = PDB.PDBList()

    output_file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    if overwrite or not os.path.exists(output_file_path):
        file_name = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")
        output_file_path = join(output_dir, file_name)

    if os.path.exists(output_file_path):
        pdb_file_path = join(output_dir, f"{pdb_id}.pdb")
        os.rename(output_file_path, pdb_file_path)
    else:
        raise Exception(f"Download {pdb_id} failed")

    return pdb_file_path

def get_similar_structures(pdb_id):
    # see: https://search.rcsb.org/
    r = {}
    url = f'https://search.rcsb.org/rcsbsearch/v2/query'

    payload = {
      "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "structure",
                "parameters": {
                  "value": {
                    "entry_id": pdb_id,
                    "assembly_id": "1"
                  },
                  "operator": "strict_shape_match"
                }
            }
        ]
      },
      "request_options": {
        "return_all_hits": True
      },
      "return_type": "entry"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        results = response.json()
        for entry in results['result_set']:
            r[entry['identifier']] = entry['score']
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    return r


def view_structure(pdb_file_path, res_colors=None, output_file_path=None, width=800, height=800):
    """Displays a 3D structure of a protein"""
    print(pdb_file_path)

    with open(pdb_file_path) as ifile:
        system = "".join([x for x in ifile])
   
    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(system)

    colors = [
        '#22FFFF',
        '#44BBBB',
        '#669999',
        '#887777',
        '#AA5555',
        '#CC3333',
        '#EE1111',
    ]

    i = 0
    for line in system.split("\n"):
        split = line.split()

        if len(split) == 0 or split[0] != "ATOM":
            continue

        if res_colors is not None:
            resid = int(split[5])
            color = res_colors[resid-1]
            view.setStyle({'model': -1, 'serial': i+1}, {"sphere": {'color': color}})
        else:
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {}})
        i += 1

    view.zoomTo()
    view.show()
    view.render(filename=output_file_path)

