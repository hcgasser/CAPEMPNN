""" Utility functions for the loch module. """

import hashlib
from kit.hashes import str_to_hash

def chains_to_seq(chains):
    seq = None
    if isinstance(chains, dict):  # potential complex
        keys = sorted(chains)
        chains = [chains[k] for k in keys]
        seq = "/".join(chains)
    elif isinstance(chains, str):
        seq = chains
    else:
        raise Exception("'chains' wrong datatye")
    return seq

def get_seq_hash(chains, translate=("", "", "*-")):
    """Returns the SHA-256 hash of a sequence.

    :param seq: str - sequence
    :param translate: tuple - characters to translate
        the sequence with before its hash is computed
    :return hash_code: str - SHA-256 hash of the sequence
    """

    seq = chains_to_seq(chains)

    seq = seq.translate(str.maketrans(*translate))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq.encode("utf-8"))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code

def get_set_hash(seq_hashes):
    seq_hashes = sorted(seq_hashes)
    seq_hashes = ''.join(seq_hashes)
    return str_to_hash(seq_hashes)
