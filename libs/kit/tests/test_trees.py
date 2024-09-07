from kit.bioinf import AA1_FULL

from kit.data.trees import PrefixTree


def test_seq1():
    seq = "MGNKWSKGWPAVRERMRKTEPAAVGVGAVSRDLERHGAITSSNTPATNADCAWLQAQEEEEEVGFPVRPQVPLRPMTFKGALDLSHFLKEKGGLEGLVYSQKRQDILDLWVNNTQGYFPDWQNYTQGPGIRYPLTFGWCFKLVPVDPEKIEEANEGENNSLLHPMSLHGMDDPEREVLVWKFDSRLAFHHVARELHPDYFKN"
    PrefixTree.set_alphabet(AA1_FULL)
    tree = PrefixTree()

    tree.add_seq(seq, 10)

    assert tree.has_kmer("MGNKWSKGWP")
    assert tree.has_kmer("VSRDLERHGA")
    assert tree.has_kmer("QAQEEE")
    assert tree.has_kmer("ALDLSHFLKE")
    assert tree.has_kmer("ANEGEN")
    assert tree.has_kmer("YFKN")
    assert tree.has_kmer("RELHPDYFKN")
    assert tree.has_kmer("ITSSNTPAT")
    assert tree.has_kmer("VGV")
    assert tree.has_kmer("EVGFPVRPQV")
    assert tree.has_kmer("EEEEE")

    assert not tree.has_kmer("MGNKWSKGWW")
    assert not tree.has_kmer("VSRDERHGA")
    assert not tree.has_kmer("VSDLERHGA")
    assert not tree.has_kmer("VSRDLERRGA")
    assert not tree.has_kmer("VSRDLERHGG")
    assert not tree.has_kmer("VSRDLERHGAA")
    assert not tree.has_kmer("VSRDLERHGAI")
    assert not tree.has_kmer("VSRDLEEHGA")
    assert not tree.has_kmer("EEEEEE")

def test_cnt_nodes():
    seq = "ABCBD"
    PrefixTree.set_alphabet("ABCDE")
    tree = PrefixTree()

    tree.add_seq(seq, 3)

    assert tree.cnt_nodes == 11
    assert tree.children[0].cnt_nodes == 3  # A, AB, ABC
    assert tree.children[1].cnt_nodes == 4  # B, BC, BCB, BD
    assert tree.children[2].cnt_nodes == 3  # C, CB, CBD
    assert tree.children[3].cnt_nodes == 1  # D


def test_max_depth():
    seq = "ABCBD"
    PrefixTree.set_alphabet("ABCDE")
    tree = PrefixTree()

    tree.add_seq(seq, 3)

    assert tree.max_depth == 3
    assert tree.children[0].max_depth == 2  # A, AB, ABC
    assert tree.children[1].max_depth == 2  # B, BC, BCB, BD
    assert tree.children[2].max_depth == 2  # C, CB, CBD
    assert tree.children[3].max_depth == 0  # D

    assert tree.children[1].children[2].max_depth == 1  # BC, BCB
    assert tree.children[1].children[3].max_depth == 0  # BD