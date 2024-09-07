import os
import shutil
import tempfile
import json
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from kit.bioinf.proteins import ProteinType
from kit.bioinf.immuno.mhc_1 import MHC_1_PEPTIDE_LENGTHS

from CAPE.MPNN.ProteinMPNN.protein_mpnn_utils import \
    gather_nodes, ProteinMPNN, cat_neighbors_nodes, \
    StructureDataset as StructureDatasetJSON
from CAPE.MPNN.ProteinMPNN.training.utils import StructureLoader
from CAPE.MPNN import run_parse_multiple_chains, run_make_tied_positions_dict
from CAPE.MPNN.overwrite import tied_featurize_original_modified as tied_featurize


REPO_PATH = None
MAX_SEQ_LEN = 10000
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
OMIT_AAs_NP = np.array([AA in "X" for AA in ALPHABET]).astype(np.float32)
BIAS_AAs_NP = np.zeros(len(ALPHABET))



def set_config(repo_path):
    global REPO_PATH
    REPO_PATH = repo_path


def sample_encoder(self, X, chain_encoding_all,
            residue_idx, mask=None, device="cpu"):
    # Prepare node and edge embeddings
    E, E_idx = self.actor.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)  # seem to be the node embeddings
    h_E = self.actor.W_e(E)  # seem to be the edge embeddings

    # Encoder is unmasked self-attention
    mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in self.actor.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    return h_V, h_E, E_idx


def get_decoding_order_and_masks(mask, chain_mask, randn, E_idx, tied_pos_list_of_lists_list=None, device="cpu"):
    decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))
    mask_size = E_idx.shape[1]

    permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()

    order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)

    # for tied positions in e.g. Homooligomers
    tied_decoding_order = None
    if tied_pos_list_of_lists_list is not None:
        tied_pos = tied_pos_list_of_lists_list[0]
        tied_decoding_order = []  # called new_decoding_order in the original code
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*tied_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    tied_decoding_order.append(list_a[0])
                else:
                    tied_decoding_order.append([t_dec])
        # decoding_order = torch.tensor(list(itertools.chain(*tied_decoding_order)), device=device)[None,].repeat(mask.shape[0],1)
    else:
        tied_decoding_order = [[int(p)] for p in decoding_order[0]]

    return tied_decoding_order, mask_bw, mask_fw


def get_in_proteome(seq, proteome_tree, max_checked_kmer_length=9):
    in_proteome = [1] * len(proteome_tree.alphabet)
    nodes = [None] * len(proteome_tree.alphabet)
    for candidate_i, candidate_aa in enumerate(proteome_tree.alphabet):
        kmer = candidate_aa
        node = proteome_tree.get_kmer(kmer)
        for t in seq[::-1]:
            kmer = t + kmer
            kmer_len = len(kmer)
            if kmer_len > max_checked_kmer_length:
                break
            _node = proteome_tree.get_kmer(kmer)

            if _node is None:
                break

            node = _node
            in_proteome[candidate_i] += 1

        nodes[candidate_i] = node

    return in_proteome, nodes


def get_is_presented(seq_kmers, alphabet, immuno_setup, predictor_setup, in_proteome=None):
    is_presented = [None] * len(alphabet)
    for candidate_i, candidate_aa in enumerate(alphabet):
        if in_proteome is not None and in_proteome[candidate_i]:
            continue
        is_presented[candidate_i] = False
        kmer = candidate_aa
        for t in seq_kmers[::-1]:
            kmer = t + kmer
            if len(kmer) > np.max(MHC_1_PEPTIDE_LENGTHS):
                break
            elif len(kmer) in MHC_1_PEPTIDE_LENGTHS:
                alleles = predictor_setup['mhc_1'].resolve_alleles(immuno_setup['mhc_1'])
                for allele in alleles:
                    predictor_setup['mhc_1'].predict_peptides([kmer], allele)
                    if predictor_setup['mhc_1'].peptide_presented(kmer, allele):
                        is_presented[candidate_i] = True

    return is_presented


def get_log_prob_adjustment(seq_kmers, proteome_tree, max_checked_kmer_length=9):
    kmer = ""
    adjustment = 0.
    if proteome_tree is not None:
        for t in seq_kmers[::-1]:
            kmer = t + kmer
            if len(kmer) > max_checked_kmer_length:
                break

            node = proteome_tree.get_kmer(kmer)  # the node in the proteome tree corresponding to the kmer
            if node is None:  # the kmer itself is not in the human proteome
                return -np.inf
            else:  # the kmer exists in the human proteome
                adjustment += np.log(node.cnt_nodes / proteome_tree.cnt_nodes)

    return adjustment


def sample_decoder_next(self, X, S_true, temperature, t_list, tied_beta,
        h_V, h_E, E_idx, mask, chain_mask,
        mask_fw, mask_bw,
        sampling_state, 
        alphabet,
        omit_AAs_np=None, bias_AAs_np=None, omit_AA_mask=None, bias_by_res=None,
        n_samples=1,
        max_checked_kmer_length=10,
        proteome_tree=None,
        n_most_likely_continuations=None,
        immuno_setup=None, predictor_setup=None, max_non_proteome=None, min_proteome_kmer_length=0, prob_factor=1.,
        device="cpu"):
    N_batch, N_nodes = X.size(0), X.size(1)
    if n_most_likely_continuations is None:
        n_most_likely_continuations = len(alphabet)

    assert N_batch == 1
    constant = torch.tensor(omit_AAs_np, device=device)  # shape = (21, ) - one for each AA. 1 if the AA should not be sampled
    constant_bias = torch.tensor(bias_AAs_np, device=device)
    omit_AA_mask_flag = omit_AA_mask != None
    
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)  # adds zeros to the last dimension of the final Edge encodings of the Encoder
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)  # adds the node encodings to the last dimension
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder  # masks all nodes that come AFTER the current node - BUT from the ENCODER
    
    if len(sampling_state) == 0:
        # output
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        all_log_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)  # the predicted sequence

        # technical
        h_S = torch.zeros_like(h_V, device=device)  # holds the input sequence embedding for the decoder
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.actor.decoder_layers))]
        acc_log_prob = 0.
        seq = [None] * X.shape[1]
        seq_kmers = ""
        non_proteome_aa = 0  # number of amino acids in the sequence that were added requireing a non-proteome kmer
    else:
        # output
        all_probs = sampling_state['all_probs']
        all_log_probs = sampling_state['all_log_probs']
        S = sampling_state['S']

        # technical
        h_S = sampling_state['h_S']
        h_V_stack = sampling_state['h_V_stack']
        acc_log_prob = sampling_state['acc_log_prob']
        seq = sampling_state['seq']
        seq_kmers = sampling_state['seq_kmers']
        non_proteome_aa = sampling_state['non_proteome_aa']
    
    # get adjustments for the probabilities based on the sequence
    in_proteome, is_presented = None, None
    if proteome_tree is not None:
        in_proteome, nodes = get_in_proteome(seq_kmers, proteome_tree, max_checked_kmer_length=max_checked_kmer_length)

    allow_non_presented_kmers = immuno_setup is not None and max_checked_kmer_length >= np.max(MHC_1_PEPTIDE_LENGTHS)
    if allow_non_presented_kmers:
        is_presented = get_is_presented(seq_kmers, alphabet, immuno_setup, predictor_setup, in_proteome=[x == max_checked_kmer_length for x in in_proteome])


    # actual sampling starts here
    #chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]).detach().clone().cpu() #[B,1]
    #mask_gathered = torch.gather(mask, 1, t[:,None]) #[B,1]
    #bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]

    # obtain the probabilities over the next amino acids
    log_probs = torch.zeros(1, 21)

    if bool(torch.all(mask[0, t_list] == 1.)):
        logits = 0.
        for t_ in t_list:
            if float(mask[0, t_]) > 0: # for not padded or missing regions
                bias_by_res_gathered = bias_by_res[:, t_, :] 

                # reduce to tensors only for the current node to be decoded
                E_idx_t = E_idx[:, t_:t_+1,:]  # get the neighbors of the current node    
                h_E_t = h_E[:, t_:t_+1,:,:]  # the Encoder produced edge encodings [N, 1, neighbors, 128]
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t) # concat with currently available residue encodings of the neighbors [N, 1, neighbors, 256]

                # only include the information from the encoder into h_ESV_t, where not more current information is already available from the decoder
                h_EXV_encoder_t = h_EXV_encoder_fw[:, t_:t_+1,:,:]

                mask_t = mask[:,t_:t_+1]
                for l, layer in enumerate(self.actor.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)  # edge, residue and node encoding
                    h_ESV_t = mask_bw[:,t_:t_+1,:,:]*h_ESV_decoder_t + h_EXV_encoder_t
                    
                    h_V_t = h_V_stack[l][:,t_:t_+1,:]

                    # update the node features (first element is the output of the Encoder)
                    h_V_stack[l+1][:,t_,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)

                # get last node feature of current node
                h_V_t = h_V_stack[-1][:,t_,:]

                # get the probability distribution for those over the amino acids
                logits += tied_beta[t_]*(self.actor.W_out(h_V_t) / temperature)/len(t_list)

        probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)
        if omit_AA_mask_flag:
            omit_AA_mask_gathered = omit_AA_mask[:,t_] #[B, 21]
            probs_masked = probs*(1.0-omit_AA_mask_gathered)
            probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]

        probs = probs.detach().clone().cpu()
        log_probs = torch.log(probs + 1e-20)

        # save results
        all_probs[:, t_] = probs
        all_log_probs[:, t_] = log_probs

        most_likely_continuations = probs[0, :].sort().indices[-n_most_likely_continuations:]
        for candidate_i, _ in enumerate(alphabet):
            _prob_factor = 1.
            if proteome_tree is not None:  # we check if kmers are in the proteome
                can_be_continued = nodes[candidate_i].cnt_nodes > 0 if nodes[candidate_i] is not None else False

                # if not all kmers are in the human proteome, the +1 is because the kmers are including the new AA
                if (in_proteome[candidate_i] < max_checked_kmer_length 
                and in_proteome[candidate_i] < (len(seq_kmers) + 1)):
                    _prob_factor = 0.
                    if (allow_non_presented_kmers 
                    and candidate_i in most_likely_continuations
                    and is_presented[candidate_i] == False  
                    and min_proteome_kmer_length <= in_proteome[candidate_i]
                    and non_proteome_aa + 1 <= max_non_proteome):
                        _prob_factor = prob_factor # 1.

                if not allow_non_presented_kmers and not can_be_continued:
                    _prob_factor = 0.
                
            probs[0, candidate_i] *= _prob_factor

        if probs[0, :].sum() != 0.:  
            probs[0, :] /= probs[0, :].sum()

    # actual sampling ends here
    sampling_states = []
    for n_sample in range(n_samples):
        _sampling_state = {}
        _sampling_state['all_probs'] = all_probs
        _sampling_state['all_log_probs'] = all_log_probs
        _sampling_state['h_V_stack'] = [h_V.detach().clone() for h_V in h_V_stack]

        _non_proteome_aa = non_proteome_aa
        fixed_position = False
        if bool(torch.any(mask[0, t_list] == 0.)):  # for padded or missing regions only we use the actual residue value
            assert mask[0, t_list[0]] == 0.
            fixed_position = True
            S_t = S_true[:, t_list[0]]
            log_prob = float(log_probs[0, int(S_t)])
        else:
            # sample an AA from the distribution
            if not torch.all(probs.isnan()):
                #S_t = torch.multinomial(probs, 1)
                S_t = torch.topk(probs, n_sample+1).indices[0, -1]
                if probs[0, S_t] == 0.:  # if the probability of this is zero (e.g. not in human proteome)
                    break
                log_prob = float(log_probs[0, int(S_t)])
                assert probs[0, int(S_t)] != float("-inf")
                if in_proteome is not None:
                    if in_proteome[int(S_t)] == False:
                        _non_proteome_aa += 1
                    else:
                        _non_proteome_aa = 0

            else:  
                # this beam ended outside the allowed kmers (human proteome, ...)
                # print(f"term: {seq_kmers}")
                S_t = torch.tensor(20)
                log_prob = float("-inf")

        new_token = alphabet[int(S_t)]
        # S_t = (_S_t*chain_mask[:, t_] + S_true[:, t_]*(1.0-chain_mask[:, t_])).long()
        _h_S = self.actor.W_s(S_t)

        # store the current output S_t into the result 
        _sampling_state['seq'] = seq.copy()
        for t_ in t_list:
            S[:, t_] = S_t
            _sampling_state['seq'][t_] = new_token

            # feed the current output S_t back into the decoder
            h_S[:, t_, :] = self.actor.W_s(S_t)  # encode
            
        
        _sampling_state['h_S'] = h_S.detach().clone()
        _sampling_state['S'] = S.detach().clone()
        _sampling_state['acc_log_prob'] = acc_log_prob + log_prob        
        _sampling_state['seq_kmers'] = seq_kmers + ("" if fixed_position else new_token)
        _sampling_state['non_proteome_aa'] = _non_proteome_aa

        sampling_states.append(_sampling_state)

    return sampling_states


def get_features_from_pdb(pdb_input_file_path, protein_type, device):
    tied_positions_dict = None

    
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        shutil.copy(pdb_input_file_path, os.path.join(tmp_dir_path, 'tmp.pdb'))
        parsed_file_path = run_parse_multiple_chains(REPO_PATH, tmp_dir_path)

        if protein_type == ProteinType.HOMOOLIGOMER:
            tied_pdbs_path = run_make_tied_positions_dict(REPO_PATH, tmp_dir_path, parsed_file_path)
            with open(tied_pdbs_path, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                tied_positions_dict = json.loads(json_str)

        dataset_valid = StructureDatasetJSON(
            parsed_file_path, 
            truncate=None, 
            max_length=MAX_SEQ_LEN, 
            verbose=False
        )

    data_loader_structure = StructureLoader(
        dataset_valid, 
        batch_size=MAX_SEQ_LEN
    )

    batch = next(iter(data_loader_structure))
    features = list(tied_featurize(batch, device, None, tied_positions_dict=tied_positions_dict))
    return features


def run_mpnn_beam_search(features_or_pdb_input_file_path, protein_mpnn, device, 
                         max_checked_kmer_length=10, 
                         proteome_tree=None, n_width=2, n_depth=10, 
                         n_most_likely_continuations=None,
                         immuno_setup=None, predictor_setup=None, max_non_proteome=None, min_proteome_kmer_length=0, prob_factor=1.,
                         chain_seqerator="/", protein_type=None,
                         return_features=False, return_final_sampling_state=False,
                         show_progress=False):
    temperature = 1.

    protein_mpnn.to(device)

    if isinstance(features_or_pdb_input_file_path, str):
        features = get_features_from_pdb(features_or_pdb_input_file_path, protein_type, device)
    else:
        features = features_or_pdb_input_file_path

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = features
    
    h_V, h_E, E_idx = sample_encoder(protein_mpnn, X, chain_encoding_all, residue_idx, mask=mask, device=device)

    # randn = torch.randn(chain_M.shape, device=X.device)
    randn = torch.arange(100, chain_M.shape[-1]+100)

    chain_mask = chain_M*chain_M_pos*mask

    tied_decoding_order, mask_bw, mask_fw = get_decoding_order_and_masks(
        mask, chain_mask, randn, E_idx, tied_pos_list_of_lists_list=tied_pos_list_of_lists_list, device=device
    )
    decoding_order = torch.tensor(list(itertools.chain(*tied_decoding_order)), device=device)[None,].repeat(1,1)

    sampling_states = None
    indices = None
    new_sampling_states = None

    t_list_idx = 0
    while t_list_idx < len(tied_decoding_order):
        prev_sampling_states = [{}] if sampling_states is None else sampling_states

        startup_phase = sampling_states is None or len(sampling_states) == 1
        for depth in range(n_depth + (1 if startup_phase else 0)):
            t_list = tied_decoding_order[t_list_idx]
            
            new_sampling_states = []
            for sampling_state in prev_sampling_states:
                _n_samples = 1 if int(mask[0, t_list].sum()) == 0 else n_width
                new_sampling_states += sample_decoder_next(protein_mpnn, X, S, temperature, t_list, tied_beta,
                                    h_V, h_E, E_idx, mask, chain_mask, 
                                    mask_fw, mask_bw,
                                    sampling_state, 
                                    ALPHABET,
                                    omit_AAs_np=OMIT_AAs_NP, bias_AAs_np=BIAS_AAs_NP, omit_AA_mask=omit_AA_mask, bias_by_res=bias_by_res_all,
                                    n_samples=_n_samples,
                                    max_checked_kmer_length=max_checked_kmer_length,
                                    proteome_tree=proteome_tree,
                                    n_most_likely_continuations=n_most_likely_continuations,
                                    immuno_setup=immuno_setup, predictor_setup=predictor_setup, 
                                    max_non_proteome=max_non_proteome, min_proteome_kmer_length=min_proteome_kmer_length,
                                    prob_factor=prob_factor,
                                    device=device
                )                    
            prev_sampling_states = new_sampling_states

            t_list_idx += 1
            if t_list_idx == len(tied_decoding_order):
                break

            t_list_prev = t_list
            t_list = tied_decoding_order[t_list_idx]

            # we switch to decoding a different chain
            if chain_encoding_all[0, t_list[0]] != chain_encoding_all[0, t_list_prev[0]]: 
                for prev_sampling_state in prev_sampling_states:
                    prev_sampling_state['seq_kmers'] = ""
                break

            if show_progress:
                print(f"\rt_list_idx={t_list_idx}     ", end="")


        acc_log_probs = [_sampling_state['acc_log_prob'] for _sampling_state in new_sampling_states]

        # in case we only consider human proteome kmers,
        # we consider the number of possible continuations for the pruning decision
        # otherwise it is just the model's log probability
        if immuno_setup is None:
            log_prob_adj = [
                get_log_prob_adjustment(_sampling_state['seq_kmers'], proteome_tree, max_checked_kmer_length=max_checked_kmer_length) 
                for _sampling_state in new_sampling_states
            ]
            scores = np.array(acc_log_probs) + np.array(log_prob_adj)
        else:
            scores = np.array(acc_log_probs)
        indices = np.argsort(scores)[-n_width:]

        _sampling_states = [new_sampling_states[idx] for idx in indices if scores[idx] != -np.inf]
        if len(_sampling_states) == 0:
            print("dead end")
            break
        sampling_states = _sampling_states

        seq_kmers_set = set([_sampling_state['seq_kmers'] for _sampling_state in sampling_states])
        if len(seq_kmers_set) == 1:
            sampling_states = [sampling_states[0]]

    final_sampling_state = prev_sampling_states[indices[-1]]
    _final_seq = final_sampling_state['seq']
    final_seq = ""
    chain = chain_encoding_all[0, 0]
    for p, t in enumerate(_final_seq):
        if chain != chain_encoding_all[0, p]:
            chain = chain_encoding_all[0, p]
            final_seq += chain_seqerator
        final_seq += t 

    result = [final_seq]
    if return_features:
        result.append(features)
    if return_final_sampling_state:
        result.append(final_sampling_state)

    return result
