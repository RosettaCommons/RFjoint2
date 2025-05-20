import torch
import torch.nn as nn
from Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from Track_module import IterativeSimulator
from AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork
from util import INIT_CRDS
from opt_einsum import contract as einsum

class RoseTTAFoldModule(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=8, n_ref_block=4,\
                 d_msa=256, d_msa_full=64, d_pair=128, d_templ=64,
                 n_head_msa=8, n_head_pair=4, n_head_templ=4,
                 d_hidden=32, d_hidden_templ=64,
                 p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(RoseTTAFoldModule, self).__init__()
        #
        # Input Embeddings
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, p_drop=p_drop)
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=23, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25)
        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param=SE3_param,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)

    def forward(self, msa_latent, msa_full, seq, xyz, state, alpha, idx, t1d=None, t2d=None, xyz_t=None,
                msa_prev=None, pair_prev=None, 
                return_raw=False, return_infer=False,
                use_checkpoint=False):
        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)
        #
        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
        msa_recycle, pair_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        #
        # add template embedding
        pair = self.templ_emb(t1d, t2d, xyz_t, pair, use_checkpoint=use_checkpoint)
        
        # Predict coordinates from given inputs
        msa, pair, R, T, alpha_s, lddt, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3],
                                                               state, alpha, idx,
                                                               use_checkpoint=use_checkpoint)

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        if return_raw:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, alpha_s[-1], \
                   logits_aa.reshape(B,-1,N,L)[:,:,0].permute(0,2,1), lddt.view(B,L,1)
        #
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)
        
        if return_infer:
            # get final bb structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return logits, logits_aa.reshape(B,-1,N,L)[:,:,0].permute(0,2,1), xyz, lddt.view(B, L), alpha_s[-1], msa[:,0], pair
        else:
            # get all intermediate bb structures
            xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)
            return logits, logits_aa, xyz, alpha_s, lddt.view(B, L)
