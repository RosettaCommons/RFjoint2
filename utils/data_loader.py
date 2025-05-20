import math
import os
import csv
import random
import torch
from torch.utils import data
import numpy as np
from dateutil import parser
from parsers import parse_a3m, parse_pdb
from mask_generator import generate_masks
import util
ic.configureOutput(includeContext=True)

base_dir = "/projects/ml/TrRosetta/PDB30-20FEB17"
base_torch_dir = base_dir
fb_dir = "/projects/ml/TrRosetta/fb_af"
pdb_ss_torch_dir = "/home/jwatson3/torch/"
#fb_ss_torch_dir = "/home/jwatson3/torch/fb_ss/"
fb_ss_torch_dir = "/net/scratch/jwatson3/aws_training/ss/"
if not os.path.exists(base_dir):
    # training on AWS
    base_dir = "/data/databases/PDB30-20FEB17"
    fb_dir = "/data/databases/fb_af"
    base_torch_dir = base_dir
    pdb_ss_torch_dir = "%s/torch/"%base_dir
    fb_ss_torch_dir = "%s/ss/"%fb_dir
def set_data_loader_params(args):
    PARAMS = {
        "LIST"    : "%s/list.no_denovo.csv"%base_dir,
        "FB_LIST" : "%s/list_b1-3.csv"%fb_dir,
        "VAL"     : "%s/val_lists/xaa"%base_dir,
        "STUB_LIST"    : '/home/ahern/data/stub_dataset.csv',
        "STUB_VAL"    : '/home/ahern/data/stub_validation_set.txt',
        "DIR"     : base_torch_dir,
        "FB_DIR"  : fb_dir,
        "SS_TORCH_DIR" : pdb_ss_torch_dir,
        "FB_SS_TORCH_DIR" : fb_ss_torch_dir,
        "USE_FB" : False,
        "MINTPLT" : 0,
        "MAXTPLT" : 5,
        "MAXSEQ"  : 1024,
        "MAXLAT"  : 128, 
        "LMIN"    : 200,
        "LMAX"    : 260,
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : 3.5,
        "PLDDTCUT": 70.0,
        "SLICE"   : "CONT",
        "ROWS"    : 1,
        "seqID"   : 150.0,
        "MAXCYCLE": 4,
        "EPOCH_SIZE" : 4048,
        "HAL_MASK_HIGH": 35, #added by JW for hal masking
        "HAL_MASK_LOW": 10,
        "FLANK_HIGH": 6,
        "FLANK_LOW" : 3, 
        "MASK_FRAC" : 0.15, #added as loader param
        "STR2SEQ_FULL_LOW" : 0.9, 
        "STR2SEQ_FULL_HIGH" : 1.0, 
        "CONSTANT_RECYCLE" : True,
        "DEBUG": False,
        "DEBUG_N_EXAMPLES": -1
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            if str(getattr(args, param.lower())) == 'USE_DEFAULT':
                continue
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Input: MSA having shape (N, L)
    output: new MSA with block deletion
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params={'MAXLAT': 256, 'MAXSEQ': 1024, 'MAXCYCLE': 10}, eps=1e-6, p_mask=0.15, mut_pos=None):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
    '''
    N, L = msa.shape
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = min(N, params['MAXLAT'])
    #Nextra = min(N, params['MAXSEQ'])
    Nextra = N - Nclust
    if Nextra < 1:
        Nextra = 1
    Nextra = min(Nextra, params['MAXSEQ'])
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample = torch.randperm(N-1, device=msa.device)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        
        if mut_pos != None:
            # mask given position in the query sequence
            msa_masked[0, mut_pos-1] = 21 # mask to mask-token
            mask_pos[0, mut_pos-1] = True
        b_seq.append(msa_masked[0].clone())
        
        # get extra sequenes
        if Nclust < N: # there are extra sequences
            msa_extra = msa[1:,:][sample[Nclust-1:]]
            ins_extra = ins[1:,:][sample[Nclust-1:]]
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        else:
            msa_extra = msa_masked[:1]
            ins_extra = ins_clust[:1]
            extra_mask = mask_pos[:1]
        N_extra = msa_extra.shape[0]
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20).float() # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20).float() 
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T)
        assignment = torch.argmax(agreement, dim=-1)

        # seed MSA features
        # 1. one_hot encoded aatype: msa_clust_onehot
        # 2. cluster profile
        count_extra = ~extra_mask
        count_clust = ~mask_pos
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L)
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]
        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        #
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust), dim=-1)

        # extra MSA features
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None]), dim=-1)

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturize(tplt, qlen, params, pick_top=False):
    seqID_cut = params['seqID']
    if seqID_cut <= 100.0:
        sel = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[sel]
        tplt['qmap'] = tplt['qmap'][:,sel]
        tplt['xyz'] = tplt['xyz'][:, sel]
        tplt['seq'] = tplt['seq'][:, sel]
        tplt['f1d'] = tplt['f1d'][:, sel]

    ntplt = len(tplt['ids'])
    if ntplt<1: # no templates
        xyz = torch.full((1,qlen,14,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        return xyz, t1d
   
    npick = np.random.randint(params['MINTPLT'], min(ntplt, params['MAXTPLT'])+1)
    if npick < 1: # no template
        xyz = torch.full((1,qlen,14,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        return xyz, t1d

    if not pick_top:
        sample = torch.randperm(ntplt)[:npick]
    else:
        sample = torch.arange(npick)

    xyz = torch.full((npick,qlen,14,3),np.nan).float()
    mask = torch.full((npick,qlen,14),False)
    t1d = torch.full((npick, qlen), 20).long()
    t1d_val = torch.zeros((npick, qlen, 1)).float()

    for i,nt in enumerate(sample):
        sel = torch.where(tplt['qmap'][0,:,1]==nt)[0]
        pos = tplt['qmap'][0,sel,0]
        xyz[i,pos] = tplt['xyz'][0,sel]
        mask[i,pos] = tplt['mask'][0,sel]
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2].unsqueeze(-1) # alignment confidence

    t1d = torch.nn.functional.one_hot(t1d, num_classes=21).float()
    t1d = torch.cat((t1d, t1d_val), dim=-1)

    xyz = torch.where(mask[...,None], xyz.float(),torch.full((npick,qlen,14,3),np.nan).float())

    return xyz, t1d

def MSAFeaturize_fixbb(msa, params):
    '''
    Input: full msa information
    Output: Single sequence, with some percentage of amino acids mutated (but no resides 'masked')
    
    This is modified from autofold2, to remove mutations of the single sequence
    '''
    N, L = msa.shape

    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0)

    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        assert torch.max(msa) < 22
        msa_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=22)
        msa_fakeprofile_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=24) #add the extra two indel planes, which will be set to zero
        msa_full_onehot = torch.cat((msa_onehot, msa_fakeprofile_onehot), dim=-1)
        
        #make fake msa_extra
        msa_extra_onehot = torch.nn.functional.one_hot(msa[:1],num_classes=23)
        
        #make fake msa_clust and mask_pos
        msa_clust = msa[:1]
        mask_pos = torch.full_like(msa_clust, 1).bool()
        '''
        sample = torch.randperm(N-1, device=msa.device)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        # 5% (or params['FIX_BB_MUT_FRAC'] random masking. Note only the 20% listed below are applied now (mask token added in training loop).
        # - 33%: aa replaced with a uniformly sampled random amino acid
        # - 33%: aa replaced with an amino acid sampled from the MSA profile
        # - 34%: not replaced but scored
        random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.33*random_aa + 0.33*raw_profile + 0.34*same_aa

        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < params['FIX_BB_MUT_FRAC']
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone()) #masked single sequence
        
        msa_masked_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22)
        msa_fakeprofile_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22)
        temp = torch.full_like(msa_masked_onehot, 0)
        temp = temp[:,:,:2]
        msa_full_onehot = torch.cat((msa_masked_onehot, msa_fakeprofile_onehot, temp), dim=-1)
        
        #make fake msa_extra
        msa_extra = msa_masked[:1]
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
        temp = torch.full_like(msa_extra_onehot,0)
        temp = temp[:,:,:1]
        msa_extra_onehot = torch.cat((msa_extra_onehot, temp), dim=-1)
        msa_masked = msa_full_onehot[:1]
        '''
        b_seq.append(msa[0].clone())
        b_msa_seed.append(msa_full_onehot[:1].clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_extra.append(msa_extra_onehot[:1].clone()) #masked single sequence onehot (nb no mask so just single sequence onehot)
        b_msa_clust.append(msa_clust[:1].clone()) #unmasked original single sequence 
        b_mask_pos.append(mask_pos[:1].clone()) #mask positions in single sequence (all zeros)

    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)
    
    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturizeFixbb(seq, conf_1d=None):
    """
    Template 1D featurizer for fixed BB examples  

    Parameters:
        
        seq (torch.tensor, required): Integer sequence 
        
        conf_1d (torch.tensor, optional): Precalcualted confidence tensor
    """
    L = seq.shape[-1]

    t1d  = torch.nn.functional.one_hot(seq, num_classes=21) # one hot sequence 
    
    if conf_1d is None:
        conf = torch.ones_like(seq)[...,None]
    else:
        conf = conf_1d[:,None]
    
    
    t1d = torch.cat((t1d, conf), dim=-1)

    return t1d 


def get_train_valid_set(params):
    # read validation IDs
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])

    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4]),len(r[-1].strip())] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]

    # compile training and validation sets
    rows_subset = rows[::params['ROWS']]
    train = {}
    valid = {}
    for r in rows_subset:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append((r[:2], r[-1]))
            else:
                valid[r[2]] = [(r[:2], r[-1])]
        else:
            if r[2] in train.keys():
                train[r[2]].append((r[:2], r[-1]))
            else:
                train[r[2]] = [(r[:2], r[-1])]
    
    # compile facebook model sets
    with open(params['FB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                 if float(r[1]) > 85.0 and
                 len(r[-1].strip()) > 200]
    fb = {}
    for r in rows:
        if r[2] in fb.keys():
            fb[r[2]].append((r[:2], r[-1]))
        else:
            fb[r[2]] = [(r[:2], r[-1])]
    
    # Get average chain length in each cluster and calculate weights
    train_IDs = list(train.keys())
    fb_IDs = list(fb.keys())
    train_weights = list()
    fb_weights = list()
    for key in train_IDs:
        plen = sum([plen for _, plen in train[key]]) // len(train[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        train_weights.append(w)
    
    for key in fb_IDs:
        plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
        w = (1/512.)*max(min(float(plen),512.),256.)
        fb_weights.append(w)

    return train_IDs, torch.tensor(train_weights).float(), train, \
           fb_IDs, torch.tensor(fb_weights).float(), fb, valid

# slice long chains
def get_crop(l, device, params, unclamp=False, p_cont=0.5):

    sel = torch.arange(l,device=device)
    if l < params['LMIN']:
        return sel

    size = np.random.randint(params['LMIN'], params['LMAX']+1)
    size = min(l, size)
    
    if params['SLICE'] == 'CONT' or unclamp:
        # slice continuously
        start = np.random.randint(l-size+1)
        return sel[start:start+size]

    elif params['SLICE'] == 'DISCONT':
        if np.random.rand() < p_cont:
            # slice continuously
            start = np.random.randint(l-size+1)
            return sel[start:start+size]
        else:
            # slice discontinuously
            size1 = np.random.randint(params['LMIN']//2, size//2+1)
            size2 = size - size1
            gap = np.random.randint(l-size1-size2+1)
            
            start1 = np.random.randint(l-size1-size2-gap+1)
            sel1 = sel[start1:start1+size1]
            
            start2 = start1 + size1 + gap
            sel2 = sel[start2:start2+size2]
            
            return torch.cat([sel1,sel2])

    else:
        sys.exit('Error: wrong cropping mode:', params['SLICE'])

def loader_tbm(item, params, unclamp=False, pick_top=False):
    # TODO: change this to predict disorders rather than ignoring them
    # TODO: add featurization here

    pdb = torch.load(params['DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = torch.load(params['DIR']+'/torch/a3m/'+item[1][2:4]+'/'+item[1]+'.pt')
    tplt = torch.load(params['DIR']+'/torch/hhr/'+item[1][2:4]+'/'+item[1]+'.with_seq.pt')    
    idx = pdb['idx'][0]
   
    l_orig = a3m['msa'].shape[-1]

    # get msa features
    # removing missing residues -- TODO: change this to include missing regions as well (future work)
    msa = a3m['msa'][0].long()[...,idx]
    ins = a3m['ins'][0].long()[...,idx]
    if len(msa) > 5:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    xyz_t,f1d_t = TemplFeaturize(tplt, l_orig, params, pick_top=pick_top)
    # removing missing residues -- TODO: change this to include missing regions as well (future work)
    xyz_t = xyz_t[:,idx]
    f1d_t = f1d_t[:,idx]

    xyz = pdb['xyz'][0]

    # Residue cropping
    crop_idx = get_crop(len(idx), msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    idx = idx[crop_idx]
    ss=-1
    # For debugging
    # util.assert_all_atoms_present(seq[0], xyz, msg='loader_tbm output: xyz(true) vs seq')
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), idx.long(),\
           xyz_t.float(), f1d_t.float(), unclamp, ss, {'seq2str_mask': mask_msa}
    
class Dataset(data.Dataset):
    def __init__(self, IDs, loader, train_dict, params, unclamp_cut=0.9, pick_top=False):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.prob = {}
        for ID in self.IDs:
            p = np.array([(1/512.)*max(min(float(p_len),512.),256.) for info, p_len in self.train_dict[ID]])
            p = p / np.sum(p)
            self.prob[ID] = p

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        prob = self.prob[ID]
        sel_idx = np.random.choice(len(self.train_dict[ID]), p=prob)
        p_unclamp = np.random.rand()
        if p_unclamp > self.unclamp_cut:
            out = self.loader(self.train_dict[ID][sel_idx][0], self.params, unclamp=True, 
                              pick_top=self.pick_top)
        else:
            out = self.loader(self.train_dict[ID][sel_idx][0], self.params, pick_top=self.pick_top)
        return out

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_pdb(pdbfilename, plddtfilename, item, lddtcut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    idx_s = consecutive(np.where(plddt > lddtcut)[0])
    idx = list()
    for index in idx_s:
        if len(index) < 10:
            continue
        idx.append(index)
    idx = np.concatenate(idx)
    return {'xyz':torch.tensor(xyz[idx]), 'mask':torch.tensor(mask[idx]), 'idx':torch.tensor(idx), 'label':item}

def get_msa(a3mfilename, item):
    msa,ins = parse_a3m(a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

def loader_fb(item, params, unclamp=False, pick_top=None):
    # TODO: change this to predict disorders rather than ignoring them
    # TODO: add featurization here
    
    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'])
    idx = pdb['idx']
   
    l_orig = a3m['msa'].shape[-1]

    # get msa features
    # removing missing residues -- TODO: change this to include missing regions as well (future work)
    msa = a3m['msa'].long()[...,idx]
    ins = a3m['ins'].long()[...,idx]
    if len(msa) > 5:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features -- None
    xyz_t = torch.full((1,l_orig,14,3),np.nan).float()
    f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=21).float() # all gaps
    f1d_t = torch.cat((f1d_t, torch.zeros((1,l_orig,1)).float()), -1)
    xyz_t = xyz_t[:,idx]
    f1d_t = f1d_t[:,idx]

    xyz = pdb['xyz']

    # Residue cropping
    crop_idx = get_crop(len(idx), msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    idx = idx[crop_idx]
    mask = torch.full((len(crop_idx),), True)
    ss=-1

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), idx.long(),\
           xyz_t.float(), f1d_t.float(), unclamp, ss, {'seq2str_mask': mask_msa}

def get_loaders(task_list):
    loader_list = []
    for task in task_list:
        if task == 'seq2str' or task == 'msa_hal':
            # minkyung loader 
            loader_list.append(loader_tbm)
        elif task == 'str2seq' or task == 'hal' or task == 'str2seq_full':
            loader_list.append(loaderFixbb)
        elif 'stub' in task:
            # stub loader
            loader_list.append(loader_stub)
        else:
            raise Exception(f'task "{task}" not supported') 
    return loader_list


def get_datasets(task_names, loader_p, loader_param, n_per_epoch=None):
    loader_list = get_loaders(task_names)
    print('loader_list is ', loader_list)
    print('loader probs is ', loader_p)
    print('task names are ', task_names)

    has_stub_task = any('stub' in t for t in task_names)
    has_nonstub_task = any('stub' not in t for t in task_names)

    stub_dict = {}
    pdb_dict = {}
    valid_dict = {}
    stub_valid_dict = {}
    if has_stub_task:
        ic('Fetching stub train valid sets')
        _, stub_weights, stub_dict, stub_valid_dict = get_train_valid_set_stub(loader_param)
        ic('Done fetching stub train valid sets')
    if has_nonstub_task:
        ic('Fetching default train/valid sets')
        _, pdb_weights, pdb_dict, _, _, _, valid_dict = get_train_valid_set(loader_param)
        ic('Done fetching default train valid sets')

    def min_dataset_len(stub_dict, default_dict):
        dataset_lens = []
        if has_stub_task:
            dataset_lens.append(len(stub_dict))
        if has_nonstub_task:
            dataset_lens.append(len(default_dict))
        return min(dataset_lens)

    n_train = min_dataset_len(stub_dict, pdb_dict)
    if n_per_epoch:
        assert n_per_epoch <= n_train, f'{n_per_epoch} > {n_train}'
        n_train = n_per_epoch
    n_valid = min_dataset_len(stub_valid_dict, valid_dict)
    train_set = MultiTaskDataset(loader_list, loader_p, task_names,
                                 pdb_dict, stub_dict, n_train, loader_param)
    valid_set = MultiTaskDataset(loader_list, loader_p, task_names,
                             valid_dict, stub_valid_dict, n_valid, loader_param)
    return train_set, valid_set

class MultiTaskDataset(data.Dataset):
    """
    Generalized dataset for all inpainting and protein design tasks
    """

    def __init__(self, loaders, p_loaders, task_names,
                 pdb_dict, stub_dict, n_epoch, loader_param,
                 unclamp_cut=0.9, pick_top=False):
        """
        Parameters:

            loaders (list, required): List of functions. Each function is a
                                      different loader for new task.

            p_loaders (list, required): List of floats, probability for using each loader.

            task_names (list, required): List of strings, names for tasks. MUST match
                                         allowed names in mask making module
            ...
        """

        # sanity checks
        allowed_names = ['seq2str', 'hal', 'str2seq', 'single_ss_hal', 'str2seq_full', 'stub_fixedbb', 'stub_centroid', 'stub_fixedseq', 'stub_placebackbone']
        for name in task_names:
            if name not in allowed_names:
                raise ValueError(f'Task name {name} not in allowed names {allowed_names}')
        assert math.isclose(np.sum(p_loaders), 1.0)  # probs must sum to 1

        self.loaders = loaders
        self.p_loaders = p_loaders
        self.task_names = task_names

        self.pdb_dict       = pdb_dict
        self.stub_dict      = stub_dict
        self.n_epoch        = n_epoch
        self.loader_param   = loader_param
        self.seed           = random.randint(0, 1000)
        self.epoch = 0

        self.pick_top       = pick_top
        self.unclamp_cut    = unclamp_cut

        self.task_dicts = {
            'stub': self.stub_dict,
            'monomer': self.pdb_dict,
        }
        self.key_list_by_task = {}
        for k, d in self.task_dicts.items():
            self.key_list_by_task[k] = list(d.keys())

    def __len__(self):
        return self.n_epoch

    def task_to_dataset(self, task):
        if 'stub' in task:
            return 'stub'
        return 'monomer'

    def __getitem__(self, idx):
        # select the loader to use according to probability of each task
        task_idx = np.random.choice(np.arange(len(self.loaders)), 1, p=self.p_loaders)[0]
        chosen_task = self.task_names[task_idx]
        chosen_loader = self.loaders[task_idx]

        dataset_key = self.task_to_dataset(chosen_task)
        IDs = self.key_list_by_task[dataset_key]
        print(f'DEBUG: __getitem__ {idx}/{max(IDs)}, len(IDs):{len(IDs)}, __len__:self.n_epoch')
        ID_dict = self.task_dicts[dataset_key]

        # select cluster with idx, then grab random item in cluster
        # NOTE: Not selecting based on prob/length!!! should we?
        ID = IDs[idx]
        sel_idx = np.random.randint(0, len(ID_dict[ID]))
        item = ID_dict[ID][sel_idx][0]
        print(f'MultiTaskDataset(epoch:{self.epoch})[f{idx}/{len(self)}] chose task {chosen_task}\tID:{ID}\nitem:{item}')

        # get unclamped flag; (needed for seq2str task)
        if (np.random.rand() > self.unclamp_cut):
            unclamp = True
        else:
            unclamp = False

        # get output structure/seq/msa/templates from loader
        seq, \
        msa_seed_orig, \
        msa_seed, \
        msa_extra, \
        mask_msa, \
        xyz, \
        idx, \
        xyz_t, f1d_t, unclamp, ss, mask_kwargs = chosen_loader(item, self.loader_param, unclamp=unclamp, pick_top=self.pick_top)

        # get masks for example
        mask_dict = generate_masks(msa_seed_orig, ss, chosen_task, self.loader_param, **mask_kwargs)

        return seq, msa_seed_orig, msa_seed, msa_extra, mask_msa, xyz, idx, xyz_t, f1d_t, unclamp, mask_dict, chosen_task

    def set_epoch(self, epoch):
        ic('set_epoch called')
        self.epoch = epoch
        for task, key_list in self.key_list_by_task.items():
            random.seed(epoch)
            random.shuffle(key_list)
            self.key_list_by_task[task] = key_list


class DistilledDataset(data.Dataset):
    def __init__(self,
                 pdb_IDs,
                 pdb_dict,
                 fb_IDs,
                 fb_dict,
                 loader,
                 params):
        #
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.loader = loader
        self.params = params
        self.unclamp_cut = 0.9

        self.fb_inds = np.arange(len(self.fb_IDs))
        self.pdb_inds = np.arange(len(self.pdb_IDs))
    
    def __len__(self):
        return len(self.fb_inds) + len(self.pdb_inds)

    def __getitem__(self, index):
        p_unclamp = np.random.rand()
        if index >= len(self.fb_inds): # from PDB set
            ID = self.pdb_IDs[index-len(self.fb_inds)]
            sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.loader(self.pdb_dict[ID][sel_idx][0], self.params, unclamp=True, dataset='pdb')
            else:
                out = self.loader(self.pdb_dict[ID][sel_idx][0], self.params, unclamp=False, dataset='pdb')
        else:
            ID = self.fb_IDs[index]
            sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
            if p_unclamp > self.unclamp_cut:
                out = self.loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=True, dataset='fb')
            else:
                out = self.loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=False, dataset='fb')
        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(self, dataset, pdb_weights, fb_weights, num_pdb_per_epoch=12800, num_replicas=None, rank=None, replacement=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        assert num_pdb_per_epoch % num_replicas == 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_pdb_per_epoch = num_pdb_per_epoch
        self.num_fb_per_epoch = num_pdb_per_epoch*3
        self.total_size = num_pdb_per_epoch * 4 # (25%: PDB / 75%: FB)
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.pdb_weights = pdb_weights
        self.fb_weights = fb_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # 1. subsample fb and pdb based on length
        fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
        pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
        
        fb_indices = indices[fb_sampled]
        pdb_indices = indices[pdb_sampled + len(self.dataset.fb_IDs)]

        indices = torch.cat((fb_indices, pdb_indices))

        # shuffle indices
        indices = indices[torch.randperm(len(indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def loaderFixbb(item, params, unclamp=False, pick_top=False):
    # TODO: change this to predict disorders rather than ignoring them
    # TODO: add featurization here

    pdb = torch.load(params['DIR']   +'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = torch.load(params['DIR']   +'/torch/a3m/'+item[1][2:4]+'/'+item[1]+'.pt')

    ic(params)
    ss  = torch.load(os.path.join(params['SS_TORCH_DIR'], 'torch_ss/'+item[0][1:3]+'/'+item[0]+'.pt'))[1]

    #tplt = torch.load(params['DIR']+'/torch/hhr/'+item[1][2:4]+'/'+item[1]+'.with_seq.pt')
    idx = pdb['idx'][0]

    l_orig = a3m['msa'].shape[-1]

    # get msa features
    # removing missing residues -- TODO: change this to include missing regions as well (future work)
    msa = a3m['msa'][0].long()[...,idx]
    ins = a3m['ins'][0].long()[...,idx]
    if len(msa) > 10:
        msa, ins = MSABlockDeletion(msa, ins)


    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize_fixbb(msa, params)
    # get template 1d features 
    f1d_t = TemplFeaturizeFixbb(seq)

    xyz   = pdb['xyz'][0,:,:]
    xyz_t = torch.clone(xyz)[None]

    # removing missing residues -- TODO: change this to include missing regions as well (future work)

    #xyz_t = xyz_t[:,idx]
    #f1d_t = f1d_t[:,idx]

    # Residue cropping
    crop_idx = get_crop(len(idx), msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    idx = idx[crop_idx]
    ss = ss[crop_idx]
    
    xyz_t = xyz_t.repeat(4,1,1,1)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), idx.long(),\
           xyz_t.float(), f1d_t.float(), unclamp, ss, {}

def loaderFixbb_fb(item, params, unclamp=False, pick_top=False):
    #print('fb set!')
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
              os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
              item[0], params['PLDDTCUT'])
    # get msa features
    # removing missing residues -- TODO: change this to include missing regions as well (future work)
    idx = pdb['idx']
    msa = a3m['msa'].long()[...,idx]
    ins = a3m['ins'].long()[...,idx]
    if len(msa) > 10:
        msa, ins = MSABlockDeletion(msa, ins)

    #tplt = torch.load(params['DIR']+'/torch/hhr/'+item[1][2:4]+'/'+item[1]+'.with_seq.pt')
    ss = torch.load(params['FB_SS_TORCH_DIR'] +item[1][:2]+'/'+item[1][2:] +'/'+ item[0]+'.pt')[1]
    #print('fb seq',len(ss))
    xyz = pdb['xyz']
    xyz_t = torch.clone(xyz)[None]
    l_orig = a3m['msa'].shape[-1]

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize_fixbb(msa, params)
    # get template 1d features 
    f1d_t = TemplFeaturizeFixbb(seq)
    # Residue cropping
    crop_idx = get_crop(len(idx), msa_seed_orig.device, params, unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    idx = idx[crop_idx]
    ss=ss[crop_idx]
    
    xyz_t = xyz_t.repeat(4,1,1,1)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), idx.long(),\
           xyz_t.float(), f1d_t.float(), unclamp, ss, {}

class loaderInpaint():

    def __init__(self, p_task, task_names, seqID_cut):
        """
        Parameters:
            
            loaders (list, required): List of functions. Each function is a 
                                      different loader for new task. 

            p_loaders (list, required): List of floats, probability for using each loader. 

            task_names (list, required): List of strings, names for tasks. MUST match 
                                         allowed names in mask making module
        """

        # sanity checks 
        allowed_names = ['seq2str', 'hal', 'str2seq', 'str2seq_full']
        for name in task_names:
            if name not in allowed_names:
                raise ValueError(f'Task name {name} not in allowed names {allowed_names}')
        assert math.isclose( np.sum(p_task), 1.0 ) # probs must sum to 1 

        self.p_task  = p_task
        ic(self.p_task)
        self.task_names = task_names
        self.seqID_cut  = seqID_cut
        
        loader_mapping = {'seq2str':loader_tbm,
                       'hal':loaderFixbb,
                       'str2seq':loaderFixbb,
                       'str2seq_full':loaderFixbb}
        fb_loader_mapping = {'seq2str':loader_fb,
                       'hal':loaderFixbb_fb,
                       'str2seq':loaderFixbb_fb,
                       'str2seq_full':loaderFixbb_fb}
        
        self.loaders = [loader_mapping[i] for i in self.task_names]        
        self.fb_loaders = [fb_loader_mapping[i] for i in self.task_names]
        ic(self.loaders)
        ic(self.fb_loaders)
    def __call__(self, item, params, unclamp=False, pick_top=False, dataset='pdb'):

        # select the loader to use according to probability of each task 
        idx = np.random.choice(np.arange(len(self.loaders)), 1, p=self.p_task)[0]

        chosen_task   = self.task_names[idx]
        if dataset=='pdb':
            chosen_loader = self.loaders[idx]
        else:
            chosen_loader = self.fb_loaders[idx]

        # get output structure/seq/msa/templates from loader 
        seq,\
        msa_seed_orig,\
        msa_seed,\
        msa_extra,\
        mask_msa,\
        xyz,\
        idx,\
        xyz_t, f1d_t, unclamp, ss, mask_kwargs = chosen_loader(item, params, unclamp=unclamp, pick_top=pick_top)
         
        # get masks for example
        mask_dict = generate_masks(msa_seed_orig, ss, chosen_task, params, **mask_kwargs)
        
        return seq, msa_seed_orig, msa_seed, msa_extra, mask_msa, xyz, idx, xyz_t, f1d_t, unclamp, mask_dict, chosen_task
