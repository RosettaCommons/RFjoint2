#!/home/dimaio/.conda/envs/SE3nv/bin/python
"""
Inference script for autofold3, based on all atom RF
"""
import sys, os, subprocess, pickle, time, json
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path = sys.path + [script_dir+'/utils/']
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
import argparse
import random
import copy
from copy import deepcopy
from collections import namedtuple
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from RoseTTAFoldModel import RoseTTAFoldModule
from util import *
from inpainting_util import *
from kinematics import get_init_xyz, xyz_to_t2d
import inf_methods
import parsers
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODE_CHOICES = ['hal']

#get model params and set up model
'''
TRUNK_PARAMS = ['n_extra_block', 'n_main_block', 'n_ref_block',\
                'd_msa', 'd_msa_full', 'd_pair', 'd_templ',\
                'n_head_msa', 'n_head_pair', 'n_head_templ', 'd_hidden', 'd_hidden_templ', 'p_drop']

SE3_PARAMS = ['num_layers', 'num_channels', 'num_degrees', 'n_heads', 'div',
              'l0_in_features', 'l0_out_features', 'l1_in_features', 'l1_out_features',
              'num_edge_features']
'''
TRUNK_PARAMS = {'n_extra_block' : 4,\
                 'n_main_block' : 10,\
                 'n_ref_block' : 4,\
                 'd_msa' : 256,\
                 'd_msa_full' : 64,\
                 'd_pair' : 256,\
                 'd_templ' : 64,\
                 'n_head_msa' : 8,\
                 'n_head_pair' : 8,\
                 'n_head_templ' : 4,\
                 'd_hidden' : 32,\
                 'd_hidden_templ' : 64,\
                 'p_drop' : 0.15}

SE3_PARAMS = {'num_layers':3,\
              'num_channels':32,\
              'num_degrees' : 2,\
              'n_heads' : 4,\
              'div' : 4,\
              'l0_in_features' : 64,\
              'l0_out_features' : 64,\
              'l1_in_features' : 14,\
              'l1_out_features' : 2,\
              'num_edge_features' : 64}

TRUNK_PARAMS['SE3_param'] = SE3_PARAMS
DEFAULT_CKPT = f'{script_dir}/checkpoints/InpaintingApril22.pt'
# for perlmutter
#DEFAULT_CKPT = '/global/cfs/cdirs/m4129/software/trdesign/autofold/InpaintingApril22.pt'

def get_args():
    """
    Parse command line args
    """
    parser = argparse.ArgumentParser()

    # design-related args
    parser.add_argument('--pdb','-p',dest='pdb',
            help='input protein')
    parser.add_argument('--contigs', default=None, nargs='+',
            help='Pieces of input protein to keep ')
    parser.add_argument('--length',default=None,type=str,
            help='Specify length, or length range, you want the outputs. e.g. 100 or 95-105')
    parser.add_argument('--task', default='hal', choices=MODE_CHOICES,
            help='Design task to do.')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT,
            help='Checkpoint to pretrained RFold module')
    parser.add_argument('--inpaint_str', type=str, default=None, nargs='+',
         help='Predict the structure at these residues. Similar mask (and window), but is '\
              'specifically for structure.')
    parser.add_argument('--inpaint_seq', type=str, default=None, nargs='+',
         help='Predict the sequence at these residues. Similar mask (and window), but is '\
              'specifically for sequence.')
    parser.add_argument('--n_cycle', type=int, default=500,
            help='Number of recycles through RFold')
    parser.add_argument('--tmpl_conf', type=str, default='1',
            help='1D confidence value for template residues')
    parser.add_argument('--num_designs', type=int, default=1,
            help='Number of designs to make')
    parser.add_argument('--start_num', type=int, default=0,
            help='Number of first design to output')
    parser.add_argument('--topo_pdb', help='input protein topology')
    parser.add_argument('--topo_conf',default=0.1,help='template confidence for topology input')
    parser.add_argument('--topo_contigs', default=None, nargs='+',
        help='contig string for topology input. should be same total length as --contigs')
    parser.add_argument('--res_translate',type=str,default=None,
        help='Which residues to translate (randomly in x, y and z direction), '\
             'with maximum distance to translate specified, e.g. "A35,2:B22,4" translates '\
             'residue A35 up to 2A in a random direction, and B22 up to 4A. If specified '\
             'residues are in masked --window, they will be unmasked. In --contig mode, residues'\
'must not be masked (as need to know where to put them. Default distance to translate is 2A.')
    parser.add_argument('--tie_translate',type=str,default=None,
        help='For randomly translating multiple residues together (e.g. to move a whole secondary structure element). '\
             'Syntax is e.g. "A22,A27,A30,4.0:A48,A50" which '\
             'would randomly move residues A22, A27 and A30 together up to 4A, and A48 and A50 together (but in a different random direction/distance to the first block) '\
             'to a default distance of up to 2A.'\
             'Alternatively, residues can be specifed like "A12-26,6.0:A40-52,A56". '\
             'This can be specified alongside --res_translate, so some residues are tied, and some are not, but if residues are specified in both, they will only be moved in their'\
             ' tied block (i.e. their --res_translate will be ignored)')
    parser.add_argument('--block_rotate',type=str,default=None,
        help="Do you want to rotate a whole structural block (or single residue)? Syntax is same as tie_translate. Rotation is in degrees.")
    parser.add_argument('--multi_templates',type=str,default=None,
            help='Split contig into multiple templates (to remove inter-domain information). Syntax: A1-100,C1-40:B1-50 puts A and C in one template, B on a different template')
    parser.add_argument('--multi_tmpl_conf',type=str,default=None,
            help='Template confidence for each template (as specified in --multi_templates flat). Syntax is e.g. 1,1,0.5')
    parser.add_argument('--temperature',type=float,default=0.1,
            help='Softmax temperature for autoregressive sequence decoding')
    parser.add_argument('--min_decoding_distance',type=int,default=15,
            help='Minimum distance between residues being simultaneously decoded')
    parser.add_argument('--exclude_aa',type=str,default=None,
            help='Do you want to exclude certain amino acids from the inpainted region (syntax: --exclude_aa P)')
    
    # i/o args
    parser.add_argument('--out', default='pdbs_test/auto_out',
            help='output directory and for files')
    parser.add_argument('--dump_pdb', default=True, action='store_true',
            help='Whether to dump pdb output')
    parser.add_argument('--dump_trb', default=True, action='store_true',
            help='Whether to dump trb files in output dir')
    parser.add_argument('--dump_npz', default=False, action='store_true',
            help='Whether to dump npz (disto/anglograms) files in output dir')
    parser.add_argument('--dump_all', default=False, action='store_true',
            help='If true, will dump all possible outputs to outdir')
    parser.add_argument('--input_json', type=str, default=None,
        help='JSON-formatted list of dictionaries, each containing command-line arguments for 1 '\
             'design.')
    parser.add_argument('--cautious', default=False, action='store_true',
            help='If true, will not run a design if output file already exists.')

    # specific contig arguments. Don't use unless you REALLY know what you're doing. TODO implement actually using these
    parser.add_argument('--ref_idx', default=None, type=str,
        help='pdb indexing in the reference. Gap is given by ("_","_")')
    parser.add_argument('--hal_idx',default=None,type=str,
        help='pdb indexing in the output')
    parser.add_argument('--idx_rf',default=None, type=str,
        help='indexing for rosettafold')
    parser.add_argument('--inpaint_seq_tensor', default=None, type=str,
        help='Boolean list of whether residues have sequence masked')
    parser.add_argument('--inpaint_str_tensor',default=None, type=str,
        help='Boolean list of whether residues have structure masked')
    args = parser.parse_args()
    args = process_args(args)

    return args

class EMA(nn.Module):
      def __init__(self, model, decay):
          super().__init__()
          self.decay = decay
  
          self.model = model
          self.shadow = deepcopy(self.model)
  
          for param in self.shadow.parameters():
              param.detach_()
  
      @torch.no_grad()
      def update(self):
          if not self.training:
              print("EMA update should only be called during training", file=stderr, flush=True)
              return
  
          model_params = OrderedDict(self.model.named_parameters())
          shadow_params = OrderedDict(self.shadow.named_parameters())
  
          # check if both model contains the same set of keys
          assert model_params.keys() == shadow_params.keys()
  
          for name, param in model_params.items():
              if param.requires_grad:
                  # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
                  # shadow_variable -= (1 - decay) * (shadow_variable - variable)
                  shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))
  
          model_buffers = OrderedDict(self.model.named_buffers())
          shadow_buffers = OrderedDict(self.shadow.named_buffers())
  
          # check if both model contains the same set of keys
          assert model_buffers.keys() == shadow_buffers.keys()
  
          for name, buffer in model_buffers.items():
              # buffers are copied
              shadow_buffers[name].copy_(buffer)
  
      def forward(self, *args, **kwargs):
          if self.training:
              return self.model(*args, **kwargs)
          else:
              return self.shadow(*args, **kwargs)

def dump_args(args):
    """
    Dump flags into output dir 
    """
    outdir = args.outdir

    with open(os.path.join(outdir, 'FLAGS.txt'), 'w') as fp:
        for key,val in vars(args).items():
            fp.write(str(key) + ' '*8 + str(val) + '\n')


def process_args(args):
    """
    Does any argument postprocessing     
    """
    if args.dump_all:
        args.dump_pdb   = True
        args.dump_trb   = True
        args.dump_npz   = True

    if args.out != None:
        args.outdir= '/'.join(args.out.split('/')[:-1])
        args.prefix = args.out.split('/')[-1]


    return args

def process_topo(args, seq):

    # allow inputting a single topo confidence or pdb for multiple contig strings
    topo_contig_list = [x.strip().split() for x in ' '.join(args.topo_contigs).split(':')]
    if args.topo_pdb is None:
        topo_pdbfn_list = [args.pdb]
    else:
        topo_pdbfn_list = [x.strip() for x in args.topo_pdb.split(':')]
    if len(topo_pdbfn_list) == 1:
        topo_pdbfn_list = len(topo_contig_list)*topo_pdbfn_list
    else:
        assert(len(topo_pdbfn_list)==len(topo_contig_list))
    topo_conf_list = [x.strip() for x in args.topo_conf.split(':')]
    if len(topo_conf_list) == 1:
        topo_conf_list = len(topo_contig_list)*topo_conf_list
    else:
        assert(len(topo_conf_list)==len(topo_contig_list))
    
    xyz_topo_all = []
    t1d_topo_all = []
    for topo_pdbfn, topo_contigs, topo_conf in zip(topo_pdbfn_list, 
                                                   topo_contig_list, 
                                                   topo_conf_list):
        topo_pdb = parsers.parse_pdb(topo_pdbfn)
        rm_topo = ContigMap(topo_pdb, topo_contigs)
        mappings_topo=get_mappings(rm_topo)

        xyz_topo_ = torch.full((1,1,len(rm_topo.ref),14,3), np.nan)
        xyz_topo_[:,:,rm_topo.hal_idx0,:,:] = \
            torch.from_numpy(topo_pdb['xyz'])[rm_topo.ref_idx0,:,:][None, None,...]
        xyz_topo_[:,3:] = float('nan') # don't use sidechain info for topology input
        xyz_topo_all.append(xyz_topo_)

        mask_str_topo = torch.from_numpy(rm_topo.inpaint_str)[None,:]
        conf_1d_topo = torch.ones_like(seq)*float(topo_conf)
        conf_1d_topo[~mask_str_topo[0]] = 0 # conf=0 where structure is masked
        t1d_topo_ = TemplFeaturizeFixbb(seq, conf_1d=conf_1d_topo)[None,None,:]
        t1d_topo_all.append(t1d_topo_)

    xyz_topo = torch.cat(xyz_topo_all,dim=1)
    t1d_topo = torch.cat(t1d_topo_all,dim=1)
    return xyz_topo, t1d_topo


def main():
    
    # parse args 
    args = get_args()
    if os.path.exists(args.checkpoint) is False:
        print("WARNING: couldn't find checkpoint")
    #dump_args(args)

    # make model and load checkpoint 
    print('Loading model checkpoint...')
    model = RoseTTAFoldModule(**TRUNK_PARAMS).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model_state = ckpt['model_state_dict']
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print('Successfully loaded model checkpoint')

    # loop through dicts of arguments from json input
    if args.input_json is not None:
        with open(args.input_json) as f_json:
            argdicts = json.load(f_json)
        print(f'List of argument dicts loaded from JSON {os.path.abspath(args.input_json)}')
    else:
        # no json input, spoof list of argument dicts
        argdicts = [{}]


    for i_argdict, argdict in enumerate(argdicts):

        if args.input_json is not None:
            print(f'\nAdding argument dict {i_argdict} from input JSON ({len(argdicts)} total):')
            print(argdict)
            for k,v in argdict.items():
                setattr(args, k, v)


        # output prefix for this set of arguments
        if args.input_json is None or \
            (args.input_json is not None and \
                ('out' in argdict or 'outf' in argdict or 'prefix' in argdict)):
            # new output name specified in this argdict
            argdict_prefix = args.out
        else:
            # same output prefix, add number to make it unique
            argdict_prefix = args.out+f'_{i_argdict}'


        params = {'MAXCYCLE':args.n_cycle,'TEMPERATURE':args.temperature, 'DISTANCE':args.min_decoding_distance}
        
        # parse pdb
        parsed_pdb = parsers.parse_pdb(args.pdb)
        L = len(parsed_pdb['idx'])

        #get positions of residues to translate
        if args.res_translate is not None:
            res_translate = get_translated_coords(args)
        if args.tie_translate is not None:
            if args.res_translate is not None:
                res_translate = get_tied_translated_coords(args, res_translate)
            else:
                res_translate = get_tied_translated_coords(args)
        if args.block_rotate is not None:
            block_rotate = parse_block_rotate(args)


        for i_des in range(args.start_num, args.start_num+args.num_designs):

            out_prefix = f'{args.out}_{i_des}'

            if args.cautious and os.path.exists(out_prefix + '.pdb'):
                print(f'CAUTIOUS MODE: Skipping design because output file '\
                      f'{out_prefix + ".pdb"} already exists.')
                continue

            print('On design', i_des)
            t0 = time.time()

        
            # parse pdb 
            parsed_pdb = parsers.parse_pdb(args.pdb)

            # process contigs and generate masks
            rm = ContigMap(parsed_pdb, args.contigs, args.inpaint_seq, args.inpaint_str, args.length, args.ref_idx, args.hal_idx, args.idx_rf, args.inpaint_seq_tensor, args.inpaint_str_tensor)
            mappings = get_mappings(rm)
            mask_str = torch.from_numpy(rm.inpaint_str)[None,:]
            mask_seq = torch.from_numpy(rm.inpaint_seq)[None,:]
            blank_mask = torch.ones(mask_str.size()[-1])[None,:].bool()
            
            seq = torch.from_numpy(parsed_pdb['seq']) 
            xyz_true = torch.from_numpy(parsed_pdb['xyz'][:,:,:])
            
            # get raw inputs before remapping 
            #inputs = get_inputs(parsed_pdb, design_params, use_af=args.use_af)
            if args.res_translate is not None or args.tie_translate is not None:
                xyz_true,translate_dict = translate_coords(parsed_pdb, res_translate)
                xyz_true = torch.from_numpy(xyz_true)
            else:
                xyz_true = torch.from_numpy(parsed_pdb['xyz'][:,:,:])
            if args.block_rotate is not None:
                xyz_true,rotate_dict = rotate_block(xyz_true,block_rotate,parsed_pdb['pdb_idx'])
            '''
            if args.contigs is not None:
                    xyz_t    = rm.scatter_1d(xyz_true[:,:].clone(), np.nan, feature='str')[None,None,...]  # (1,1,L,14,3)
                    seq      = rm.scatter_1d(seq, 20, feature='seq')
            '''
            xyz_t = torch.full((1,1,len(rm.ref),14,3), np.nan)
            xyz_t[:,:,rm.hal_idx0,:,:] = xyz_true[rm.ref_idx0,:,:][None, None,...]
            seq_t = torch.full((1,len(rm.ref)),20).squeeze()
            seq_t[rm.hal_idx0] = seq[rm.ref_idx0]
            seq=seq_t
            # template confidence 
            conf_1d = torch.ones_like(seq)*float(args.tmpl_conf)      
            conf_1d[~mask_str[0]] = 0 # zero confidence for places where structure is masked
            
            # Get sequence and MSA input features 
            seq_hot, msa, msa_hot, msa_extra_hot, _ = MSAFeaturize_fixbb(seq[None,:],params)
            t1d = TemplFeaturizeFixbb(seq, conf_1d=conf_1d)[None,None,:]        
            idx_pdb = torch.from_numpy(np.array(rm.rf)).int()[None,:]
            seq_hot = seq_hot.unsqueeze(dim=0)
            msa = msa.unsqueeze(dim=0)
            msa_hot = msa_hot.unsqueeze(dim=0)
            msa_extra_hot = msa_extra_hot.unsqueeze(dim=0)

            # topology input features
            if args.topo_contigs is not None:
                xyz_topo, t1d_topo = process_topo(args, seq)

            # Mask the inputs 
            seq, msa_masked, msa_full, xyz_t, t1d = mask_inputs(seq_hot,
                                                                msa_hot, 
                                                                msa_extra_hot, 
                                                                xyz_t, 
                                                                t1d,
                                                                input_seq_mask=mask_seq, 
                                                                input_str_mask=mask_str, 
                                                                input_t1dconf_mask=blank_mask)
                                                                
            if args.topo_contigs is not None:
                xyz_t=torch.cat((xyz_t, xyz_topo), dim=1)
                t1d = torch.cat((t1d, t1d_topo), dim = 1)
            
            # get index 1D features and masks 
            idx_pdb = idx_pdb.to(DEVICE, non_blocking=True) # (B, L)
            mask_str = mask_str.to(DEVICE, non_blocking=True) # (B, L)
            xyz_t = xyz_t.to(DEVICE, non_blocking=True)
            t1d = t1d.to(DEVICE, non_blocking=True)
           
            # check if need to split input into multiple templates
            if args.multi_templates is not None:
                xyz_t, t1d = split_templates(xyz_t, t1d, args.multi_templates, mappings, args.multi_tmpl_conf)            
            
            # check if want to exclude certain amino acids:
            if args.exclude_aa is not None:
                print(args.exclude_aa)
                exclude_aa = [i.upper() for i in args.exclude_aa.split(",")]
                print(exclude_aa)
            else:
                exclude_aa=None

            # put tensors on Device
            seq = seq.to(DEVICE, non_blocking=True)
            msa = msa.to(DEVICE, non_blocking=True)
            msa_masked = msa_masked.to(DEVICE, non_blocking=True)
            msa_full = msa_full.to(DEVICE, non_blocking=True)
            ti_dev =  torsion_indices.to(DEVICE, non_blocking=True)
            ti_flip = torsion_can_flip.to(DEVICE, non_blocking=True)
            ang_ref = reference_angles.to(DEVICE, non_blocking=True)
             
            # preprocess featuers 
            t2d, alpha, alpha_mask, chis, t1d, xyz_t, xyz_prev, state = preprocess(xyz_t, 
                                                                                   t1d,
                                                                                   DEVICE, 
                                                                                   None,
                                                                                   ti_dev, 
                                                                                   ti_flip, 
                                                                                   ang_ref) 
            
            # fwd passes 
            with torch.no_grad():
                logit_s,\
                logit_aa_s,\
                pred_crds,\
                pred_lddts,\
                seq_out, seq_prear, str_prear, pred_lddts_prear = inf_methods.classic_inference(model, 
                                                        msa_masked.float(), 
                                                        msa_full.float(), 
                                                        seq.long(), 
                                                        t1d.float(), 
                                                        t2d, 
                                                        idx_pdb.long(), 
                                                        args.n_cycle,
                                                        xyz_prev, 
                                                        state, alpha, 
                                                        xyz_t.float(), 
                                                        ti_dev, 
                                                        msa.long(),params,exclude_aa=exclude_aa)

            atoms_out = pred_crds[-1].squeeze()
            #atoms_out_prear = str_prear[-1].squeeze()
            mask_str=mask_str.squeeze()  

            if args.dump_pdb:
                chain_ids = [i[0] for i in rm.hal]
                fname = out_prefix + '.pdb'
                #fname_prear = out_prefix + '_pre_AR.pdb'
                write_pdb(fname, seq_out.squeeze(), atoms_out, Bfacts=pred_lddts.squeeze().cpu().numpy(),chains=chain_ids)
                
                #write_pdb(fname_prear, seq_prear.squeeze(), atoms_out_prear, Bfacts=pred_lddts_prear.squeeze().cpu().numpy(), chains=chain_ids)
            lddt = pred_lddts.squeeze().cpu().numpy()
            strmasktemp = mask_str.cpu().numpy()
            partial_lddt = [lddt[i] for i in range(np.shape(strmasktemp)[0]) if strmasktemp[i] == 0]
            trb = {}
            trb['lddt'] = lddt
            trb['inpaint_lddt'] = partial_lddt
            trb['flags'] = args
            trb['contigs'] = args.contigs
            trb['device'] = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU'
            trb['time'] = time.time() - t0

            for key, value in mappings.items():
                #NOTE added in the line below to skip if dictionary flag is empty
                if type(value) == list and value == []:
                    continue
                if type(value) == list and type(value[0]) != tuple:
                    value=np.array(value)
                trb[key] = value
            if args.dump_trb:
                with open(f'{args.out}_{i_des}.trb','wb') as f_out:
                    pickle.dump(trb, f_out)        
            if args.dump_npz:
                probs = [torch.nn.functional.softmax(l, dim=1) for l in logit_s]
                dict_pred = {}
                dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
                dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
                dict_pred['p_theta'] = probs[2].permute([0,2,3,1])
                dict_pred['p_phi'] = probs[3].permute([0,2,3,1])

                np.savez(f'{args.out}_{i_des}.npz',
                    dist=dict_pred['p_dist'].detach().cpu().numpy()[0,],
                    omega=dict_pred['p_omega'].detach().cpu().numpy()[0,],
                    theta=dict_pred['p_theta'].detach().cpu().numpy()[0,],
                    phi=dict_pred['p_phi'].detach().cpu().numpy()[0,])

            print(f'Finished design {args.out}_{i_des} in {(time.time() - t0)/60:.2f} minutes.')

if __name__ == '__main__':
    main()
