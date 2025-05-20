import torch 
import torch.nn as nn 
from util_module import ComputeAllAtomCoords
from util import *
from inpainting_util import MSAFeaturize_fixbb, TemplFeaturizeFixbb
from kinematics import xyz_to_t2d
conversion = 'ARNDCQEGHILKMFPSTWYV-'
def classic_inference(model, msa, msa_extra, seq, t1d, t2d, idx_pdb, N_cycle, xyz_prev, state, alpha, xyz_t, ti_dev, msa_orig, params, exclude_aa=None):
    """ 
    Trying Autoregressive 
    """
    msa_prev  = None
    pair_prev = None
    best_lddt = -1*torch.ones((1, seq[0].shape[1]), device=seq.device,  dtype=float) # amir's hack
    compute_allatom_coords=ComputeAllAtomCoords().to(seq.device) 
    print(''.join([conversion[i] for i in seq[0,0]]))
    counter=0
    
    # This is a hacky way of taking the last structure in some strange edge cases
    take_last = False
    # Create a mask which is either True (inpainted residue) or False (fixed residue)
    inpaint_mask = torch.clone(torch.where(seq[0,0] == 20, True, False))[None, :]
    if torch.sum(inpaint_mask) == 0:
        print("You are not inpainting any structure. Measuring plddt across the whole structure, rather than in the (non-existent) inpainting region")
        inpaint_mask = ~inpaint_mask
        take_last = True
        N_cycle = 10
    with torch.no_grad():
        for i_cycle in range(N_cycle):
            if i_cycle+1 < N_cycle:
                mask_chis = get_tor_mask(seq[:,i_cycle+1], ti_dev) # get mask with next round sequence
                mask_chis = mask_chis[...,None] # (B, L, 10, 1)
            with torch.cuda.amp.autocast(True):
                logit_s, logit_aa_s, xyz_prev, pred_lddt, alpha, msa_prev, pair_prev = model(msa[:,i_cycle], msa_extra[:,i_cycle],
                                                           seq[:,i_cycle], xyz_prev,
                                                           state, alpha,
                                                           idx_pdb,
                                                           t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                           msa_prev=msa_prev,
                                                           pair_prev=pair_prev,
                                                           return_infer=True)
                seq_out = torch.argmax(logit_aa_s, dim=-1)
                _, xyz_prev = compute_allatom_coords(seq_out, xyz_prev, alpha)
                if i_cycle+1 < N_cycle:
                    chis = alpha / (torch.norm(alpha, dim=-1, keepdim=True) + 1e-6) # (B, L, 10, 2)
                    chis *= mask_chis
                    chis = torch.cat((chis, mask_chis), dim=-1)
                    B = chis.shape[0]
                    L = chis.shape[1]
                    state = torch.cat((nn.Softmax(dim=-1)(logit_aa_s), pred_lddt[...,None], chis.reshape(1,L,-1)), dim=-1)
            if pred_lddt[inpaint_mask].mean() > best_lddt[inpaint_mask].mean() or take_last == True:
                best_xyz = torch.clone(xyz_prev)
                best_lddt = torch.clone(pred_lddt)
                best_seq = torch.clone(seq_out)
                best_logit_s = [torch.clone(l) for l in logit_s]
                best_logit_aa_s = torch.clone(logit_aa_s)
                counter = 0
                best_logits = torch.clone(logit_aa_s)
            else:
                counter += 1
            print(''.join([conversion[i] for i in seq[0,0]]))
            print(''.join([conversion[i] for i in torch.argmax(logit_aa_s[0,:,:], dim=-1)]))
            print ("RECYCLE [%02d/%02d] current LDDT (inpainted region): %.4f | best LDDT (inpainted region): %.4f"%(i_cycle, N_cycle, pred_lddt[inpaint_mask].mean().item(), best_lddt[inpaint_mask].mean().item()))
            if counter == 10:
                if "best_xyz" not in locals():
                    print("This is a strange case where inpainting has not improved through recycling (in its plddt metric). Taking the last model anyway.")
                    best_xyz = torch.clone(xyz_prev)
                    best_lddt = torch.clone(pred_lddt)
                    best_seq = torch.clone(seq_out)
                    best_logit_s = [torch.clone(l) for l in logit_s]
                    best_logit_aa_s = torch.clone(logit_aa_s)
                    best_logits = torch.clone(logit_aa_s)
                else:
                    print("LDDT hasn't improved for 10 recycles. Moving to AR decoding")
                break
        best_lddt_prear = torch.clone(best_lddt)
        # best_lddt = np.array([0])
        best_lddt = torch.zeros((1, seq[0].shape[1]), device=seq.device,  dtype=float) # amir's hack
        #save sequence for later
        seq_pre_ar = torch.clone(best_seq)
        str_t_pre_ar = torch.clone(best_xyz)
        seq = seq[:,0]
        msa = msa[:,0,:,:,:]
        msa_extra=msa_extra[:,0,:,:,:]
        L = seq.shape[1]
        n_decoding = (seq==20).sum(dim=-1)
        temperature = params['TEMPERATURE']
        logit_aa_s = torch.clone(best_logits)
        counter = 0
        for i_cycle in range(n_decoding):
            #select a random index
            decode = False
            n_left_to_decode = (seq==20).sum(dim=-1)
            if n_left_to_decode == 0:
                break
            while decode is False:
                temp=np.random.randint(L)
                if seq[:,temp] == 20:
                    decode=True
                    decode_idx = temp
            decoding_mask = get_decoding_mask(decode_idx, xyz_prev, seq,params['DISTANCE'])            

            #logits=logit_aa_s[:,decoding_mask,:]
            logits=logit_aa_s
            if exclude_aa is not None:
                for aa in exclude_aa:
                    logits[:,decoding_mask,conversion.index(aa)] = 0
            probs = torch.nn.functional.softmax(logits/temperature+1e-7, dim=-1)[0]
            sampled_aa = torch.multinomial(probs,1)[:,0]
            seq[:,decoding_mask] = sampled_aa[decoding_mask]
            _,_,msa_temp,msa_extra_temp,_=MSAFeaturize_fixbb(seq, params)
            msa[:,:,decoding_mask,:] = msa_temp[:1,:,decoding_mask,:].float()
            msa_extra[:,:,decoding_mask,:] = msa_extra_temp[:1,:,decoding_mask,:].float()
            temp = [i for i in range(len(decoding_mask.numpy().tolist())) if decoding_mask.numpy().tolist()[i] == True]
            print(f'Sampling at residues {", ".join([str(i) for i in temp])} with temperature {temperature}')
            #print(f'Highest probability residue: {conversion[torch.argmax(probs)]}')
            #print(f'Sampled amino acid: {conversion[sampled_aa[0]]}')
            
            if i_cycle+1 < n_decoding:
                mask_chis = get_tor_mask(seq, ti_dev) # get mask with next round sequence
                mask_chis = mask_chis[...,None] # (B, L, 10, 1)

            with torch.cuda.amp.autocast(True):
                logit_s, logit_aa_s, xyz_prev, pred_lddt, alpha, msa_prev, pair_prev = model(msa, msa_extra,
                                                           seq, xyz_prev,
                                                           state, alpha,
                                                           idx_pdb,
                                                           t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                           msa_prev=msa_prev,
                                                           pair_prev=pair_prev,
                                                           return_infer=True)
                seq_out = torch.argmax(logit_aa_s, dim=-1)
                _, xyz_prev = compute_allatom_coords(seq_out, xyz_prev, alpha)
                if i_cycle+1 < N_cycle:
                    chis = alpha / (torch.norm(alpha, dim=-1, keepdim=True) + 1e-6) # (B, L, 10, 2)
                    chis *= mask_chis
                    chis = torch.cat((chis, mask_chis), dim=-1)
                    B = chis.shape[0]
                    L = chis.shape[1]
                    state = torch.cat((nn.Softmax(dim=-1)(logit_aa_s), pred_lddt[...,None], chis.reshape(1,L,-1)), dim=-1)
            if pred_lddt[inpaint_mask].mean() > best_lddt[inpaint_mask].mean():
                best_xyz = torch.clone(xyz_prev)
                best_lddt = torch.clone(pred_lddt)
                best_seq = torch.clone(seq_out)
                best_logit_s = [torch.clone(l) for l in logit_s]
                best_logit_aa_s = torch.clone(logit_aa_s)
                counter = 0 
            else:
                counter += 1
            print(''.join([conversion[i] for i in seq[0]]))
            print(''.join([conversion[i] for i in torch.argmax(logit_aa_s[0,:,:], dim=-1)]))
            print ("RECYCLE [%02d/%02d] current LDDT (inpainted region): %.4f | best LDDT (inpainted region): %.4f"%(i_cycle, N_cycle, pred_lddt[inpaint_mask].mean().item(), best_lddt[inpaint_mask].mean().item()))
        torch.cuda.empty_cache()
        
        # get sequence by argmaxing final logits  
        return best_logit_s, best_logit_aa_s, best_xyz, best_lddt, best_seq, seq_pre_ar, str_t_pre_ar, best_lddt_prear

def get_fake_templates(L):
        xyz_t = torch.full((1,1,L,14,3), np.nan)
        seq = torch.full((1,L),20).squeeze()
            # template confidence 
        conf_1d = torch.zeros_like(seq)    
    
        # Get sequence and MSA input features 
        t1d = TemplFeaturizeFixbb(seq, conf_1d=conf_1d)[None,None,:]    
        idx_pdb = torch.from_numpy(np.arange(0,L)).int()[None,:]
        t2d=xyz_to_t2d(xyz_t)
        return xyz_t, t1d, t2d

def get_decoding_mask(decode_idx, xyz, seq, distance=15):
    L=seq.shape[1]
    output_mask = torch.zeros(L).bool()
    output_mask[decode_idx] = True
    seq_masked = torch.where(seq[0] == 20, True, False)
    if distance > 20:
        print('WARNING: any distance specified greater than 20A is treated as 20A, as distances are binned')
    assert seq_masked[decode_idx]==True
    xyz=xyz[None].float()
    t2d = xyz_to_t2d(xyz)[0,0,:,:,:37]
    t2d = torch.argmax(t2d, dim=-1)
    distance=int(distance)
    if distance >= 20:
        min_bin=35
    else:
        min_bin = (distance-2)*2

    possibilities = torch.where(t2d[decode_idx] > min_bin, True, False)*seq_masked
    while torch.sum(possibilities) > 0:
        decode=False
        while decode is False:
            temp=np.random.randint(L)
            if possibilities[temp] == True:
                decode=True
                decode_idx = temp
        output_mask[decode_idx] = True
        possibilities = torch.clone(torch.where(t2d[decode_idx] > min_bin, True, False)*possibilities)
        possibilities[decode_idx] = False
        #print(f'Number of possible positions to decode = {torch.sum(possibilities)}')
    return output_mask

