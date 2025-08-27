# Protein Inpainting

Codes for running RFjoint2.

Note that this code was a precursor to RFdiffusion, which supersedes it in all regards. This code is provided for reproducibility purposes, and we do not suggest using it for actual design campaigns. RFdiffusion, or other alternative and more modern methods will provide better outputs.

This was used in Castro, K.M. et al. 2024, to scaffold multiple RSV epitopes
This was used in Kim, D.E. et al., 2025, to generate parametrically-defined beta-barrels.

## Description
RFjoint2 is a method for "conditional joint protein sequence/structure generation". This means that given some combination of protein sequence and structure that you have, you can use this method to simultaneously generate more sequence and structure conditioned on that input. 

**Things inpainitng is good at:**
- Refining non-ideal parts of proteins 
- Resampling protein structures near a starting structure 
- Re-looping proteins (i.e., keep tertiary/secondary structure but changing the order in which elements appear in sequence space)
- Rigidly fusing two protein domains 
- Loop building 
- Scaffolding medium-sized motifs
- Building beta-barrels given template input.

## Getting started
To get started using inpainting, you will need to clone this git repository. 

1. Navigate to a place on digs where you would like to clone this repo 
2. `git clone https://github.com/joewatchwell/RFjoint2.git` 

### Conda Install SE3-Transformer

Ensure that you have either [Anaconda or Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

You also need to install [NVIDIA's implementation of SE(3)-Transformers](https://developer.nvidia.com/blog/accelerating-se3-transformers-training-using-an-nvidia-open-source-model-implementation/) Here is how to install the NVIDIA SE(3)-Transformer code:

```
conda env create -f env/SE3nv_rfj.yml

conda activate SE3nv_rfj
cd env
pip install --no-cache-dir -r requirements.txt
python setup.py install

```
Anytime you run RFjoint2 you should be sure to activate this conda environment by running the following command:
```
conda activate SE3nv
```
Total setup should take less than 30 minutes on a standard desktop computer.
Note: Due to the variation in GPU types and drivers that users have access to, we are not able to make one environment that will run on all setups. As such, we are only providing a yml file with support for CUDA 11.1 and leaving it to each user to customize it to work on their setups. This customization will involve changing the cudatoolkit and (possibly) the PyTorch version specified in the yml file.

### Download model weights

```
mkdir checkpoints && cd checkpoints
wget http://files.ipd.uw.edu/pub/rfjoint2/InpaintingApril22.pt
```


### Running the inpainting script 
The actual script you will execute is called `inpaint.py`. There are many ways to run it, which we will explore through examples below. 

**Basic execution** 
A baseline execution of the script requires 3 pieces of information, which are provided in the form of flags coming in from the command line that you provide. The three things are 
1. A template protein structure/sequence, in the form of a pdb file. (`--pdb`)
2. Specification of which parts of the protein are being kept, removed, and inpainted. This is provided as a "contigs string". (`--contigs`)
3. A location where output from the script should be written. (`--out`)

```
python inpaint.py --pdb examples/pdbs/2KL8.pdb  --contigs A25-50,10,A61-79 --out  examples/out
```
Let's dissect this line. 

The first flag denotes that the pdb file `2KL8.pdb` will be used as the source for any template protein information we want to scaffold. 

The second flag is the contigs string, which says a few things:
- The A25-50 denotes the first contiguously *kept* region ("contig") - meaning that we are essentially going to copy the sequence and structure of all residues from A25 to A50 in `2KL8.pdb` (inclusive). Since `A25` comes first in the contigs string, it will be the first residue in the output protein. Note that even though `2KL8.pdb` also contains residues A1-24, they will not be considered during design because they are not in the contigs string.  
- The `,10` denotes that *directly* attached to `A1-25` there will be 10 inpainted residues, where both sequence and structure will be generated. 
- The `,A61-79` denotes that *directly* attached to the 10-residue inpainted region we will have residues `A61-79` from `2KL8.pdb`.  

Finally, the last flag denotes where output will go. This is always in the form `path/to/my/outdir/prefix` - that is, it begins with the path to a folder, and the last bit denotes the prefix that all files will be tagged with when they go into that folder. The standard files which are output is a `.pdb` file for each design, allong with a metadata `.trb` file which contains information about how your contigs ("kept regions") were mapped from the input protein to the output protein, the RoseTTAFold pLDDT predictions, etc. **NOTE:** within a given design run with `inpaint.py`, any files being output into your output direectory will not clobber each other because each individual design will be tagged with a unique ID number. But designs from separate runs can clobber each other. 

## Understanding the --contigs flag in depth.

The contig string is very flexible, but requires an in depth understanding of how it works.
In the example above, this is the 'simplest' case of inpainting, where a gap of a fixed length (10 residues) is being filled with inpainting (A25-50 is seen by the network, and is connected by 10 'inpainted' residues to A61-79, which is also seen by the network).
However, there are many more things that can be done with the contig string.

1. Variable length inpainting. In many cases, you don't know the exact number of residues needed to fill a gap. Therefore, you can specify a range, e.g. `A25-50,7-13,A61-79`. This will allow between 7 and 13 (inclusive) residues to be inpainted. However, there is no 'intelligent' sampling here - the length is currently just randomly sampled from the given range. Therefore, you'll need to run multiple inpainting runs to sample this range, with e.g. `--num_designs 3`
2. Inpainting in the presence of another, fixed chain. Sometimes, you'll want to inpaint in the presence of another chain, e.g. a receptor that you want to inpaint a binder against. This can be specified in the contig by adding the pdb indices of this chain, e.g. if your target is on the B chain, you might specify `B30-750,0 A25-50,7-13,A61-79`. The ',0 ' here specifies a chain break. This contig string is equivalent to `A25-50,7-13,A61-79,0 B30-750` or to `A25-50,7-13,A61-79,0 B30-750,0`. If you have another target/receptor chain, just add it to the contig! E.g. `B30-750,0 C1-100,0 A25-50,7-13,A61-79`
3. The problem with inpainting in the presence of a target/receptor is that it slows inpainting down a lot (it scales quadratically with the length of the total input). Therefore, you might want to crop your receptor into multiple chunks, to limit the size of the total input to the network. *Probably* (although this hasn't been fully tested), the best way to input these fragments is as follows: E.g. `B30-45,B70-84,B730-748,0 C4-12,C70-92,0 A25-50,7-13,A61-79`. This keeps the relative residue indexing between fragments of the SAME chain. I.e. `B30-45,B70-84` represent two fragments of the same chain, which, in this input, will have a relative 'index' jump, going into the network, of 24 residues. In other words, even though the network doesn't 'see' residues B46-69, it *knows* that there are only these 24 residues missing. Fragments from separate chains (e.g. the B and C chains in this example), MUST be separated by a ',0 ', indicating a chain break. An alternative way of specifying this input would be `B30-45,0 B70-84,0 B730-748,0 etc etc`. In this input, each fragment is seen as a separate chain to the network (i.e. it treats them as if they were never connected). This is *probably* not good practice (in reality it may not really matter).
4. Inpainting on multiple chains. It's easy to inpaint on multiple chains if you want. E.g. `A25-50,7-13,A61-79,0 B1-10,40-47,B600-629`. If you want to do this in the presence of another target/receptor, you can add that in as above.
5. Fusing multiple chains. Your 'visible' parts of structure can be linked together with inpainting, even if they're on separate chains in the input file. E.g. `A25-50,15-25,B1-50` will fuse part of the 'A' chain to part of the 'B' chain.
5. Generating designs of fixed (or within a range of) lengths. If you have a complex contig string, e.g. `A1-10,10-40,A20-50,30-50,B1-20,10-20`. Here, there are many combinations of lengths to be inpainted, but you might want all of your outputs to be the same length (or below a certain length e.g. for chip ordering). Add the `--length` flag, with either a fixed value (if you want a fixed range), e.g. `--length 100` or with a range, e.g. `--length 0-130`. Obviously inpainting will crash if the length given is incompatible with the contig input. Note that the `--length` only refers to chains where inpainting is happening (i.e. not receptor/target chains)

## Understanding the outputs

1. The PDB file. We now output pdb files in the 'Rosetta-style' pdb indexing. By default, inpainted chains (i.e. chains where inpainting took place) will be on chains starting from 'A'(e.g. if there is only one chain where inpainting is taking place, this will be chain 'A'. If there are two, they will be on chains 'A' and 'B', ordered as in the contig string input). Any fixed chains (target/receptor chains) will be output on the NEXT AVAILABLE chain. The numbers are continuous from 1 (i.e. if chain A is length 75, chain A will be numbered 1-75 and chain B from 76-).
2. The trb file. This contains metadata about the inpainting run. Open this with np.load([File], allow_pickle=True).
    - `lddt`- This is the inpainting network's prediction at how 'good' the output structure is. This is per residue, and includes all of the unmasked region.
    - `inpaint_lddt` - This is just the part of the 'lddt' output corresponding to the inpainted region. Normally, we filter on the mean of this output, to take the 'best' ~5-10% of outputs.
    - details about mapping (i.e. how residues in the input map to residues in the output)
        - `con_ref_pdb_idx`/`con_hal_pdb_idx` - These are two arrays including the input pdb indices (in con_ref_pdb_idx), and where they are in the output pdb (in con_hal_pdb_idx). This only contains the chains where inpainting took place (i.e. not any fixed receptor/target chains)
        - `con_ref_idx0`/`con_hal_idx0` - These are the same as above, but 0 indexed, and without chain information. This is useful for splicing coordinates out (to assess alignment etc).
        - `complex_con...` and `receptor_con...` - If you have included fixed receptor/target chains, these will be included either on their own (in receptor_con...) or with the inpainted chains (in complex_con...).
        - `sampled_mask` - if you've specified a range of lengths to be inpainted, this `sampled_mask` will give the precise length used during that specific inpainting run

## Other flags

- `--inpaint_seq`: This allows the sequence of residues to be masked, while the backbone coordinates are given. This is handy if you're, for example, making a fusion between two monomeric proteins, and part of what was the surface is now going to be in the core of the protein. This is specified similarly to the contig string, but without chain breaks (e.g. `A1-3,A4,A6,B10-100`)
- `--inpaint_str`: This allows the structure of a residue to be masked, but not its sequence. This essentially asks the network to predict the structure of this residue.
- `--res_translate`: Which residues to translate (randomly in x, y and z direction), with maximum distance to translate specified, e.g. `A35,2:B22,4` translates residue A35 up to 2A in a random direction, and B22 up to 4A. If specified residues are in masked --window, they will be unmasked. In --contig mode, residues must not be masked (as need to know where to put them. Default distance to translate is 2A.
- `--tie_translate`: For randomly translating multiple residues together (e.g. to move a whole secondary structure element). Syntax is e.g. `A22,A27,A30,4.0:A48,A50` which would randomly move residues A22, A27 and A30 together up to 4A, and A48 and A50 together (but in a different random direction/distance to the first block) to a default distance of up to 2A. Alternatively, residues can be specifed like `A12-26,6.0:A40-52,A56`. This can be specified alongside --res_translate, so some residues are tied, and some are not, but if residues are specified in both, they will only be moved in their tied block (i.e. their --res_translate will be ignored)
- `--block_rotate`: Do you want to rotate a whole structural block (or single residue)? Syntax is same as tie_translate. Rotation is in degrees.
- `--multi_templates`: THIS IS REALLY IMPORTANT TO UNDERSTAND. Sometimes, you have two (or more) blocks that you want to fuse together. Sometimes you care about the relative position of the parts in the output (e.g. you want to hold two minibinders in the perfect conformation to drive receptor signalling). For that, the `--contigs` flag achieves this. E.g. `A1-100,20-50,B1-100`. Other times, you don't care how the two parts are oriented in the output (you just want the internal structure of each part to be fixed). For this, the `--multi_templates` flag is useful. Specifying that different parts should be on different 'templates' gives the network the internal structure of each part, but NOT their relative position. E.g. `--contigs A1-100,20-50,B1-100 --multi_templates A1-100:B1-100` means the network will fuse chain A to chain B but won't know their original relative positions, and will therefore 'choose' how they fit together during inpainting. Multiple blocks can be specified on the same template, such as `--contigs A1-100,20-50,B1-100,0 C20-70 --multi_templates A1-100,C20-70:B1-100` gives the network the relative position of A to C, but not their position with respect to B.
- `--num_designs`: Number of designs (inpaints) to generate. Note INPAINTING IS DETERMINISTIC. Therefore, you need at least as many possible input combinations (e.g. lengths of inpainted regions etc) to generate actual diversity. Otherwise, identical inputs in = identical inputs out (and a LOT of wasted compute)
- `--topo_pdb`: pdb for "topology input". This will be input as a 2nd (backbone-only) template to inpainting, with a user-specified confidence, to bias the structure of the inpainted regions.
- `--topo_conf`: pLDDT-like residue-wise confidence to assign the topology template input features. Higher values will result in the output being more structurally similar to the --topo_pdb.
- `--topo_contigs`: contig string representing the portions of the topology pdb to be used as template inputs. This must match the total length of the `--contigs`. This means you cannot use length ranges in the `--contigs` argument when using `--topo_contigs`. To sample gap lengths when using `--topo_contigs`, you must generate a list of commands with different gap lengths and make sure they are consistent between `--contigs` and `--topo_contigs`.
`--temperature`: During autoregressive decoding, sequence is sampled with a certain temperature. Default is 0.1 and it's probably best to keep it at this level, but higher or lower could be experimented with.
`--min_decoding_distance`: Currently, by default, we decode multiple amino acids simultaneously when doing autoregressive sequence design. This value specifies the minimum distance apart two residues can be while being simultaneously decoded. 15A is the default, and is pretty good. If you want to be more conservative (but also slower), set this to a higher value.

All other flags can generally be left as default, but either dive into the code or ask us if you have any questions.

## FAQs
1. 'How much protein can inpainting inpaint?' 
        This depends on the problem, but generally it will struggle with inpainting more than around 60 residues. If you have multiple regions to be inpainted (between visible blocks), the total amount of protein to be inpainted could be quite a lot more than this though (if each segment was say, around 50 residues).
2. 'I see chain breaks/clashes in my pdb output'
        Inpainting often fails, probably because the set of lengths given to the network during that run were incompatible with making a good protein (at least with inpainting). These designs will generally have a low mean 'inpaint_lddt' metric though, so if you only take the top-scoring 10% or so of designs, these shouldn't have bad clashes etc.
3. 'What is a good cutoff for the mean 'inpaint_lddt' score?
         The raw value for this depends on a range of factors, so will vary problem-by-problem. Normally, I just visually inspect a few outputs with a range of 'inpaint_lddt' metrics, and choose a cutoff based on this. Or, I just take the top-scoring 10% of outputs.

## Authors and acknowledgment
This work was developed by Joseph Watson (jwatson3@uw.edu), David Juergens (davidcj@uw.edu), Jue Wang (jue@uw.edu) and Woody Ahern (ahern@uw.edu)
