#!/bin/bash

path_to_pdb="./pdbs/barrelCylinder_n8_S12_nres12.0_b4.8_dtw1.0A-bb.pdb"
output_path="./example_outputs/inpainting_test"
number_of_designs=10
contig_string="2-2,A1-12,3-3,A13-24,3-3,A25-36,3-3,A37-48,3-3,A49-60,3-3,A61-72,3-3,A73-84,3-3,A85-96,2-2"
inpaint_seq="A1-96"
tmpl_conf=0.99

conda activate SE3nv
python ../inpaint.py \
        --pdb $path_to_pdb \
        --out $output_path \
        --num_designs $number_of_designs \
        --contigs $contig_string \
        --inpaint_seq $inpaint_seq \
        --tmpl_conf $tmpl_conf
