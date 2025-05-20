#!/bin/bash

conda activate SE3nv

path_to_pdb="./pdbs/minibinder_fusion.pdb"
output_path="./example_outputs/inpainting_test"
number_of_designs=10
contig_string="B1-55,15-50,A1-55"
multiple_templates="B1-55:A1-55"

python ../inpaint.py\
        --pdb $path_to_pdb \
        --out $output_path \
        --num_designs $number_of_designs \
        --contigs $contig_string\
        --multi_templates $multiple_templates
