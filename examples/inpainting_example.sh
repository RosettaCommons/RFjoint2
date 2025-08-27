#!/bin/bash

conda activate SE3nv_rfj

path_to_pdb="./pdbs/2KL8.pdb"
output_path="./inpainting_output"
number_of_designs=1
contig_string="A1-20,20-50,A30-60"

python ../inpaint.py\
        --pdb $path_to_pdb \
        --out $output_path \
        --num_designs $number_of_designs \
        --contigs $contig_string
