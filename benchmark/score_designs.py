#!/usr/bin/env python
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import os, argparse, itertools, json, glob
import numpy as np
import slurm_tools

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=-1,help='How many designs to score in each job')
    parser.add_argument('--job_list',type=str,help='file to store list of jobs in. also will be slurm job name')
    parser.add_argument('--gpu', type=str, default='rtx2080',help='Type of GPU, either rtx2080 or a4000')
    parser.add_argument('--tmp_pre',type=str,default='score.list', help='Name prefix of temporary files with lists of designs to score')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--keep_logs', dest='keep_logs', action="store_true", default=False, help='Keep slurm logs.')
    args = parser.parse_args()

    filenames = sorted(glob.glob(args.datadir+'/*.pdb'))
    if args.chunk == -1:
        args.chunk = len(filenames)

    job_list_file = open(args.job_list, 'w') if args.job_list is not None else sys.stdout

    for i in np.arange(0,len(filenames),args.chunk):
        tmp_fn = f'{args.datadir}/{args.tmp_pre}.{i}'
        with open(tmp_fn,'w') as outf:
            for j in np.arange(i,min(i+args.chunk, len(filenames))):
                print(filenames[j], file=outf)
        print(f'source activate ampere; python {script_dir}/af2_metrics.py '\
              f'--outcsv {args.datadir}/af2_metrics.csv.{i} '\
              f'{tmp_fn}', file=job_list_file)

    if args.job_list is not None: job_list_file.close()

    # submit job
    if args.submit and args.job_list is not None:
        if args.gpu == 'a4000':
            slurm_job, proc = slurm_tools.gpu_array_submit(args.job_list, p = 'gpu-remote', gres='gpu:a4000:1', log=args.keep_logs)
        else: # rtx2080
            slurm_job, proc = slurm_tools.gpu_array_submit(args.job_list, log=args.keep_logs)
        print(slurm_job)

if __name__ == "__main__":
    main()
