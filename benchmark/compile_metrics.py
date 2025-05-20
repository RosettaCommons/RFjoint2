#!/usr/bin/env python
#
# Compiles metrics from scoring runs into a single dataframe CSV
#

import os, argparse, glob
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs')
    parser.add_argument('--outcsv',type=str,default='compiled_metrics.csv',help='Output filename')
    args = parser.parse_args()

    filenames = glob.glob(args.datadir+'/*.trb')

    records = []
    for fn in filenames:
        name = os.path.basename(fn).replace('.trb','')
        trb = np.load(fn, allow_pickle=True)

        record = {'name':name}
        if 'lddt' in trb:
            record['lddt'] = trb['lddt'].mean()
        if 'inpaint_lddt' in trb:
            record['inpaint_lddt'] = np.mean(trb['inpaint_lddt'])
        if 'sampled_mask' in trb:
            record['sampled_mask'] = trb['sampled_mask']
        if 'flags' in trb:
            record.update(vars(trb['flags']))

        records.append(record)

    df = pd.DataFrame.from_records(records)
    af2 = pd.concat([
        pd.read_csv(fn,index_col=0) for fn in glob.glob(args.datadir+'/af2_metrics.csv*')
    ])
    df = df.merge(af2, on='name')
    df.to_csv(args.datadir+'/'+args.outcsv)
    print(f'Wrote metrics dataframe {df.shape} to "{args.datadir}/{args.outcsv}"')

if __name__ == "__main__":
    main()
