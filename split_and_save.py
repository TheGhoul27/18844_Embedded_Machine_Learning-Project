#!/usr/bin/env python3
"""
split_and_save.py

Split MATLAB v7.3 dataset cell arrays into smaller files containing a fixed
number of records (instances). Saves grouped records as compressed NPZ files
in an output folder.

Usage:
  python split_and_save.py --group-size 10
  python split_and_save.py Part_1.mat --group-size 5

Default behavior: process all files matching Part_*.mat in the script folder.
Each output file will be named like:
  split_records/Part_1_group_000.npz

Each NPZ contains arrays named record_0, record_1, ... and metadata saved as
an additional file with the same base name and `.meta.json` containing the
original part name and record indices.
"""

import argparse
import glob
import h5py
import json
import numpy as np
import os
import sys


def find_cell_var(f):
    # find first top-level variable with MATLAB_class == 'cell'
    for k in f.keys():
        try:
            cls = f[k].attrs.get('MATLAB_class')
            if cls is None:
                continue
            sval = cls.decode() if isinstance(cls, (bytes, bytearray)) else str(cls)
            if 'cell' in sval.lower():
                return k
        except Exception:
            continue
    return None


def deref_record(f, part_obj, idx):
    # part_obj is the cell array object
    # in these MAT files part_obj[i][0] is an HDF5 reference
    try:
        ref = part_obj[idx][0]
        ds = f[ref]
        arr = ds[()]
        return arr
    except Exception as e:
        raise RuntimeError(f'Could not dereference record {idx}: {e}')


def process_mat_file(matpath, outdir, group_size=10, varname=None):
    print(f'Processing {matpath} ...')
    basename = os.path.splitext(os.path.basename(matpath))[0]
    with h5py.File(matpath, 'r') as f:
        if varname is None:
            var = find_cell_var(f)
            if var is None:
                raise KeyError(f'No cell variable found in {matpath} (please specify --var)')
        else:
            var = varname
            if var not in f:
                raise KeyError(f'Variable {var} not found in {matpath}')

        part = f[var]
        try:
            n_records = int(part.shape[0])
        except Exception:
            # fallback: try len(part)
            n_records = int(len(part))

        print(f' Found cell variable "{var}" with {n_records} records')

        os.makedirs(outdir, exist_ok=True)
        group_idx = 0
        i = 0
        written = 0
        while i < n_records:
            chunk = []
            indices = []
            for j in range(group_size):
                idx = i + j
                if idx >= n_records:
                    break
                try:
                    arr = deref_record(f, part, idx)
                except Exception as e:
                    print(f'  Warning: skipping record {idx}: {e}')
                    continue
                chunk.append(arr)
                indices.append(idx)

            if len(chunk) == 0:
                break

            outbase = f'{basename}_group_{group_idx:04d}'
            outnpz = os.path.join(outdir, outbase + '.npz')
            metajson = os.path.join(outdir, outbase + '.meta.json')

            # prepare dict for np.savez
            save_dict = {}
            for k, rec in enumerate(chunk):
                save_dict[f'record_{k}'] = rec

            # also save a small index array
            save_dict['__indices__'] = np.array(indices, dtype=np.int32)

            np.savez_compressed(outnpz, **save_dict)

            meta = {
                'source_mat': os.path.basename(matpath),
                'mat_variable': var,
                'group_index': group_idx,
                'record_indices': indices,
                'n_records_in_group': len(chunk),
            }
            with open(metajson, 'w') as mf:
                json.dump(meta, mf, indent=2)

            print(f'  Wrote {outnpz} ({len(chunk)} records)')
            written += len(chunk)
            group_idx += 1
            i += group_size

        print(f' Finished {basename}: wrote {group_idx} files containing {written} records total')
        return group_idx, written


def main():
    ap = argparse.ArgumentParser(description='Split MATLAB v7.3 cell-array datasets into small NPZ files')
    ap.add_argument('files', nargs='*', help='.mat files to process (default: Part_*.mat)')
    ap.add_argument('--group-size', type=int, default=10, help='Number of records per output file (default 10)')
    ap.add_argument('--out-dir', default='split_records', help='Output folder (default split_records)')
    ap.add_argument('--var', default=None, help='MATLAB variable name if not auto-detectable')
    args = ap.parse_args()

    cwd = os.path.abspath(os.path.dirname(__file__))
    if len(args.files) == 0:
        pattern = os.path.join(cwd, 'Part_*.mat')
        files = sorted(glob.glob(pattern))
    else:
        files = [os.path.abspath(p) for p in args.files]

    if len(files) == 0:
        print('No .mat files found to process. Looked for Part_*.mat in', cwd)
        sys.exit(2)

    outdir = os.path.join(cwd, args.out_dir)

    total_files = 0
    total_records = 0
    for mat in files:
        try:
            gf, gr = process_mat_file(mat, outdir, group_size=args.group_size, varname=args.var)
            total_files += gf
            total_records += gr
        except Exception as e:
            print(f'Error processing {mat}:', e)

    print(f'All done. Wrote {total_files} output files containing {total_records} records to {outdir}')


if __name__ == '__main__':
    main()
