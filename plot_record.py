#!/usr/bin/env python3
"""
plot_record.py
Load a single record from a MATLAB v7.3 .mat file (or from sample_record.npz) and plot PPG and ABP.

Usage examples:
  python plot_record.py Part_1.mat --index 0
  python plot_record.py --use-sample --index 0

The script saves a PNG in the same folder, by default named `ppg_abp_record_{index}.png`.
"""

import argparse
import os
import sys

def load_record_from_mat(matpath, index=0, varname='Part_1'):
    import h5py
    with h5py.File(matpath, 'r') as f:
        if varname not in f:
            raise KeyError(f"Variable '{varname}' not found in {matpath}")
        part = f[varname]
        # part is a cell array stored as object references; shape (N,1)
        if part.shape[0] <= index:
            raise IndexError(f"Index {index} out of range for {varname} with length {part.shape[0]}")
        ref = part[index][0]
        ds = f[ref]
        arr = ds[()]
        return arr

def load_record_from_npz(npzpath):
    import numpy as np
    d = np.load(npzpath)
    if 'record' in d:
        return d['record']
    # try first array
    keys = list(d.keys())
    if len(keys) > 0:
        return d[keys[0]]
    raise KeyError('No arrays found in npz file')

def plot_and_save(arr, outpath, fs=125.0, title=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Expect arr shape (samples, 3) or (3, samples)
    if arr.ndim != 2 or (arr.shape[1] != 3 and arr.shape[0] != 3):
        raise ValueError(f'Unexpected array shape {arr.shape}; expected (samples,3) or (3,samples)')

    if arr.shape[0] == 3:
        # channels x samples -> transpose
        arr = arr.T

    ppg = arr[:, 0]
    abp = arr[:, 1]

    t = np.arange(ppg.shape[0]) / float(fs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, ppg, color='tab:blue')
    axes[0].set_ylabel('PPG (a.u.)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, abp, color='tab:red')
    axes[1].set_ylabel('ABP (mmHg)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('matfile', nargs='?', default='Part_1.mat', help='Path to .mat file (default Part_1.mat)')
    ap.add_argument('--index', type=int, default=0, help='Record index inside the cell array (default 0)')
    ap.add_argument('--out', default=None, help='Output PNG path')
    ap.add_argument('--fs', type=float, default=125.0, help='Sampling frequency in Hz (default 125)')
    ap.add_argument('--use-sample', action='store_true', help='Use sample_record.npz instead of reading .mat')
    ap.add_argument('--var', default='Part_1', help='MATLAB variable name containing the cell array (default Part_1)')
    args = ap.parse_args()

    folder = os.path.abspath(os.path.dirname(args.matfile))
    if args.out is None:
        outname = f'ppg_abp_record_{args.index}.png'
        outpath = os.path.join(folder, outname)
    else:
        outpath = args.out

    try:
        if args.use_sample:
            npzpath = os.path.join(folder, 'sample_record.npz')
            if not os.path.exists(npzpath):
                raise FileNotFoundError(f'Sample file not found: {npzpath}')
            arr = load_record_from_npz(npzpath)
        else:
            arr = load_record_from_mat(args.matfile, args.index, varname=args.var)

        plot_and_save(arr, outpath, fs=args.fs, title=f'Record {args.index} - PPG & ABP')
        print(f'Saved plot to: {outpath}')
    except Exception as e:
        print('Error:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
