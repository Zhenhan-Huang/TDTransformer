import os
import sys
import re
import argparse

import numpy as np

from collections import OrderedDict, defaultdict


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse result")
    parser.add_argument('--wdir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_seeds', type=int)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--ptrn', type=str, default=None)
    parser.add_argument("--no_ci95", action="store_true", default=False)
    return parser.parse_args()


def read_performance(parent_dir, directory):
    fpath = os.path.join(parent_dir, directory, 'log.txt')
    assert os.path.isfile(fpath), f'File "{fpath}" not exist'
    num_match = 0
    auc = 0.
    acc = 0.
    with open(fpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            assert num_match < 2, f'Number of matches is {num_match}, expected to be < 2'
            line = line.strip()
            match = re.compile(fr"\* Auc is ([\.\deE+-]+), Acc is ([\.\deE+-]+)").search(line)

            if match:
                auc = float(match.group(1))
                acc = float(match.group(2)) * 100
                num_match += 1
    return {'file': fpath, 'auc': auc, 'acc': acc} 


def main():
    args = parse_arguments()
    args.ci95 = not args.no_ci95
    dir_base = args.ptrn.replace('[dataset]', args.dataset)

    results = []
    for seed in range(args.num_seeds):
        seed_str = str(seed + 1)
        dir_base_nm = dir_base.replace('[seed]', seed_str)
        ret = read_performance(args.wdir, dir_base_nm)
        results.append(ret)

    metrics_results = defaultdict(list)
    for ret in results:
        msg = ""
        for key, value in ret.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()
    print("===", flush=True)
    print(f"Summary of directory: {args.wdir} and dataset {args.dataset}", flush=True)
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%", flush=True)
        output_results[key] = avg
    print("===\n\n", flush=True)


if __name__ == '__main__':
    main()