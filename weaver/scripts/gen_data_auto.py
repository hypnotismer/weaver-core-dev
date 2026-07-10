#!/usr/bin/env python3
"""Generate .auto.yaml (standardization + reweight hists) on a single process.

Uses the full training file list (no DDP rank split). Intended to run before
multi-GPU training so all ranks share the same reweight hists.
"""

import argparse
import glob
import os
import sys

# allow `python scripts/gen_data_auto.py` from the weaver directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data.config import DataConfig, _md5
from utils.data.preprocess import AutoStandardizer, WeightMaker
from utils.logger import _logger, _configLogger


def _to_filelist(data_train):
    file_dict = {}
    for entry in data_train:
        if ':' in entry:
            name, fp = entry.split(':', 1)
        else:
            name, fp = '_', entry
        files = sorted(glob.glob(fp))
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files
    return file_dict


def _autogen_path(data_config):
    digest = _md5(data_config)
    return data_config.replace('.yaml', '.%s.auto.yaml' % digest)


def _generate(file_dict, data_config_file, remake_weights=False):
    autogen_file = _autogen_path(data_config_file)
    loaded_config = data_config_file
    if os.path.exists(autogen_file) and not remake_weights:
        loaded_config = autogen_file
        _logger.info('Found existing auto config, will update if needed: %s', autogen_file)

    data_config = DataConfig.load(loaded_config)

    if data_config._missing_standardization_info:
        _logger.info('Running AutoStandardizer')
        std = AutoStandardizer(file_dict, data_config)
        data_config = std.produce(autogen_file)

    if data_config.weight_name and not data_config.use_precomputed_weights:
        if remake_weights or data_config.reweight_hists is None:
            if os.path.exists(autogen_file):
                data_config = DataConfig.load(autogen_file)
            _logger.info('Running WeightMaker')
            wgt = WeightMaker(file_dict, data_config)
            data_config = wgt.produce(autogen_file)

    return autogen_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', required=True, help='input YAML data config')
    parser.add_argument('--data-train', nargs='+', required=True,
                        help='training files, same format as train.py --data-train')
    parser.add_argument('--remake-weights', action='store_true',
                        help='recompute even if .auto.yaml already exists')
    parser.add_argument('--log-file', type=str, default='',
                        help='also write weaver logger output (incl. reweight matrices) to this file')
    args = parser.parse_args()

    # Must configure handlers; otherwise _logger.info (reweight matrices) is discarded.
    _configLogger('weaver', stdout=sys.stdout, filename=args.log_file or None)

    if not os.path.exists(args.data_config):
        parser.error('data config not found: %s' % args.data_config)

    autogen_file = _autogen_path(args.data_config)
    if os.path.exists(autogen_file) and not args.remake_weights:
        _logger.info('Auto config already exists, skip: %s', autogen_file)
        return 0

    file_dict = _to_filelist(args.data_train)
    n_files = sum(len(v) for v in file_dict.values())
    if n_files == 0:
        _logger.error('No input files matched --data-train')
        return 1

    _logger.info('Generating auto config from %d files -> %s', n_files, autogen_file)
    out = _generate(file_dict, args.data_config, remake_weights=args.remake_weights)

    if not os.path.exists(out):
        _logger.error('Expected auto config was not written: %s', out)
        return 1

    _logger.info('Done: %s', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
