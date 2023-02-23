import argparse
import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random

logger = logging.getLogger('__main__')

class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')

        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading/preprocessing. By default, equals num. of available cores.')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed',
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')
        # Dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
                                      "If in [0,1], it will be interpreted as a proportion of the dataset, "
                                      "otherwise as an integer absolute number of samples")
        self.parser.add_argument('--test_only', choices={'testset', 'fold_transduction'},
                                 help='If set, no training will take place; instead, trained model will be loaded and evaluated on test set')
        self.parser.add_argument('--test_from',
                                 help='If given, will read test IDs from specified text file containing sample IDs one in each row')
        self.parser.add_argument('--labels', type=str,
                                 help="In case a dataset contains several labels (multi-task), "
                                      "which type of labels should be used in regression or classification, i.e. name of column(s).")
        self.parser.add_argument('--val_ratio', type=float, default=0,
                                 help="Proportion of the dataset to be used as a validation set")
        self.parser.add_argument('--test_ratio', type=float, default=0,
                                 help="Set aside this proportion of the dataset as a test set")
        self.parser.add_argument('--pattern', type=str, default="TRAIN",
                                 help='Regex pattern used to select files contained in `data_dir`. If None, all data will be used.')
        self.parser.add_argument('--val_pattern', type=str,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the validation set.
                            If None, a positive `val_ratio` will be used to reserve part of the common data set.""")
        self.parser.add_argument('--test_pattern', type=str, default="TEST",
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the test set.
                            If None, `test_ratio`, if specified, will be used to reserve part of the common data set.""")
        self.parser.add_argument('--normalization',
                                 choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
                                 default='standardization',
                                 help='If specified, will apply normalization on the input features of a dataset.')
        self.parser.add_argument('--norm_from',
                                 help="""If given, will read normalization values (e.g. mean, std, min, max) from specified pickle file.
                            The columns correspond to features, rows correspond to mean, std or min, max.""")
        self.parser.add_argument('--subsample_factor', type=int,
                                 help='Sub-sampling factor used for long sequences: keep every kth sample')
        
        # Dataset
        self.parser.add_argument('--data_dir', default='../Classification/JapaneseVowels',
                                 help='Data directory')
        self.parser.add_argument('--task', choices={"imputation", "transduction", "classification", "regression"},
                                 default="classification",
                                 help=("Training objective/task: imputation of masked values,\n"
                                       "                          transduction of features to other features,\n"
                                       "                          classification of entire time series,\n"
                                       "                          regression of scalar(s) for entire time series"))
        self.parser.add_argument('--n_splits', default=5,
                                 help='how many splits')
                                 
        # Draw
        self.parser.add_argument('--differ', default=True,
                                 help="""if use, we use different color and marker to draw the images""") 
        self.parser.add_argument('--override', default=False,
                                 help="""if use, override the images if they exist""") 
        self.parser.add_argument('--n_patches', default=2)
        self.parser.add_argument('--linewidth', default=1)
        self.parser.add_argument('--markersize', default=1)
        self.parser.add_argument('--min_scale', default=0)
        self.parser.add_argument('--max_scale', default=100)



        
    def parse(self):

        args = self.parser.parse_args()

        return args


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary
    return config