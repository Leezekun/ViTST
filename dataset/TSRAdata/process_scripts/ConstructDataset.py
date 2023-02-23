import os
from multiprocessing import Pool, cpu_count
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import json
import copy

from datasets import utils
from options import Options, setup

from utils import split_dataset, load_from_tsfile_to_dataframe


def standardize(x, mean, std):
    return (x-mean)/(std+1e-18)

class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        #self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))
        
        print(f"Loading data from {input_paths}")

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                 replace_missing_vals_with='NaN')
            labels_df = None

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            print("Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            print("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class ClassifiregressionDataset(Dataset):

    def __init__(self, data, indices):
        super(ClassifiregressionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]
        self.labels_df = self.data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        # return torch.from_numpy(X), torch.from_numpy(y), self.IDs[ind]
        return X, y, self.IDs[ind]

    def __len__(self):
        return len(self.IDs)


def draw_image(id, save_path, feature, ts_scales, min_scale, max_scale,
            override, 
            differ,
            image_size,
            linestyle, linewidth, markersize,
            ts_marker_mapping, ts_color_mapping, ts_idx_mapping):

    feature_length, feature_dim = np.array(feature).shape

    for i in range(100):
        if i**2 >= feature_dim:
            grid_height = i
            grid_width = i
            break

    grid_layout = [grid_height, grid_width]
    if image_size is None:
        cell_height = 64
        cell_width = 64
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 100
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (img_width/dpi, img_height/dpi)
    plt.rcParams['figure.frameon'] = False

    # save img path
    if min_scale and max_scale:
        if differ:
            save_image_path = os.path.join(save_path, f"differ_grid{grid_layout[0]}*{grid_layout[1]}_{img_height}*{img_width}_{min_scale}-{max_scale}_images")
        else:
            save_image_path = os.path.join(save_path, f"grid{grid_layout[0]}*{grid_layout[1]}_{img_height}*{img_width}_{min_scale}-{max_scale}_images")
    else:
        if differ:
            save_image_path = os.path.join(save_path, f"differ_grid{grid_layout[0]}*{grid_layout[1]}_{img_height}*{img_width}_images")
        else:
            save_image_path = os.path.join(save_path, f"grid{grid_layout[0]}*{grid_layout[1]}_{img_height}*{img_width}_images")

    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    img_path = os.path.join(save_image_path, f"{id}.png")
    if os.path.exists(img_path):
            if not override:
                return []

    drawed_params = []
    
    for param_idx in range(feature_dim):
        param = str(param_idx)

        ts_value = feature[:,param_idx]
        ts_value[ts_value == 0] = np.nan # remove the missing values
        ts_time = np.arange(0,feature_length)

        # the scale of x, y axis
        param_scale_x = [0, feature_length]
        param_scale_y = ts_scales[param_idx]
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0]-0.5, param_scale_y[0]+0.5]# the scale of x, y axis

        plt.subplot(grid_layout[0], grid_layout[1], param_idx+1)

        if differ: # using different colors and markers
            ##### draw the plot for each parameter 
            param_marker = ts_marker_mapping[param]
            param_color = ts_color_mapping[param]
            param_idx = ts_idx_mapping[param]
            # plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, markersize=markersize, color=param_color, marker="*")
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
        else:
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth)

        plt.xlim(param_scale_x)
        plt.ylim(param_scale_y)
        plt.xticks([])
        plt.yticks([])

        drawed_params.append(param_idx)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()

    return drawed_params 


def construct_prompt(feature):
    prompt = []
    param_prompts = []
    feature_length, feature_dim = np.array(feature).shape # (600, 17)
    prompt.append(f"From {0} to {feature_length}, we have ")
    for param_idx in range(feature_dim):
        ts_values = feature[:,param_idx]
        if ts_values.sum() == 0: continue
        ts_values = ts_values[ts_values!=0] # remove the missing values
        
        prompt_ts_values = []
        for value in ts_values:
           if float(value).is_integer(): value = int(value) # 23.0 -> 23
           prompt_ts_values.append(str(value))
        prompt_ts_values = " ".join(prompt_ts_values)
        prompt.append(f"x{param_idx} = {prompt_ts_values};")
        param_prompts.append(f"x{param_idx} = {prompt_ts_values};")

    prompt.append("what should be y?")
    prompt = " ".join(prompt)
    return prompt, param_prompts 


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)
    
    # load train and val data
    train_val_data = TSRegressionArchive(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    train_val_indices = train_val_data.all_IDs 
    feature_dim = train_val_data.feature_df.shape[1]

    # load test data
    test_data = TSRegressionArchive(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
    test_indices = test_data.all_IDs

    # initialize datasets
    train_val_dataset = ClassifiregressionDataset(train_val_data, train_val_indices)
    test_dataset = ClassifiregressionDataset(test_data, test_indices)

    # record feature dim and ts_params
    for i, (feature, label, id) in enumerate(train_val_dataset):
        feature_length, feature_dim = np.array(feature).shape
        break
    print(feature_length, feature_dim)
    ts_params = [str(i) for i in range(feature_dim)]

    # load markers and colors
    f = open('../processed_data/plt_markers_desc.json', 'r')
    plt_markers_description = json.load(f)
    f = open('../processed_data/plt_colors_desc.json', 'r')
    plt_colors_description = json.load(f)

    plt_markers = list(plt_markers_description.keys())
    num_markers = len(plt_markers)
    print(f"{num_markers} markers!")

    plt_colors = list(plt_colors_description.keys())
    num_colors = len(plt_colors)
    print(f"{num_colors} colors!")

    # construct the mapping from param to marker, color, and idx
    ts_marker_mapping = {}
    ts_idx_mapping = {}
    ts_color_mapping = {}
    try:
        for idx, param in enumerate(ts_params):
            if idx < num_markers:
                ts_marker_mapping[param] = plt_markers[idx]
            else: # if not enough markers, use (num_sides, 0/1/2, angles) markers
                marker = (int((idx-num_markers)/3)+3, int((idx-num_markers)%3)) # starting from (3,0)
                ts_marker_mapping[param] = marker
            ts_color_mapping[param] = plt_colors[idx]
            ts_idx_mapping[param] = idx
    except:
        pass

    # start constructing images and prompts
    save_path = os.path.join(args.data_dir, "processed_data")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ImageDict_list = []
    PromptDict_list = []
    arr_outcomes = []
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    # first round, find the mean and std for each param across all the data
    min_scale = args.min_scale
    max_scale = args.max_scale
    all_ts_values = [[] for _ in range(feature_dim)]
    stat_ts_values = np.ones(shape=(feature_dim, 4)) # mean, std, min, max
    for i, dataset in enumerate([train_val_dataset, test_dataset]):
        for j, (feature, label, id) in tqdm(enumerate(dataset)):
            feature_length, feature_dim = np.array(feature).shape
            for param_idx in range(feature_dim): # ts_desc: (60, 34)
                ts_value = feature[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1,1)
                ts_value = ts_value[ts_value != 0] # remove the missing values
                all_ts_values[param_idx].extend(list(ts_value))
    for param_idx in range(feature_dim):
        param_ts_value = np.array(all_ts_values[param_idx])
        stat_ts_values[param_idx,0] = param_ts_value.mean()
        stat_ts_values[param_idx,1] = param_ts_value.std()
        # stat_ts_values[param_idx,2] = param_ts_value.min()
        # stat_ts_values[param_idx,3] = param_ts_value.max()
        stat_ts_values[param_idx,2] = np.percentile(param_ts_value, min_scale)
        stat_ts_values[param_idx,3] = np.percentile(param_ts_value, max_scale)

    for i, dataset in enumerate([train_val_dataset, test_dataset]):
        for j, (feature, label, id) in tqdm(enumerate(dataset)):
        
            feature_length, feature_dim = np.array(feature).shape
            
            # concat the test dataset at the end of train_val dataset
            if i == 1: id += len(train_val_indices) 

            ts_scales = stat_ts_values[:,-2:]
            ts_values = feature
            
            # draw the images
            drawed_params = draw_image(id, save_path, ts_values, ts_scales, min_scale, max_scale,
                                        override=True, 
                                        differ=config["differ"],
                                        image_size=None,
                                        linestyle="-", 
                                        linewidth=config["linewidth"], 
                                        markersize=config["markersize"],
                                        ts_marker_mapping=ts_marker_mapping, 
                                        ts_color_mapping=ts_color_mapping, 
                                        ts_idx_mapping=ts_idx_mapping)            
            
            ImageDict = {
            "id": id, 
            "param_num": len(drawed_params), 
            "text": "",
            "label": label[0],
            "label_name": str(label[0])
            }
            ImageDict_list.append(ImageDict)

            # construct the prompt
            # prompt, param_prompts = construct_prompt(feature)
            # prompt_length = len(tokenizer(prompt)[0])
            # param_num = len(param_prompts)

            # PromptDict = {
            # "id": id, 
            # "prompt": prompt,
            # "prompt_length": prompt_length,
            # "param_num": param_num,
            # "label": label[0],
            # "label_name": str(label[0]),
            # "target": str(label[0]),
            # }
            # PromptDict_list.append(PromptDict)

            # labels
            arr_outcomes.append(label)
        
    # save the data
    np.save(os.path.join(save_path, 'ImageDict_list.npy'), ImageDict_list)
    print(f"Save data in ImageDict_list.npy")
    # np.save(os.path.join(save_path, 'PromptDict_list.npy'), PromptDict_list)
    # print(f"Save data in PromptDict_list.npy")
    np.save(os.path.join(save_path, 'arr_outcomes.npy'), arr_outcomes)
    print(f"Save data in arr_outcomes.npy")

    # Split dataset
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = train_val_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    """
    Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    """
    if config['val_ratio'] > 0:
        train_indices, val_indices, _ = split_dataset(data_indices=train_val_data.all_IDs,
                                                                validation_method=validation_method,
                                                                n_splits=config['n_splits'],
                                                                validation_ratio=config['val_ratio'],
                                                                test_set_ratio=0.0,  # used only if test_indices not explicitly specified
                                                                test_indices=None, # don't split a part from train data as test set
                                                                random_seed=1337,
                                                                labels=labels)
    else:
        train_indices = [train_val_indices for _ in range(config['n_splits'])] # `split_dataset` returns a list of indices *per fold/split*
        val_indices = [[] for _ in range(config['n_splits'])] # `split_dataset` returns a list of indices *per fold/split*
    test_indices += len(train_val_indices) # concat the test data at the end of train and val data

    save_split_path = os.path.join(args.data_dir, "splits")
    if not os.path.exists(save_split_path): os.mkdir(save_split_path)

    for i in range(config['n_splits']):
        
        train_indice = list(map(int, train_indices[i]))
        val_indice = list(map(int, val_indices[i]))
        test_indice = list(map(int, test_indices))

        print("{} samples may be used for training".format(len(train_indice)))
        print("{} samples will be used for validation".format(len(val_indice)))
        print("{} samples will be used for testing".format(len(test_indice)))

        np.save(os.path.join(save_split_path, f'split_{i+1}.npy'), (train_indice, val_indice, test_indice))










    




        
       


        


    