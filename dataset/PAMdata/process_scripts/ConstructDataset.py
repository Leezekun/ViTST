from cmath import nan
from email.mime import image
from multiprocessing.util import is_exiting
from xmlrpc.client import FastMarshaller
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
from transformers import AutoTokenizer


def standardize(x, mean, std):
    return (x-mean)/(std+1e-18)


def draw_image(pid, ts_values, ts_scales, override, differ, outlier, interpolation,
                missing_ratio, 
                feature_removal_level,
                image_size,
                grid_layout, 
                linestyle, linewidth, markersize,
                ts_marker_mapping, ts_color_mapping, ts_idx_mapping):

    max_hours, num_params = ts_values.shape[0], ts_values.shape[1]

    # set matplotlib param
    assert grid_layout[0] * grid_layout[1] >= num_params
    grid_height = grid_layout[0]
    grid_width = grid_layout[1]
    if image_size is None:
        cell_height = 64
        cell_width = 64
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 100
    plt.rcParams['savefig.dpi'] = dpi  # default=100
    plt.rcParams['figure.figsize'] = (img_width / dpi, img_height / dpi)
    plt.rcParams['figure.frameon'] = False

    base_path = f"{grid_height}*{grid_width}_images"
    if interpolation:
        base_path = "interpolation_" + base_path
    if differ:
        base_path = "differ_" + base_path
    if outlier:
        base_path = f"{outlier}_" + base_path
    if missing_ratio in [0.1,0.2,0.3,0.4,0.5]:
        base_path = f"{feature_removal_level}_{missing_ratio}_" + base_path
    base_path = "../processed_data/" + base_path

    if not os.path.exists(base_path): os.mkdir(base_path)
    img_path = os.path.join(base_path, f"{pid}.png")
    if os.path.exists(img_path):
        if not override:
            return []

    drawed_params = []

    max_hours, num_params = ts_values.shape[0], ts_values.shape[1]
    num_missing_features = round(missing_ratio * num_params)
     
    # leave sensor out
    if feature_removal_level == 'random' and missing_ratio > 0:
        selected_param_idxs = [i for i in range(num_params)]
        random.shuffle(selected_param_idxs)
        dropped_param_nums = round(missing_ratio*num_params)
        keeped_param_nums = num_params-dropped_param_nums
        selected_param_idxs = selected_param_idxs[:keeped_param_nums]
    elif feature_removal_level == 'set' and missing_ratio > 0:
        density_score_indices = np.load('../../raindrop/baselines/saved/IG_density_scores_PAM.npy', allow_pickle=True)[:, 0]
        dropped_param_idxs = set(list(density_score_indices[:num_missing_features].astype(int)))
        all_param_idxs = set([i for i in range(num_params)])
        selected_param_idxs = list(all_param_idxs.difference(dropped_param_idxs))
    else:
        selected_param_idxs = [i for i in range(num_params)]
    

    for param_idx in selected_param_idxs: # ts_desc: (215, 36)
    
        param = str(param_idx)

        ts_time = np.arange(0, max_hours, dtype=float)
        ts_value = ts_values[:, param_idx]

        # the scale of x, y axis
        param_scale_x = [0, max_hours]
        param_scale_y = ts_scales[param_idx]
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0]-0.5, param_scale_y[0]+0.5]
            
        ts_time = np.array(ts_time).reshape(-1,1)
        ts_value = np.array(ts_value).reshape(-1,1)
        # handling missing value and extreme values
        kept_index = (ts_value != 0) & (ts_value < param_scale_y[1]) & (ts_value > param_scale_y[0])
        removed_index = (ts_value == 0) | (ts_value > param_scale_y[1]) | (ts_value < param_scale_y[0])
        if interpolation:
            ts_time = ts_time[kept_index]
            ts_value = ts_value[kept_index]
        else:
            ts_time[removed_index] = np.nan
            ts_value[removed_index] = np.nan

        ##### draw the plot for each parameter 
        param_marker = ts_marker_mapping[param]
        param_color = ts_color_mapping[param]
        param_idx = ts_idx_mapping[param]

        plt.subplot(grid_layout[0], grid_layout[1], param_idx+1)
        
        if differ: # using different colors and markers
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, markersize=markersize, color=param_color, marker="*")
        else:
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth)

        plt.xlim(param_scale_x)
        plt.ylim(param_scale_y)
        plt.xticks([])
        plt.yticks([])

        drawed_params.append(param)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()
    return drawed_params
            
def construct_prompt(feature, missing_ratio):
    prompt = []
    param_prompts = []
    feature_length, feature_dim = np.array(feature).shape # (600, 17)
    prompt.append(f"From {0} to {feature_length}, we have ")

    # random drop some params
    selected_param_idxs = [i for i in range(feature_dim)]
    random.shuffle(selected_param_idxs)
    dropped_param_nums = round(missing_ratio*feature_dim)
    keeped_param_nums = feature_dim-dropped_param_nums
    selected_param_idxs = selected_param_idxs[:keeped_param_nums]

    for param_idx in selected_param_idxs:
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

def construct_dataset(
    construct_image_data=False,
    construct_prompt_data=False,
    grid_layout=(4,5),
    image_size=None,
    linestyle="-", linewidth=1, markersize=2, 
    differ=True, 
    override=True, 
    outlier=None,
    interpolation=True,
    missing_ratios=[0.],
    feature_removal_level='random'
    ):
    
    # load data
    Pdict_list = np.load(f'../processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load('../processed_data/arr_outcomes.npy', allow_pickle=True)
    num_samples = len(Pdict_list)
    print(f"{num_samples} patients in total!") 
    
    print(Pdict_list[0].shape) # (600,17)
    max_hours = Pdict_list[0].shape[0]
    num_params = Pdict_list[0].shape[1]
    ts_params = [str(i) for i in range(num_params)] # 0,1,2,3,4,5,6,7 (8 classes)

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
    for idx, param in enumerate(ts_params):
        if idx < num_markers:
            ts_marker_mapping[param] = plt_markers[idx]
        else: # if not enough markers, use (num_sides, 0/1/2, angles) markers
            marker = (int((idx-num_markers)/3)+3, int((idx-num_markers)%3)) # starting from (3,0)
            ts_marker_mapping[param] = marker
        ts_color_mapping[param] = plt_colors[idx]
        ts_idx_mapping[param] = idx

    # start constructing image dicts
    ImageDict_list, ms10_ImageDict_list, ms20_ImageDict_list, ms30_ImageDict_list, ms40_ImageDict_list, ms50_ImageDict_list = [], [], [], [], [], []
    PromptDict_list, ms10_PromptDict_list, ms20_PromptDict_list, ms30_PromptDict_list, ms40_PromptDict_list, ms50_PromptDict_list = [], [], [], [], [], []
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    # first round, find the mean and std for each param across all the data
    all_ts_values = [[] for _ in range(num_params)]
    stat_ts_values = np.ones(shape=(num_params, 10)) # mean, std, min, max
    for idx, p in tqdm(enumerate(Pdict_list)):
        ts_values = p #(600,17)
        for param_idx in range(num_params): # ts_desc: (60, 34)
            ts_value = ts_values[:, param_idx]
            ts_value = np.array(ts_value).reshape(-1,1)
            ts_value = ts_value[ts_value != 0] # remove the missing values
            all_ts_values[param_idx].extend(list(ts_value))
    for param_idx in range(num_params): # ts_desc: (60, 34)
        param_ts_value = np.array(all_ts_values[param_idx])
        stat_ts_values[param_idx,0] = param_ts_value.mean()
        stat_ts_values[param_idx,1] = param_ts_value.std()
        stat_ts_values[param_idx,2] = param_ts_value.min()
        stat_ts_values[param_idx,3] = param_ts_value.max()

        """
        1. remove outliers with boxplot
        """
        q1 = np.percentile(param_ts_value, 25)
        q3 = np.percentile(param_ts_value, 75)
        med = np.median(param_ts_value)
        iqr = q3-q1
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
        stat_ts_values[param_idx,4] = lower_bound
        stat_ts_values[param_idx,5] = upper_bound
        param_ts_value1 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
        outlier_ratio = 1 - (len(param_ts_value1) / len(param_ts_value))
        print(f"{param_idx}, {outlier_ratio}")
        
        """
        2. remove outliers with standard deviation
        """
        med = np.median(param_ts_value)
        std = np.std(param_ts_value)
        upper_bound = med + (3*std)
        lower_bound = med - (3*std)
        stat_ts_values[param_idx,6] = lower_bound
        stat_ts_values[param_idx,7] = upper_bound
        param_ts_value2 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
        outlier_ratio = 1 - (len(param_ts_value2) / len(param_ts_value))
        print(f"{param_idx}, {outlier_ratio}")

        """
        3. remove outliers with modified z-score
        """
        med = np.median(param_ts_value)
        deviation_from_med = param_ts_value - med
        mad = np.median(np.abs(deviation_from_med))
        # modified_z_score = (deviation_from_med / mad)*0.6745
        lower_bound = (-3.5/0.6745)*mad + med
        upper_bound = (3.5/0.6745)*mad + med
        stat_ts_values[param_idx,8] = lower_bound
        stat_ts_values[param_idx,9] = upper_bound
        param_ts_value3 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
        outlier_ratio = 1 - (len(param_ts_value3) / len(param_ts_value))
        print(f"{param_idx}, {outlier_ratio}")

    # second round, draw the image and prompt for each sample
    for idx, p in tqdm(enumerate(Pdict_list)):
        
        pid = idx
        ts_values = p # (600, 17) time_length = 600, num_param = 17
        label = arr_outcomes[idx] # 0-7, 8 classes

        # normalize the values
        if not outlier:
            ts_scales = stat_ts_values[:,2:4] # no removal
        elif outlier == "iqr":
            ts_scales = stat_ts_values[:,4:6] # iqr
        elif outlier == "sd":
            ts_scales = stat_ts_values[:,6:8] # sd
        elif outlier == "mzs":
            ts_scales = stat_ts_values[:,8:10] # mzs

        if construct_image_data:
            # draw the image for each p
            for missing_ratio in missing_ratios:
                drawed_params = draw_image(pid, ts_values, ts_scales, override, differ, outlier, interpolation,
                        missing_ratio, 
                        feature_removal_level,
                        image_size,
                        grid_layout, 
                        linestyle, linewidth, markersize,
                        ts_marker_mapping, ts_color_mapping, ts_idx_mapping)
                ImageDict = {
                    "id": pid, 
                    "param_num": len(drawed_params), 
                    "text": "",
                    "label": int(label[0]),
                    "label_name": str(int(label[0])),
                    }
                if missing_ratio == 0:
                    ImageDict_list.append(ImageDict)
                elif missing_ratio == 0.1:
                    ms10_ImageDict_list.append(ImageDict)
                elif missing_ratio == 0.2:
                    ms20_ImageDict_list.append(ImageDict)
                elif missing_ratio == 0.3:
                    ms30_ImageDict_list.append(ImageDict)
                elif missing_ratio == 0.4:
                    ms40_ImageDict_list.append(ImageDict)
                elif missing_ratio == 0.5:
                    ms50_ImageDict_list.append(ImageDict)

        if construct_prompt_data:
            # construct the images for each sample
            for missing_ratio in missing_ratios:
                prompt, param_prompts = construct_prompt(ts_values, missing_ratio)
                prompt_length = len(tokenizer(prompt)[0])
                param_num = len(param_prompts)
                PromptDict = {
                "id": id, 
                "prompt": prompt,
                "prompt_length": prompt_length,
                "param_num": param_num,
                "label": int(label[0]),
                "label_name": str(int(label[0])),
                "target": str(int(label[0])),
                }
                if missing_ratio == 0:
                    PromptDict_list.append(PromptDict)
                elif missing_ratio == 0.1:
                    ms10_PromptDict_list.append(PromptDict)
                elif missing_ratio == 0.2:
                    ms20_PromptDict_list.append(PromptDict)
                elif missing_ratio == 0.3:
                    ms30_PromptDict_list.append(PromptDict)
                elif missing_ratio == 0.4:
                    ms40_PromptDict_list.append(PromptDict)
                elif missing_ratio == 0.5:
                    ms50_PromptDict_list.append(PromptDict)

    if construct_image_data:
        for idx, ImageDict_list in enumerate([ImageDict_list, ms10_ImageDict_list, ms20_ImageDict_list, ms30_ImageDict_list, ms40_ImageDict_list, ms50_ImageDict_list]):
            print(len(ImageDict_list))
            if len(ImageDict_list) > 0:
                if idx == 0:
                    save_path = f'../processed_data/ImageDict_list.npy'
                else:
                    missing_ratio = missing_ratios[idx]
                    save_path = f'../processed_data/{feature_removal_level}_{missing_ratio}_ImageDict_list.npy'
                np.save(save_path, ImageDict_list)
                print(f"Save data in {save_path}")

    if construct_prompt_data:
        for idx, PromptDict_list in enumerate([PromptDict_list, ms10_PromptDict_list, ms20_PromptDict_list, ms30_PromptDict_list, ms40_PromptDict_list, ms50_PromptDict_list]):
            print(len(PromptDict_list))
            if len(PromptDict_list) > 0:
                if idx == 0:
                    save_path = f'../processed_data/PromptDict_list.npy'
                else:
                    missing_ratio = missing_ratios[idx]
                    save_path = f'../processed_data/{feature_removal_level}_{missing_ratio}_PromptDict_list.npy'
                np.save(save_path, PromptDict_list)
                print(f"Save data in {save_path}")


if __name__ == "__main__":
    construct_dataset(
        construct_image_data=True,
        construct_prompt_data=False,
        grid_layout=(4,5),
        image_size=None,
        linestyle="-", linewidth=0.5, markersize=0.5, 
        differ=True, 
        override=False, 
        outlier=None,
        interpolation=True,
        missing_ratios=[0.]
        # missing_ratios=[0.,0.1,0.2,0.3,0.4,0.5],
        # feature_removal_level='set'
        )
            




                



    
