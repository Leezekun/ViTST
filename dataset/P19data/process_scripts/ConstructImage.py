import os
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

max_hours = 60

def construct_demogr_description(static_demogr, length):
    desc = []
    # age
    if static_demogr[0]:
        desc.append(f"{int(static_demogr[0])} years old")
    # gender
    if int(static_demogr[1]) == 0:
        desc.append("female")
    elif int(static_demogr[1]) == 1:
        desc.append("male") 
    # icu type
    icu = "ICU"
    if static_demogr[2] or static_demogr[3]:
        if static_demogr[2]:
            icu = "medical ICU"
        if static_demogr[3]:
            icu = "surgical ICU"
    desc.append(f"went the {icu} {round(static_demogr[3])} hours after hospital admit, had stayed there for {static_demogr[4]} hours, and has the physiological data within {length} hours")
    
    if desc:
        desc = "A patient is " + ", ".join(desc) + "."
    else:
        desc = ""

    return desc

def standardize(x, mean, std):
    return (x-mean)/(std+1e-18)

def draw_image(pid, ts_orders, ts_values, ts_times, ts_params, ts_scales, override, differ, outlier, interpolation, order,
                image_size, grid_layout,
                linestyle, linewidth, markersize,
                ts_marker_mapping, ts_color_mapping, ts_idx_mapping):
    
    # set matplotlib param
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

    # save path
    base_path = f"{grid_height}*{grid_width}_images"
    if interpolation:
        base_path = "interpolation_" + base_path
    if differ:
        base_path = "differ_" + base_path
    if order:
        base_path = f"order_" + base_path
    if outlier:
        base_path = f"{outlier}_" + base_path
    base_path = "../processed_data/" + base_path

    if not os.path.exists(base_path): os.mkdir(base_path)
    img_path = os.path.join(base_path, f"{pid}.png")
    if os.path.exists(img_path):
        if not override:
            return []

    drawed_params = []
    
    max_hours, num_params = ts_values.shape[0], ts_values.shape[1] # (60, 34)
    for idx, param_idx in enumerate(ts_orders): # ts_desc: (215, 36)
        
        param = ts_params[param_idx]
        ts_value = ts_values[:, param_idx]

        # the scale of x, y axis
        param_scale_x = [1, max_hours] # starting from 1
        param_scale_y = ts_scales[param_idx]
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0]-0.5, param_scale_y[0]+0.5]

        ts_time = np.array(ts_times).reshape(-1,1)
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

        # plt.subplot(grid_height, grid_width, param_idx+1) # 6*6
        plt.subplot(grid_height, grid_width, idx+1) # 6*6
        
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
            

def construct_image(
    linestyle="-", linewidth=1, markersize=2, 
    override=False, 
    differ=False, 
    outlier=None,
    interpolation=True,
    order=False,
    image_size=None,
    grid_layout=(6,6)
    ):
    
    # load data
    Pdict_list = np.load(f'../processed_data/PT_dict_list_6.npy', allow_pickle=True)
    arr_outcomes = np.load('../processed_data/arr_outcomes_6.npy', allow_pickle=True)
    num_samples = len(Pdict_list)
    print(f"{num_samples} patients in total!") 
    print(Pdict_list[0].keys()) # ['id', 'length', 'extended_static', 'arr', 'time']

    # load param descriptions
    f = open('../processed_data/ts_params_desc.json', 'r')
    param_detailed_description = json.load(f)
    ts_params = list(param_detailed_description.keys())
    num_ts_params = len(ts_params) # 34
    print(f"{num_ts_params} parameters!")

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

    with open('../processed_data/param_marker_mapping.json', 'w') as f:
        json.dump(ts_marker_mapping, f)
    with open('../processed_data/param_idx_mapping.json', 'w') as f:
        json.dump(ts_idx_mapping, f)
    with open('../processed_data/param_color_mapping.json', 'w') as f:
        json.dump(ts_color_mapping, f)
    
    # start constructing imagedicts
    ImageDict_list = []
    demogr_lengths = []
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    # first round, find the mean and std and scale for each param across all the data
    all_ts_values = [[] for _ in range(num_ts_params)]
    stat_ts_values = np.ones(shape=(num_ts_params, 10)) # mean, std, min, max
    for idx, p in tqdm(enumerate(Pdict_list)):
        ts_values = p['arr'] #(60, 34)
        for param_idx in range(num_ts_params): # ts_desc: (60, 34)
            ts_value = ts_values[:, param_idx]
            ts_value = np.array(ts_value).reshape(-1,1)
            # handling missing value
            ts_value = ts_value[ts_value != 0]
            all_ts_values[param_idx].extend(list(ts_value))

    # the order of params
    if order:
        ts_value_nums = [len(_) for _ in all_ts_values]
        ts_orders = np.argsort(ts_value_nums)[::-1]
    else:
        ts_orders = list(range(num_ts_params))
    
    for param_idx in range(num_ts_params): # ts_desc: (60, 34)
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

    # second round, draw the image for each patient
    for idx, p in tqdm(enumerate(Pdict_list)):

        pid = p['id'].split("\\")[-1].split(".psv")[0].strip() # Data_physionet19/trainingA\\p000001.psv -> p000001
        length = p['length']
        ts_times = p['time']
        ts_values = p['arr'] #(215, 36)

        # label and target for generation
        label = int(arr_outcomes[idx])
        label_name = "no" if label == 0 else "yes"
        
        static_demogr = p['extended_static']
        demogr_desc = construct_demogr_description(static_demogr, length)
        demogr_length = len(tokenizer(demogr_desc)[0])
        demogr_lengths.append(demogr_length)

        # normalize the values
        if not outlier:
            ts_scales = stat_ts_values[:,2:4] # no removal
        elif outlier == "iqr":
            ts_scales = stat_ts_values[:,4:6] # iqr
        elif outlier == "sd":
            ts_scales = stat_ts_values[:,6:8] # sd
        elif outlier == "mzs":
            ts_scales = stat_ts_values[:,8:10] # mzs

        # draw the image for each p
        drawed_params = draw_image(pid, ts_orders, ts_values, ts_times, ts_params, ts_scales, override, differ, outlier, interpolation, order,
                image_size, grid_layout,
                linestyle, linewidth, markersize,
                ts_marker_mapping, ts_color_mapping, ts_idx_mapping)
        
        ImageDict = {
            "id": pid, 
            "param_num": len(drawed_params), 
            "text": demogr_desc,
            "label": label,
            "label_name": label_name
            }
        
        ImageDict_list.append(ImageDict)


if __name__ == "__main__":
    construct_image(
        linestyle="-", linewidth=1, markersize=2, 
        override=False, 
        differ=True, 
        outlier=None,
        interpolation=True,
        order=True,
        image_size=None,
        grid_layout=(6,6)
        )

            




                



    
