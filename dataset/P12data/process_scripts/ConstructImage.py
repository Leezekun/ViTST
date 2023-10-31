import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
Dataset configurations 
"""
max_tmins = 48*60 # 48 hours

param_detailed_description = {
    "ALP": "Alkaline phosphatase (IU/L)",
    "ALT": "Alanine transaminase (IU/L)",
    "AST": "Aspartate transaminase (IU/L)",
    "Albumin": "Albumin (g/dL)",
    "BUN": "Blood urea nitrogen (mg/dL)",
    "Bilirubin": "Bilirubin (mg/dL)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "Creatinine": "Serum creatinine (mg/dL)",
    "DiasABP": "Invasive diastolic arterial blood pressure (mmHg)",
    "FiO2": "Fractional inspired O2 (0-1)",
    "GCS": "Glasgow Coma Score (3-15)",
    "Glucose" :"Serum glucose (mg/dL)",
    "HCO3": "Serum bicarbonate (mmol/L)",
    "HCT": "Hematocrit (%)",
    "HR": "Heart rate (bpm)",
    "K": "Serum potassium (mEq/L)",
    "Lactate": "Lactate (mmol/L)",
    "MAP": "Invasive mean arterial blood pressure (mmHg)",
    "MechVent": "Mechanical ventilation respiration (0:false, or 1:true)",
    "Mg": "Serum magnesium (mmol/L)",
    "NIDiasABP": "Non-invasive diastolic arterial blood pressure (mmHg)",
    "NIMAP": "Non-invasive mean arterial blood pressure (mmHg)",
    "NISysABP": "Non-invasive systolic arterial blood pressure (mmHg)",
    "Na": "Serum sodium (mEq/L)", 
    "PaCO2": "partial pressure of arterial CO2 (mmHg)",
    "PaO2": "Partial pressure of arterial O2 (mmHg)",
    "Platelets": "Platelets(cells/nL)",
    "RespRate": "Respiration rate (bpm)",
    "SaO2": "O2 saturation in hemoglobin (%)",
    "SysABP": "Invasive systolic arterial blood pressure (mmHg)",
    "Temp": "Temperature (°C)",
    "TroponinI": "Troponin-I (μg/L)",
    "TroponinT": "Troponin-T (μg/L)",
    # "TropI": "Troponin-I (μg/L)",
    # "TropT": "Troponin-T (μg/L)",
    # "TroponinI": "Troponin-I (μg/L)",
    "Urine": "Urine output (mL)",
    "WBC": "White blood cell count (cells/nL)",
    "pH": "Arterial pH (0-14)",
    }

color_detailed_description = {
    "aqua": "1",
    "azure": "2",
    "beige": "3",
    "black": "4",
    "blue": "5",
    "brown": "6",
    "chartreuse": "7",
    "chocolate": "8",
    "coral": "9",
    "crimson": "10",
    "cyan": "11",
    "darkblue": "12",
    "darkgreen": "13",
    "fuchsia": "14",
    "gold": "15",
    "green": "16",
    "grey": "17",
    "indigo": "18",
    "ivory": "19",
    "khaki": "20",
    "lavender": "21",
    "lightblue": "22",
    "lightgreen": "23",
    "magenta": "24",
    "maroon": "25",
    "navy": "26",
    "olive": "27",
    "orange": "28",
    "orchid": "29",
    "pink": "30",
    "plum": "31",
    "purple": "32",
    "red": "33",
    "salmon": "34",
    "sienna": "35",
    "silver": "36",
    "tan": "37",
    "teal": "38",
    "yellow": "39",
    "yellowgreen": "40"
}

"""
Code
"""
def construct_demogr_description(static_demogr):
    desc = []
    # age
    if static_demogr[0]:
        desc.append(f"{int(static_demogr[0])} years old")
    
    # gender
    if int(static_demogr[1]) == 0:
        desc.append("female")
    elif int(static_demogr[1]) == 1:
        desc.append("male") 

    # height
    if static_demogr[2] > 0:
        desc.append(f"{static_demogr[0]} cm")

    # weight
    if static_demogr[4] > 0:
        desc.append(f"{static_demogr[4]} kg")
    
    # icu type
    if static_demogr[3] > 0:
        if int(static_demogr[3]) == 1:
            icu = "coronary care unit"
            desc.append(f"stayed in {icu}")
        elif int(static_demogr[3]) == 2:
            icu = "cardiac surgery recovery unit"
            desc.append(f"stayed in {icu}")
        elif int(static_demogr[3]) == 3:
            icu = "medical ICU"
            desc.append(f"stayed in {icu}")
        elif int(static_demogr[3]) == 4:
            icu = "surgical ICU"
            desc.append(f"stayed in {icu}")
            
    if desc:
        desc = "A patient is " + ", ".join(desc) + "."
    else:
        desc = ""

    return desc

def draw_image(pid, split_idx, ts_orders, ts_values, ts_times, ts_params, ts_scales, 
               override, differ, outlier, interpolation, order,
                image_size, grid_layout,
                linestyle, linewidth, marker, markersize,
                ts_color_mapping, ts_idx_mapping):
        
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
    base_path = f"{linestyle}*{linewidth}_{marker}*{markersize}_{grid_height}*{grid_width}_{img_height}*{img_width}_split{split_idx}_images"

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

    # find the information across all the patients
    max_hours, num_params = ts_values.shape[0], ts_values.shape[1]
    for idx, param_idx in enumerate(ts_orders): # ts_desc: (215, 36)
        
        param = ts_params[param_idx]
        ts_value = ts_values[:, param_idx]

        # the scale of x, y axis
        param_scale_x = [0, max_tmins]
        param_scale_y = ts_scales[param_idx]
        # only one value, expand the y axis
        if param_scale_y[0] == param_scale_y[1]:
            param_scale_y = [param_scale_y[0]-0.5, param_scale_y[0]+0.5]

        ts_time = np.array(ts_times).reshape(-1,1)
        ts_value = np.array(ts_value).reshape(-1,1)
        # handling missing value and extreme values
        kept_index = (ts_value != 0)
        removed_index = (ts_value == 0)
        if interpolation:
            ts_time = ts_time[kept_index]
            ts_value = ts_value[kept_index]
        else:
            ts_time[removed_index] = np.nan
            ts_value[removed_index] = np.nan
        # handling extreme values
        min_index = (ts_value < param_scale_y[0])
        ts_value[min_index] = param_scale_y[0]
        # handling extreme values
        max_index = (ts_value > param_scale_y[1])
        ts_value[max_index] = param_scale_y[1]

        ##### draw the plot for each parameter
        # param_marker = ts_marker_mapping[param]
        param_color = ts_color_mapping[param]
        param_idx = ts_idx_mapping[param]

        # plt.subplot(grid_height, grid_width, param_idx+1) # 6*6
        plt.subplot(grid_height, grid_width, idx+1) # 6*6
        if differ: # using different colors and markers
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, color=param_color)
        else:
            plt.plot(ts_time, ts_value, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,)

        # set the scale for x, y axis
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
    linestyle="-", linewidth=1, marker="*", markersize=2, 
    override=False, 
    differ=False, 
    outlier=None,
    interpolation=True,
    order=False,
    grid_layout=(6,6),
    image_size=None
    ):
    
    # load data
    Pdict_list = np.load(f'../processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load(f'../processed_data/arr_outcomes.npy', allow_pickle=True)
    ts_params = np.load(f'../processed_data/ts_params.npy', allow_pickle=True)
    
    num_samples = len(Pdict_list)
    print(f"{num_samples} patients in total!") 
    print(list(Pdict_list[0].keys())) # ['id', 'static', 'extended_static', 'arr', 'time', 'length']

    num_ts_params = len(ts_params) # 36
    print(f"{num_ts_params} parameters!")

    plt_colors = list(color_detailed_description.keys())
    num_colors = len(plt_colors)
    print(f"{num_colors} colors!")
    
    """
    for random param color exp
    """
    if num_ts_params > num_colors:
        plt_colors = []
        rs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(rs) # from 0 to 1
        gs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(gs) # from 0 to 1
        bs = list(np.linspace(0.0, 1.0, num_ts_params))
        random.shuffle(bs) # from 0 to 1
        for idx in range(num_ts_params):
            color = (rs[idx], gs[idx], bs[idx])
            plt_colors.append(color)

    # construct the mapping from param to marker, color, and idx
    ts_idx_mapping = {}
    ts_color_mapping = {}
    for idx, param in enumerate(ts_params):
        ts_color_mapping[param] = plt_colors[idx]
        ts_idx_mapping[param] = idx

    with open('../processed_data/param_idx_mapping.json', 'w') as f:
        json.dump(ts_idx_mapping, f)
    with open('../processed_data/param_color_mapping.json', 'w') as f:
        json.dump(ts_color_mapping, f)
    
    for split_idx in range(5):
        # start constructing the data list
        ImageDict_list = []
        demogr_lengths = []

        base_path = '../'
        split_path = '/splits/phy12_split' + str(split_idx+1) + '.npy'
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
        # extract train/val/test examples
        Ptrain = Pdict_list[idx_train]
        Pval = Pdict_list[idx_val]
        Ptest = Pdict_list[idx_test]

        # first round, find the mean and std for each param on training set
        train_ts_values = [[] for _ in range(num_ts_params)]
        all_ts_values = [[] for _ in range(num_ts_params)]
        stat_ts_values = np.ones(shape=(num_ts_params, 12)) # mean, std, y_min, y_max
        for idx, p in tqdm(enumerate(Ptrain)):
            ts_values = p['arr'] #(60, 34)
            for param_idx in range(num_ts_params): # ts_desc: (60, 34)
                ts_value = ts_values[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1,1)
                # handling missing value
                ts_value = ts_value[ts_value != 0]  
                train_ts_values[param_idx].extend(list(ts_value))

        for idx, p in tqdm(enumerate(Pdict_list)):
            ts_values = p['arr'] #(60, 34)
            for param_idx in range(num_ts_params): # ts_desc: (60, 34)
                ts_value = ts_values[:, param_idx]
                ts_value = np.array(ts_value).reshape(-1,1)
                # handling missing value
                ts_value = ts_value[ts_value != 0]  
                all_ts_values[param_idx].extend(list(ts_value))
        
        # sort the params based on missing ratios
        if order:
            ts_value_nums = [len(_) for _ in train_ts_values]
            ts_orders = np.argsort(ts_value_nums)[::-1]
        else:
            ts_orders = list(range(num_ts_params))

        # change from list to array
        for param_idx in range(num_ts_params):
            train_ts_values[param_idx] = np.array(train_ts_values[param_idx])

        for param_idx in range(num_ts_params): # ts_desc: (60, 34)
            param_ts_value = np.array(train_ts_values[param_idx])

            stat_ts_values[param_idx,0] = param_ts_value.mean()
            stat_ts_values[param_idx,1] = param_ts_value.std()
            stat_ts_values[param_idx,2] = param_ts_value.min()
            stat_ts_values[param_idx,3] = param_ts_value.max()

            """
            option 1. remove outliers with boxplot
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
            # print(f"{param_idx}, {outlier_ratio}")
            
            """
            option 2. remove outliers with standard deviation
            """
            mean = np.mean(param_ts_value)
            std = np.std(param_ts_value)
            upper_bound = mean + (4*std)
            lower_bound = mean - (4*std)
            stat_ts_values[param_idx,6] = lower_bound
            stat_ts_values[param_idx,7] = upper_bound
            param_ts_value2 = param_ts_value[(lower_bound<param_ts_value)&(upper_bound>param_ts_value)]
            outlier_ratio = 1 - (len(param_ts_value2) / len(param_ts_value))

            """
            option 3. remove outliers with modified z-score
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
            # print(f"{param_idx}, {outlier_ratio}")

            """
            option 4. quartile
            """
            sorted_param_ts_value = np.sort(param_ts_value)
            value_len = sorted_param_ts_value.shape[0]
            max_position = min(value_len-1, round(value_len*0.99995))
            min_position = max(0, round(value_len*0.00005))
            upper_bound = sorted_param_ts_value[max_position]
            lower_bound = sorted_param_ts_value[min_position]
            stat_ts_values[param_idx,10] = lower_bound
            stat_ts_values[param_idx,11] = upper_bound
            
        # second round, draw the image for each patient
        for idx, p in tqdm(enumerate(Pdict_list)):

            pid = int(p['id'])
            ts_values = p['arr'] #(215, 36)
            ts_times = p['time']

            # textual label
            arr_outcome = arr_outcomes[idx]
            label = int(arr_outcome[-1])
            label_name = "survivor" if label == 0 else "dead"

            # static feature
            static_demogr = p['static']
            demogr_desc = construct_demogr_description(static_demogr)

            # deal with outliers
            if not outlier:
                ts_scales = stat_ts_values[:,2:4] # no removal
            elif outlier == "iqr":
                ts_scales = stat_ts_values[:,4:6] # iqr
            elif outlier == "std":
                ts_scales = stat_ts_values[:,6:8] # std
            elif outlier == "mzs":
                ts_scales = stat_ts_values[:,8:10] # mzs
            elif outlier == "qt":
                ts_scales = stat_ts_values[:,10:12] # quartile

            # draw the image for each p
            drawed_params = draw_image(pid, split_idx, ts_orders, ts_values, ts_times, ts_params, ts_scales, override, differ, outlier, interpolation, order,
                                        image_size, grid_layout, 
                                        linestyle, linewidth, marker, markersize,
                                        ts_color_mapping, ts_idx_mapping)
            
            ImageDict = {
                "id": pid, 
                "text": demogr_desc,
                "label": label,
                "label_name": label_name,
                "arr_outcome": arr_outcome
                }
            ImageDict_list.append(ImageDict)
        
    print(len(ImageDict_list))
    np.save(f'../processed_data/ImageDict_list.npy', ImageDict_list)
    print(f"Save data in ImageDict_list.npy")


if __name__ == "__main__":
    construct_image(
        linestyle="-", linewidth=1, marker="*", markersize=2, 
        override=False, 
        differ=True, 
        outlier=None,
        interpolation=True,
        order=True,
        grid_layout=(6,6),
        image_size=None
        )

            




                



    