import json

param_detailed_description = {
    "HR": "Heart rate (bpm)",
    "O2Sat": "Pulse oximetry (%)",
    "Temp": "Temperature (°C)",
    "SBP": "Systolic BP (mm Hg)",
    "MAP": "Mean arterial pressure (mm Hg)",
    "DBP": "Diastolic BP (mm Hg)",
    "Resp": "Respiration rate (breaths per minute)",
    "EtCO2": "End tidal carbon dioxide (mm Hg)",
    "BaseExcess": "Measure of excess bicarbonate (mmol/L)",
    "HCO3": "Bicarbonate (mmol/L)",
    "FiO2": "Fraction of inspired oxygen (%)",
    "pH": "Arterial pH",
    "PaCO2": "Partial pressure of carbon dioxide from arterial blood (mm Hg)",
    "SaO2": "Oxygen saturation from arterial blood (%)",
    "AST": "Aspartate transaminase (IU/L)",
    "BUN": "Blood urea nitrogen (mg/dL)",
    "Alkalinephos": "Alkaline phosphatase (IU/L)",
    "Calcium": "Calcium (mg/dL)",
    "Chloride": "Chloride (mmol/L)",
    "Creatinine": "Creatinine (mg/dL)",
    "Bilirubin_direct": "Bilirubin direct (mg/dL)",
    "Glucose" :"Serum glucose (mg/dL)",
    "Lactate": "Lactate (mmol/L)",
    "Magnesium": "Magnesium (mmol/dL)",
    "Phosphate": "Phosphate (mg/dL)",
    "Potassium": "Potassium (mmol/L)",
    "Bilirubin_total": "Total bilirubin (mg/dL)",
    "Troponinl": "Troponin-I (ng/mL)",
    "Hct": "Hematocrit (%)",
    "Hgb": "Hemoglobin (g/dL)",
    "PTT": "partial thromboplastin time (seconds)",
    "WBC": "Leukocyte count (cells/nL)",
    "Fibrinogen": "Fibrinogen (mg/dL)",
    "Platelets": "Platelets"
}

with open('../processed_data/ts_params_desc.json', 'w') as f:
    json.dump(param_detailed_description, f)


marker_detailed_description = {
    ">": "triangle_right",
    "1": "tri_down",
    "2": "tri_up",
    "3": "tri_left",
    "4": "tri_right",
    "8": "octagon",
    "s": "square",
    "p": "pentagon",
    "P": "plus",
    "*": "star",
    "h": "hexagon1",
    "H": "hexagon2",
    "+": "plus",
    "x": "x",
    "X": "x",
    "D": "diamond",
    "d": "thin_diamond",
    "|": "vline",
    "_": "hline"
}

with open('../processed_data/plt_markers_desc.json', 'w') as f:
    json.dump(marker_detailed_description, f)

color_detailed_description = {
    "green": "1", 
    "black": "2",
    "blue": "3",
    "brown": "4",
    "chartreuse": "5",
    "chocolate": "6",
    "coral": "7",
    "crimson": "8",
    "blueviolet": "9",
    "darkblue": "10",
    "darkgreen": "11",
    "firebrick": "12",
    "gold": "13",
    "teal": "14",
    "grey": "15",
    "indigo": "16",
    "steelblue": "17",
    "indianred": "18",
    "goldenrod": "19",
    "darkred": "20",
    "darkorange": "21",
    "magenta": "22",
    "maroon": "23",
    "navy": "24",
    "olive": "25",
    "orange": "26",
    "orchid": "27",
    "pink": "28",
    "plum": "29",
    "purple": "30",
    "red": "31",
    "cornflowerblue": "32",
    "sienna": "33",
    "darkkhaki": "34",
    "tan": "35",
    "dodgerblue": "36",
    "darkseagreen": "37",
    "cadetblue": "38"
}

with open('../processed_data/plt_colors_desc.json', 'w') as f:
    json.dump(color_detailed_description, f)