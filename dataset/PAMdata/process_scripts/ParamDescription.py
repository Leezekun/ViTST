import json

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