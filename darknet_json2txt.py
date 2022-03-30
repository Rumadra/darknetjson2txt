import os
import sys
from pathlib import Path

import pandas as pd
import json

output_folder = "yolooutput_result"
os.makedirs(output_folder, exist_ok=True)

ext = '.txt'

# format into yolo output
def yolo_formatter(cls_id, cx, cy, w, h, confidence):
    text = "{} {} {} {} {} {}\n"
    return text.format(cls_id, cx, cy, w, h, confidence)

# write yolo output to txt
def write_txt(filename, text):
    with open(output_folder + "/" + filename + ext, 'a', newline="\n") as f:
        f.write(text)

# Converts darknet test result json to txt 
filepath = sys.argv[1]
print(filepath)

df_json = pd.read_json(filepath)
# print(df_json)
print(df_json.filename)

row_size = len(df_json)
print(row_size)

# get & set params from each frame
for num in range(row_size):
    # get filename
    f_path = df_json.filename[num]
    f = Path(f_path).stem
    print("filename: {}".format(f))

    obj_list = df_json.objects[num]
    print("obj_list size: {}".format(len(obj_list)))
     # if object is empty, skip or make empty txt
    if not obj_list:
        write_txt(f, "\n")
        continue
    
    # print(obj_list)
    for i in range(len(obj_list)):
        # print(obj_list[i])
        obj = obj_list[i]

        # get class_id
        class_id = obj["class_id"]
        print("class_id: {}".format(class_id))

        # get relative_coordinates
        relative_coord = obj["relative_coordinates"]
        # get center_x
        cx = relative_coord["center_x"]
        # get center_y
        cy = relative_coord["center_y"]
        # get width
        w = relative_coord["width"]
        # get height
        h = relative_coord["height"]

        # get confidence
        conf = obj["confidence"]
        
        # write result to txt
        res = yolo_formatter(class_id, cx, cy, w, h, conf)
        write_txt(f, res)

print("Done.")