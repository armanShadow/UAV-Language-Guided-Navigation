import json
import pandas as pd
from datasets import Dataset

def load_data(path):
    # Load the JSON file as an array
    with open(path, "r") as file:
        data_array = json.load(file)

    # Keys to remove
    keys_to_remove = ["destination", "attention_list", "route_index"]

    # Remove specified keys from each object in the array
    for item in data_array:
        for key in keys_to_remove:
            item.pop(key, None)

    data = []
    i = 0
    while i < len(data_array):
        first_instruction = data_array[i]["instructions"]
        first_instruction = first_instruction.replace("[INS] ", "")
        number_of_rounds = int(data_array[i]["last_round_idx"])
        for j in range(1, number_of_rounds):
            instruction_start_index = data_array[i+j]["instructions"].index("[INS]")
            question = data_array[i+j]["instructions"][5:instruction_start_index]
            answer = data_array[i+j]["instructions"][instruction_start_index+5:]

            dialogue = {
                "question" : question,
                "answer" : answer,
                'first_instruction' : first_instruction,
                'history' : data_array[i+j]["pre_dialogs"],
                'view_areas': data_array[i+j]["gt_path_corners"],
                'map_name': data_array[i+j]["map_name"],
                'gps_botm_left': data_array[i+j]["gps_botm_left"],
                'gps_top_right': data_array[i+j]["gps_top_right"],
                'lng_ratio': data_array[i+j]["lng_ratio"],
                'lat_ratio': data_array[i+j]["lat_ratio"]
            }
            data.append(dialogue)
        i += number_of_rounds

    return data

train_data = load_data('../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json')
train_df = pd.DataFrame(train_data, columns=['question', 'answer', 'first_instruction', 'history', 'view_areas', 'map_name', 'gps_botm_left', 'gps_top_right', 'lng_ratio', 'lat_ratio'])

val_seen_data = load_data('../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json')
val_seen_df = pd.DataFrame(val_seen_data, columns=['question', 'answer', 'first_instruction', 'history', 'view_areas', 'map_name', 'gps_botm_left', 'gps_top_right', 'lng_ratio', 'lat_ratio'])

val_unseen_data = load_data('../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json')
val_unseen_df = pd.DataFrame(val_unseen_data, columns=['question', 'answer', 'first_instruction', 'history', 'view_areas', 'map_name', 'gps_botm_left', 'gps_top_right', 'lng_ratio', 'lat_ratio'])

dataFrame_all = pd.concat([train_df, val_seen_df, val_unseen_df], ignore_index=True)
dataFrame_all.to_csv('train_data', index=False)