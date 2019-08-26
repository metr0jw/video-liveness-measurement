import pandas as pd


data_load = pd.read_csv("y_prediction_itzy.csv").values
pred_data = list(data_load)
cut_start_data = list()
cut_end_data = list()

idx = 0
data_length = pred_data.__len__() - 1

while idx != data_length:
    count = 0
    most_dyn_count = 0
    if pred_data[idx] == 0.5 or pred_data[idx] == 1:
        start_count = idx
        while pred_data[idx] == 0.5 or pred_data[idx] == 1:
            idx += 1
            count += 1
            if pred_data[idx] == 1:
                most_dyn_count += 1
            print("TEST")
        last_count = idx
    if count > 3 and most_dyn_count > 0:
        print("SUCCESS")
        cut_start_data.append(start_count)
        cut_end_data.append(last_count)
    idx += 1
cut_start_data = pd.DataFrame(cut_start_data, columns=["Start Time"])
cut_end_data = pd.DataFrame(cut_end_data, columns=["End Time"])
time_data = pd.concat([cut_start_data, cut_end_data], axis=1)
time_data.to_csv("timedata_itzy.csv", mode='w', index=False)