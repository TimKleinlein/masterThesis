import pickle as pkl
import os
import matplotlib.pyplot as plt
import numpy as np


disc_rounds = os.listdir('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/pvad_output_files')
disc_rounds = [x for x in disc_rounds if 'nontarget_label_dict' not in x]

# file length
# ratio 0, 1, 2
# ratio 50/60/70/80/90/95 % probability 2 label
# 6829 discussion rounds in total
file_lengths = []
total_0 = 0
total_1 = 0
total_2 = 0
ratio_0 = []
ratio_1 = []
ratio_2 = []
ratio_50_percent = []
ratio_60_percent = []
ratio_70_percent = []
ratio_80_percent = []
ratio_90_percent = []
ratio_95_percent = []
ratio_99_percent = []


for ind, dr in enumerate(disc_rounds):
    with open(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/Speech-Extraction/PVAD_Application/pvad_output_files/{dr}', 'rb') as file:
        dic = pkl.load(file)
        nr_0_pred = 0
        nr_1_pred = 1
        nr_2_pred = 2
        nr_2_pred_50_percent = 0
        nr_2_pred_60_percent = 0
        nr_2_pred_70_percent = 0
        nr_2_pred_80_percent = 0
        nr_2_pred_90_percent = 0
        nr_2_pred_95_percent = 0
        nr_2_pred_99_percent = 0

        for k in dic.keys():
            if dic[k].index(max(dic[k])) == 0:
                nr_0_pred += 1
            elif dic[k].index(max(dic[k])) == 1:
                nr_1_pred += 1
            else:
                nr_2_pred += 1
                if dic[k][2] > 0.5:
                    nr_2_pred_50_percent += 1
                    if dic[k][2] > 0.6:
                        nr_2_pred_60_percent += 1
                        if dic[k][2] > 0.7:
                            nr_2_pred_70_percent += 1
                            if dic[k][2] > 0.8:
                                nr_2_pred_80_percent += 1
                                if dic[k][2] > 0.9:
                                    nr_2_pred_90_percent += 1
                                    if dic[k][2] > 0.95:
                                        nr_2_pred_95_percent += 1
                                        if dic[k][2] > 0.99:
                                            nr_2_pred_99_percent += 1

        nr_2_pred_50_percentRATIO = nr_2_pred_50_percent / nr_2_pred
        nr_2_pred_60_percentRATIO = nr_2_pred_60_percent / nr_2_pred
        nr_2_pred_70_percentRATIO = nr_2_pred_70_percent / nr_2_pred
        nr_2_pred_80_percentRATIO = nr_2_pred_80_percent / nr_2_pred
        nr_2_pred_90_percentRATIO = nr_2_pred_90_percent / nr_2_pred
        nr_2_pred_95_percentRATIO = nr_2_pred_95_percent / nr_2_pred
        nr_2_pred_99_percentRATIO = nr_2_pred_99_percent / nr_2_pred

        file_lengths.append(len(dic.keys()))  # frames predicted (10ms)
        total_0 += nr_0_pred
        total_1 += nr_1_pred
        total_2 += nr_2_pred
        ratio_0.append(nr_0_pred / len(dic.keys()))
        ratio_1.append(nr_1_pred / len(dic.keys()))
        ratio_2.append(nr_2_pred / len(dic.keys()))
        ratio_50_percent.append(nr_2_pred_50_percentRATIO)
        ratio_60_percent.append(nr_2_pred_60_percentRATIO)
        ratio_70_percent.append(nr_2_pred_70_percentRATIO)
        ratio_80_percent.append(nr_2_pred_80_percentRATIO)
        ratio_90_percent.append(nr_2_pred_90_percentRATIO)
        ratio_95_percent.append(nr_2_pred_95_percentRATIO)
        ratio_99_percent.append(nr_2_pred_99_percentRATIO)

        print(ind)



# Set the style
plt.style.use('seaborn-deep')

# General settings for plots
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.fontsize': 10,
    'axes.grid': False,
    'grid.alpha': 0.5,
})

# Ratio how many of 2 label predictions will be kept for each threshold
for threshold in [50, 60, 70, 80, 90, 95, 99]:
    plt.hist(eval(f'ratio_{threshold}_percent'), bins=10, range=(0, 1), edgecolor='black', color='lightblue')

    # Add title and labels
    plt.xlabel('Ratio Kept')
    plt.ylabel('Frequency')

    # Save plot with high dpi
    plt.savefig(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/ratio_{threshold}_percent_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Total ratio of label two values lost for each threshold
x_values = [50, 60, 70, 80, 90, 95, 99]
y_values = [np.mean(ratio_50_percent), np.mean(ratio_60_percent), np.mean(ratio_70_percent), np.mean(ratio_80_percent), np.mean(ratio_90_percent), np.mean(ratio_95_percent), np.mean(ratio_99_percent)]

plt.bar(x_values, y_values, color='skyblue', edgecolor='black')

# Add title and labels
# plt.title('Mean Ratio of Label 2 Predictions Kept for Each Probability Threshold')
plt.xlabel('Probability Threshold (%)')
plt.ylabel('Mean Ratio of Label 2 Predictions Kept')

# Save plot with high dpi
plt.savefig('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/mean_ratio_label_2_predictions_kept.png', dpi=300, bbox_inches='tight')
plt.show()

# For each label how often it was predicted
x_values = ['No speech', 'Non-Target speaker speech', 'Target speaker speech']
y_values = [total_0, total_1, total_2]

plt.bar(x_values, y_values, color='skyblue', edgecolor='black')

# Add title and labels
# plt.title('Total Frequency of Predictions for Each Label')
plt.xlabel('Label')
plt.ylabel('Frequency of Prediction')

# Save plot with high dpi
plt.savefig('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/total_frequency_predictions_per_label.png', dpi=300, bbox_inches='tight')
plt.show()

# For each label distribution of ratio within a file
for label in [0, 1, 2]:
    plt.hist(eval(f'ratio_{label}'), bins=10, range=(0, 1), edgecolor='black', color='lightblue')

    # Add title and labels
    #plt.title(f'Distribution of Ratio of Label {label} Within File')
    plt.xlabel(f'Ratio of Label {label} Within a Discussion Round')
    plt.ylabel('Frequency')

    # Save plot with high dpi
    plt.savefig(f'/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/ratio_distribution_label_{label}.png', dpi=300, bbox_inches='tight')
    plt.show()

# File lengths
file_lengths = [x/100 for x in file_lengths]
plt.hist(file_lengths, bins=10, range=(min(file_lengths), max(file_lengths)), edgecolor='black', color='lightblue')

# Add title and labels
# plt.title('Distribution of File Lengths')
plt.xlabel('Discussion Round Length in Seconds')
plt.ylabel('Frequency')

# Save plot with high dpi
plt.savefig('/Users/timkleinlein/Documents/Uni Konstanz/Masterarbeit/gitlab/among-us-analysis/final_plots/file_lengths_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
