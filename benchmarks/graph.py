#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = 'parsed_log_output.csv'
HEADER_ROW = 0

X_AXIS_COLUMN = 'Total Time (ms)'
Y_AXIS_COLUMN = 'RECALL@100'

PARAMETERS_TO_PLOT = [
    'NPROBE',
    'COARSE PROBE',
    'SUBQUANTIZERS'
]

try:
    df = pd.read_csv(FILE_PATH, header=HEADER_ROW)
    print(f"Successfully loaded '{FILE_PATH}'.")
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found. Please check the file name and path.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    exit()

df = df.dropna(how='all')

columns_to_convert = [X_AXIS_COLUMN, Y_AXIS_COLUMN] + PARAMETERS_TO_PLOT

for col in columns_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Column '{col}' not found in the file. It will be skipped.")

df = df.dropna(subset=columns_to_convert)
print(f"Data cleaned. Found {len(df)} valid data points.")

sns.set_style("whitegrid")
palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

df = df.sort_values(by=[X_AXIS_COLUMN, Y_AXIS_COLUMN])

for param in PARAMETERS_TO_PLOT:
    if param not in df.columns:
        continue

    plt.figure(figsize=(11, 7))
    
    sns.lineplot(
        data=df,
        x=X_AXIS_COLUMN,
        y=Y_AXIS_COLUMN,
        hue=param,
        style=param,
        palette=palette,
        marker='o',
        markersize=8,
        linewidth=2.5,
        estimator=None
    )

    plt.title(f'{Y_AXIS_COLUMN} vs. TIME(ms)', fontsize=16, weight='bold')
    plt.xlabel(f'{X_AXIS_COLUMN} →', fontsize=12)
    plt.ylabel(f'{Y_AXIS_COLUMN} →', fontsize=12)
    plt.legend(title=param, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f'{Y_AXIS_COLUMN}_vs_{X_AXIS_COLUMN}_by_{param}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Generated plot: {filename}")

print("\nAll plots have been generated successfully.")
