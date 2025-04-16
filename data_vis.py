# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:34:44 2025

@author: kenji
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def getData(filename, columns):
    df = pd.read_csv(filename)
    return df[columns]

def process(filename, dv):
    data = getData(filename, ["filename", dv])

    # Files were named by the individual who wrote the data, followed by
    # a 4 digit time. For instance: "k_21.23". This block seperates the 
    # initial and time spent writing, and sorts by time
    
    data["group"] = data["filename"].apply(lambda x: x.split('_')[0])
    data["time"] = data["filename"].apply(lambda x: float(os.path.splitext(x.split('_')[1])[0]))
    data = data.sort_values(by=["group", "time"])

    line_graph(data, "time", dv)

def line_graph(data, x_col, y_col):
    colors = {'k': 'blue', 'f': 'green', 'j': 'orange'}
    plt.figure(figsize=(10, 6))

    # Plot lines by group
    for group, group_data in data.groupby('group'):
        plt.plot(
            group_data[x_col],
            group_data[y_col],
            marker='o',
            label=f'{group.upper()} Data',
            color=colors.get(group, 'gray')
        )
        
    min_x = int(data[x_col].min())
    max_x = int(data[x_col].max())
    ticks = list(range(min_x, max_x + 2, 2))
    plt.xticks(ticks)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col} by Group')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    filename = "total_results.csv"
    dv = "readability" # Replace with dependent variable
    process(filename, dv)

if __name__ == "__main__":
    main()
