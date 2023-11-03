import  torch
import  numpy as np


import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("./results.csv")

    plt.figure(figsize=(5,10))
    for t in data:
        plt.scatter(float(t['SA']), float(t['RA']), label=t['model'])

    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()