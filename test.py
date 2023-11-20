import  torch
import  numpy as np


import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("./result.csv")

    plt.figure(figsize=(5,10))
    for t in data:
        plt.scatter(float(t['last_val']), float(t['last_adv_val']), label=t['model'])
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()