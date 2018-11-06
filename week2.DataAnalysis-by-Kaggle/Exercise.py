import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

train = pd.read_csv("./data/train.csv")

def main():
    train['Cabin'] = train['Cabin'].str.extract('([a-zA-Z ]+)', expand=False)
    Pclass1 = train[train["Pclass"] == 1]['Cabin'].value_counts()
    Pclass2 = train[train["Pclass"] == 2]['Cabin'].value_counts()
    Pclass3 = train[train["Pclass"] == 3]['Cabin'].value_counts()

    return Pclass1, Pclass2, Pclass3

if __name__ == "__main__":
    Pclass_list = list(main())
    df = pd.DataFrame(Pclass_list)
    df.index = ['1st class', '2nd class', '3rd class']
    plt.bar( figsize=(10, 5))
    plt.show()