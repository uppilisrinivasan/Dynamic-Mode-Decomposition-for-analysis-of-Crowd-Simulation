import pandas as pd
from datafold import TSCDataFrame



df = pd.read_csv("result_df.csv", index_col=[0, 1, 2], header=[0])


# tscdf = TSCDataFrame.from_csv("result_df.csv")

print(df)