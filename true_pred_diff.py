import pandas as pd
import numpy as np

true_file = "../ecg_mini_dataset/data/attributes.csv"
pred_file = "./predicted_age.csv"

true_df = pd.read_csv(true_file)
n_valid = 100
true_age_all = true_df["age"]
true_age = true_age_all[:n_valid]
print(true_age.dtypes)
print(type(true_age))

pred_df = pd.read_csv(pred_file)
pred_age = pred_df["predicted_age"]

diff = pred_age - true_age

print(diff)

new_df = pd.DataFrame({'diff_age': diff})
new_df.to_csv('diff.csv')



