import pandas as pd


def convert_to_tfb_series(data):
    data = data.set_index("date")
    melted_df = data.melt(value_name="data", var_name="cols", ignore_index=False)
    return melted_df.reset_index()[['date', 'data', 'cols']]


data = pd.read_csv(r"D:\waterc\nj.csv")
melted_df = convert_to_tfb_series(data)
melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj.csv", encoding='utf-8-sig',index=False)








