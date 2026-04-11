import pandas as pd


def convert_to_tfb_series(data):
    data = data.set_index("date")
    melted_df = data.melt(value_name="data", var_name="cols", ignore_index=False)
    return melted_df.reset_index()[['date', 'data', 'cols']]

# data = pd.read_csv(r"D:/tfbtest.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:/tfbtest_long_all.csv", index=False)

# data = pd.read_csv(r"D:/waste-date123.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:/waste-date123.csv", index=False)


##包含缺失的原始nj水厂数据32786x37
# data = pd.read_csv(r"D:\10w\nj.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\10w\feature\tfb_nj.csv", index=False)

# ##南江水厂数据32856x37
# data = pd.read_csv(r"D:\lhw\10w\nj_32856.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\10w\tfb_nj_32856.csv", index=False)

# #南江水厂数据32856x13
# data = pd.read_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\nj\nj_32856x13.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x13.csv", encoding='utf-8-sig',index=False)

# #南江水厂数据32856x13
# data = pd.read_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\jyl_hour_with_dose.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_jyl_32856x39.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x10 延迟对齐
# data = pd.read_csv(r"D:\waterc\features_out_turbidity_best.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_jyl_delay_32856x10.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x39
# data = pd.read_csv(r"D:\waterc\21888x39.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x39.csv", encoding='utf-8-sig',index=False)


# # #南江水厂数据21888x39
# data = pd.read_csv(r"D:\waterc\10968x37.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_10968x37.csv", encoding='utf-8-sig',index=False)
#
# # #南江水厂数据32856x39
# data = pd.read_csv(r"D:\waterc\32856x39_jymode.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x39_jymode.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x13 没有液位、阀门数据
# data = pd.read_csv(r"D:\waterc\32856x13.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x13.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x18 液位、阀门数据平均+加药量
# data = pd.read_csv(r"D:\waterc\32856x18.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x18.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x6 预测滤前浊度
# data = pd.read_csv(r"D:\waterc\32856x6.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x6.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x6 预测滤前浊度
# data = pd.read_csv(r"D:\waterc\32856x5\18793x10.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_18793x10.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据32856x6 预测滤前浊度
# data = pd.read_csv(r"D:\waterc\32856x5_new\32856x10.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x10.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x25
# data = pd.read_csv(r"D:\waterc\21888x25.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x25.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x14 除去所有水厂运行数据
# data = pd.read_csv(r"D:\waterc\21888x14.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x14.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x7 只包含目标变量
# data = pd.read_csv(r"D:\waterc\21888x7.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x7.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x142
# data = pd.read_csv(r"D:\waterc\21865x142.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21865x142.csv", encoding='utf-8-sig',index=False)


# # #南江水厂数据21888x55
# data = pd.read_csv(r"D:\waterc\21865x55.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21865x55.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x55
# data = pd.read_csv(r"D:\waterc\32856x3.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_32856x3.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x55
# data = pd.read_csv(r"D:\waterc\norm\21888x39_normalized.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x39_normalized.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x12
# data = pd.read_csv(r"D:\waterc\21888x12.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x12.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x50
# data = pd.read_csv(r"D:\waterc\1-29\21888x50.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x50.csv", encoding='utf-8-sig',index=False)

# # #南江水厂数据21888x50
# data = pd.read_csv(r"D:\waterc\1-31\21888x43.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x43.csv", encoding='utf-8-sig',index=False)


# # #龙床井水厂 数据32856x53
# data = pd.read_csv(r"D:\waterc\lcj\cleaned_water_data.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_lcj_32856x53.csv", encoding='utf-8-sig',index=False)

# # #南江水厂三班加药数据
# data = pd.read_csv(r"D:\waterc\3-1\jyl.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_jyl3.csv", encoding='utf-8-sig',index=False)
#
# # #南江水厂三班加药数据
# data = pd.read_csv(r"D:\waterc\21888x40.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x40.csv", encoding='utf-8-sig',index=False)


# # #南江水厂21888x14 删除所有液位开水阀信息，只保留不同粒度加药
# data = pd.read_csv(r"D:\waterc\nj_21888x14.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x14_new.csv", encoding='utf-8-sig',index=False)

# # #南江水厂21888x14 删除所有液位开水阀信息，只保留不同粒度加药
# data = pd.read_csv(r"D:\waterc\21888x40_aligned.csv")
# melted_df = convert_to_tfb_series(data)
# melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x40_aligned.csv", encoding='utf-8-sig',index=False)


# #南江水厂21888x9 只保留不同粒度加药、原水信息、2个滤前浊度
data = pd.read_csv(r"D:\waterc\21888x9.csv")
melted_df = convert_to_tfb_series(data)
melted_df.to_csv(r"D:\waterc\TFB-master-nj\dataset\forecasting\tfb_nj_21888x9.csv", encoding='utf-8-sig',index=False)








