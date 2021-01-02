import pandas as pd
import numpy as np
import os
df = pd.read_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/质量标注1_33.csv')

keys = ['filename', 'folder', 'octTags0', 'octTags1', 'octTags2', 'octTags3', 'octTags4']
label = ['fundus_image_qualified', 'fundus_image_unqualified', 'fi_unqualified_disc-position',
         'fi_unqualified_macular-position', 'fi_unqualified_focus-clearness', 'fi_unqualified_readable-range',
         'fi_unqualified_others']


# df['all_label'] = df['octTags0'].map(str) + '_' + df['octTags1'].map(str) + '_' + df['octTags2'].map(str) + '_'
# + df['octTags3'].map(str) + '_' + df['octTags4'].map(str)

df['all_label'] = df['folder']

for i in range(len(df)):
    df['all_label'][i] = str(df['octTags0'][i]) + str(df['octTags1'][i]) + str(df['octTags2'][i])\
                             + str(df['octTags3'][i]) + str(df['octTags4'][i])


label_np = np.zeros((len(df), len(label)))

for i in range(len(df)):
    for j in range(len(label)):
        if label[j] in df['all_label'][i]:
            label_np[i, j] = 1

'''
# 跟张博沟通，决定使用标签正常+不正常的子标签方式，之前融合视盘位置和黄斑位置的方式不采用
Label_Merge = ['fundus_image_qualified', 'fundus_image_unqualified',
               'fi_unqualified_disc-position + fi_unqualified_macular-position',
               'fi_unqualified_focus-clearness', 'fi_unqualified_readable-range',
         'fi_unqualified_others']


label_np[:, 2] += label_np[:, 3]
label_np[:, 2][label_np[:, 2] > 1] = 1

for i in range(len(Label_Merge)):
    if i < 3:
        df[Label_Merge[i]] = label_np[:, i]
    elif i >= 3:
        df[Label_Merge[i]] = label_np[:, i+1]

'''

# 将文件制作成one-hot模式
for i in range(len(label)):
    df[label[i]] = label_np[:, i]

# 保存文件
df.to_excel('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/Multi_Lable_质量标注.xls')


# 读取文件
df2 = pd.read_excel('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/Multi_Lable_质量标注.xls')

# 在文件中增加path路径
base_image_dir = '/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/crop_new_name'
df2['path'] = df2['filename'].map(lambda x: os.path.join(base_image_dir, '{}'.format(x.replace('jpg', 'png'))))
df2['exists'] = df2['path'].map(os.path.exists)
df2 = df2.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df2 = df2.drop(columns=['Unnamed: 0'])
df2.to_excel('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/Multi_Lable_质量标注.xls')

# 对文件进行分割，按照0.2的比例划分测试集和验证集
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df2, test_size=0.2)
# 查看清晰度标签在测试集和验证集的效果
train_df.pivot_table(index='fi_unqualified_focus-clearness', aggfunc=len)
val_df.pivot_table(index='fi_unqualified_focus-clearness', aggfunc=len)
# 舍弃多余的标签
train_df = train_df.drop(columns=['Unnamed: 0.1'])
val_df = val_df.drop(columns=['Unnamed: 0.1'])

#保存相关的文件
train_df.to_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/Multi_Lable_Train.csv')
val_df.to_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a1_33/Multi_Lable_Val.csv')

# import torch
# nums = torch.tensor([855., 1219., 471., 235., 548., 686.])
# weights = 1. / (nums / nums.min())
#
#
#
# # 创建clearness的单独文件，训练一个网络查看是否可区分
# train_df = pd.read_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_Train.csv')
# val_df = pd.read_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_Val.csv')
#
# df = train_df.append(val_df)
#
# def balance_label(file, label):
#     class_size = file.pivot_table(index=label, aggfunc=len).max().max()
#     clearness_file = file.groupby([label]).apply(lambda x: x.sample(int(class_size), replace=True)).reset_index(drop=True)
#     clearness_file = clearness_file.sample(frac=1).reset_index(drop=True)
#     return clearness_file
#
#
# file_val = balance_label(val_df, 'fi_unqualified_focus-clearness')
# file_train = balance_label(train_df, 'fi_unqualified_focus-clearness')
#
# file_val.to_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_Val_Clearness.csv')
# file_train.to_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_Train_Clearness.csv')
#
#
#






