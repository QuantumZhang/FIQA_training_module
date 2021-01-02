from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, recall_score, roc_auc_score
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch
from dataloader.EyeQ_loader import DatasetGenerator, get_image
import csv

model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=6)


# df_tmp = pd.read_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_Val.csv')
# df_tmp = pd.read_excel('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_质量标注.xls')
# df_tmp = pd.read_excel('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/Multi_Lable_质量标注_delete.xls')
# df_tmp = pd.read_csv('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/D_M_Multi_Lable_Val_delete_merge.csv')

# 加载模型
# model_name = 'cf_6class_acc_best_6_0.01_weight_1.tar'
# loaded_model = torch.load('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a_1_30/results/6class_sup_01/cf_6class_acc_best_6_0.01_weight_1.tar')

model_name = 'cf_6class_acc_best_6_0.08_6_65epo.tar'
loaded_model = torch.load('/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a_1_30/results/6class_sup_06/cf_6class_acc_best_6_0.08_6_65epo.tar')

model.load_state_dict(loaded_model['state_dict'], strict=False)

label = ['fundus_image_qualified', 'fundus_image_unqualified', 'fi_unqualified_disc-position + fi_unqualified_macular-position',
         'fi_unqualified_focus-clearness', 'fi_unqualified_readable-range', 'fi_unqualified_others']
label_pre = ['label0', 'label1', 'label2', 'label3', 'label4', 'label5']
# 模型处理及分析
re_root = '/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a_1_30/'
label_test_file = re_root + 'Multi_Lable_Val.csv'
df_tmp = pd.read_csv(label_test_file)

# 模型处理及分析
model = model.cuda()
matrix = torch.zeros(len(df_tmp), 6)
model.eval()
for i in range(len(df_tmp)):
    imagesA, imagesB, imagesC = get_image(df_tmp['path'][i])
    imagesA = imagesA.cuda()
    with torch.no_grad():
        combine = model(imagesA)
    combine = torch.sigmoid(combine)
    combine[combine > 0.5] = 1
    combine[combine <= 0.5] = 0
    if combine[0, 1:].sum() == 0:
        combine[0, 0] = 1
    if combine[0, 1:].sum() > 0:
        combine[0, 0] = 0
        combine[0, 1] = 1

    matrix[i] = combine


df_tmp['label0'] = matrix[:, 0]
df_tmp['label1'] = matrix[:, 1]
df_tmp['label2'] = matrix[:, 2]
df_tmp['label3'] = matrix[:, 3]
df_tmp['label4'] = matrix[:, 4]
df_tmp['label5'] = matrix[:, 5]


log_target_csv = '/home/yyang2/data/yyang2/Data/Export-Fundus-JPG/a_1_30/results/validate_6class.csv'
for i in range(len(label)):
    acc = accuracy_score(df_tmp[label[i]], df_tmp[label_pre[i]])
    kappa = cohen_kappa_score(df_tmp[label[i]], df_tmp[label_pre[i]])
    recall = recall_score(df_tmp[label[i]], df_tmp[label_pre[i]])
    recall0 = recall_score(df_tmp[label[i]], df_tmp[label_pre[i]], pos_label=0)
    auc = roc_auc_score(df_tmp[label[i]], df_tmp[label_pre[i]])
    matrix_ = confusion_matrix(df_tmp[label[i]], df_tmp[label_pre[i]])
    print(matrix_)
    print(label[i] + '== acc:', acc, '== kappa:', kappa, '== recall:', recall, '== recall0:', recall0, '== auc:', auc)

    with open(log_target_csv, 'a+') as f:
        if i == 0:
            csv_write = csv.writer(f)
            data_row = ['name', 'acc', 'kappa', 'recall', 'recall0', 'auc', model_name]
            csv_write.writerow(data_row)
        csv_write = csv.writer(f)
        data_row = [label[i], acc, kappa, recall, recall0, auc]
        csv_write.writerow(data_row)

