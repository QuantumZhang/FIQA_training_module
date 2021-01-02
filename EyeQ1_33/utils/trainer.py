import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, recall_score
from tqdm import tqdm
import torch.nn as nn
import csv
Label_Merge = ['fundus_image_qualified', 'fi_unqualified_disc-position',
               'fi_unqualified_macular-position',
               'fi_unqualified_focus-clearness', 'fi_unqualified_readable-range',
         'fi_unqualified_others']




def show_message(y_tru, y_p, model='train'):
    message = []
    y_p[y_p > 0.5] = 1
    y_p[y_p <= 0.5] = 0
    acc = 0
    kappa = 0
    kappa_weight = [1.0, 1.0, 0.8, 1.2, 0.8, 1.2]
    for i in range(y_p.shape[1]):
        kappa_t = cohen_kappa_score(y_tru[:, i], y_p[:, i])
        kappa += kappa_weight[i] * kappa_t
        acc_t = accuracy_score(y_tru[:, i], y_p[:, i])
        acc += acc_t

        confu_t = confusion_matrix(y_tru[:, i], y_p[:, i])
        recall_t = recall_score(y_tru[:, i], y_p[:, i])
        recall0_t = recall_score(y_tru[:, i], y_p[:, i], pos_label=0)
        message.append(kappa_t)
        message.append(acc_t)
        message.append(recall_t)
        message.append(recall0_t)
        print('label==>%s<==>%s<==的信息：'%(Label_Merge[i], model))
        print('kappa：', kappa_t, '-----acc:', acc_t, '-----recall:', recall_t, '---recall0', recall0_t)
        print('confusion_matrix:\n', confu_t)
    return acc / y_p.shape[1], kappa / y_p.shape[1], message

# 定义训练类
def train_step(train_loader, model, epoch, optimizer, criterion, epochs, log_csv, cycle_scheduler= None):

    # switch to train mode
    model.train()
    epoch_loss = 0.0


    iters_per_epoch = len(train_loader)

    y_tru = None
    y_p = None

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        if y_tru is None:
            y_tru = np.array(labels)
        else:
            y_tru = np.vstack((y_tru, np.array(labels)))

        labels = labels.float().cuda()
        # labels = labels.cuda().long()
        # labels = torch.tensor(labels).reshape(4,-1)
        # labels = labels.reshape(labels.shape[0],1)
        # labels = torch.zeros(labels.shape[0], 2).scatter_(1, labels, 1).cuda()

        combine = model(imagesA)
        combine = torch.sigmoid(combine)
        # out_A, out_B, out_C, out_F, combine = model(imagesA, imagesB, imagesC)

        # loss_x = criterion(out_A, labels)
        # loss_y = criterion(out_B, labels)
        # loss_z = criterion(out_C, labels)
        # loss_c = criterion(out_F, labels)
        loss_f = criterion(combine, labels)
        lossValue = loss_f
        # lossValue = loss_w[0]*loss_x+loss_w[1]*loss_y+loss_w[2]*loss_z+loss_w[3]*loss_c+loss_w[4]*loss_f
        # writer.add_scalar('/epoch_loss', lossValue, step)

        # pre = torch.cat((pre, combine), 0)
        # tru = torch.cat((tru, labels.float()), 0)
        y_pre = combine.detach().cpu().numpy()
        if y_p is None:
            y_p = np.array(y_pre)
        else:
            y_p = np.vstack((y_p, np.array(y_pre)))

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        if cycle_scheduler is not None:
            cycle_scheduler.batch_step()
        epoch_loss += lossValue.item()

    acc, kappa, message = show_message(y_tru, y_p, model='train')
    with open(log_csv, 'a+') as f:
        if epoch == 0:
            csv_write = csv.writer(f)
            data_row = ['epoch',
                        'qua_kappa', 'qua_acc', 'qua_recall', 'qua_recall0',
                        'disc_kappa', 'disc_acc', 'disc_recall', 'disc_recall0',
                        'macular_kappa', 'macular_acc', 'macular_recall', 'macular_recall0',
                        'clear_kappa', 'clear_acc', 'clear_recall', 'clear_recall0',
                        'read_kappa', 'read_acc', 'read_recall', 'read_recall0',
                        'others_kappa', 'others_acc', 'others_recall', 'others_recall0',
                        ]
            csv_write.writerow(data_row)
        csv_write = csv.writer(f)
        data_row = [epoch] + message
        csv_write.writerow(data_row)
    epoch_loss = epoch_loss / iters_per_epoch
    return epoch_loss, acc, kappa


def validation_step(train_loader, model, epoch, optimizer, criterion, epochs, log_csv, cycle_scheduler=None):

    # switch to train mode
    model.eval()
    epoch_loss = 0.0


    iters_per_epoch = len(train_loader)

    y_tru = None
    y_p = None

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        if y_tru is None:
            y_tru = np.array(labels)
        else:
            y_tru = np.vstack((y_tru, np.array(labels)))

        labels = labels.float().cuda()
        # labels = labels.cuda().long()
        # labels = torch.tensor(labels).reshape(4,-1)
        # labels = labels.reshape(labels.shape[0],1)
        # labels = torch.zeros(labels.shape[0], 2).scatter_(1, labels, 1).cuda()
        with torch.no_grad():
            combine = model(imagesA)
        combine = torch.sigmoid(combine)
        # out_A, out_B, out_C, out_F, combine = model(imagesA, imagesB, imagesC)

        # loss_x = criterion(out_A, labels)
        # loss_y = criterion(out_B, labels)
        # loss_z = criterion(out_C, labels)
        # loss_c = criterion(out_F, labels)
        loss_f = criterion(combine, labels)
        lossValue = loss_f
        # lossValue = loss_w[0]*loss_x+loss_w[1]*loss_y+loss_w[2]*loss_z+loss_w[3]*loss_c+loss_w[4]*loss_f
        # writer.add_scalar('/epoch_loss', lossValue, step)

        # pre = torch.cat((pre, combine), 0)
        # tru = torch.cat((tru, labels.float()), 0)
        y_pre = combine.detach().cpu().numpy()
        if y_p is None:
            y_p = np.array(y_pre)
        else:
            y_p = np.vstack((y_p, np.array(y_pre)))

        # optimizer.zero_grad()
        # lossValue.backward()
        # optimizer.step()
        # cycle_scheduler.batch_step()
        epoch_loss += lossValue.item()

    acc, kappa, message = show_message(y_tru, y_p, model='val')
    with open(log_csv, 'a+') as f:
        if epoch == 0:
            csv_write = csv.writer(f)
            data_row = ['epoch',
                        'qua_kappa', 'qua_acc', 'qua_recall', 'qua_recall0',
                        'disc_kappa', 'disc_acc', 'disc_recall', 'disc_recall0',
                        'macular_kappa', 'macular_acc', 'macular_recall', 'macular_recall0',
                        'clear_kappa', 'clear_acc', 'clear_recall', 'clear_recall0',
                        'read_kappa', 'read_acc', 'read_recall', 'read_recall0',
                        'others_kappa', 'others_acc', 'others_recall', 'others_recall0',
                        ]
            csv_write.writerow(data_row)
        csv_write = csv.writer(f)
        data_row = [epoch] + message
        csv_write.writerow(data_row)
    epoch_loss = epoch_loss / iters_per_epoch
    return epoch_loss, acc, kappa

#
# def validation_step(val_loader, model, criterion):
#
#     # switch to train mode
#     model.eval()
#     epoch_loss = 0
#     iters_per_epoch = len(val_loader)
#     y_tru = None
#     y_p = None
#     for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
#         imagesA = imagesA.cuda()
#         imagesB = imagesB.cuda()
#         imagesC = imagesC.cuda()
#
#         if y_tru is None:
#             y_tru = np.array(labels)
#         else:
#             y_tru = np.vstack((y_tru, np.array(labels)))
#
#         labels = labels.float().cuda()
#         # _, _, _, _, outputs = model(imagesA, imagesB, imagesC)
#         combine = model(imagesA)
#         combine = torch.sigmoid(combine)
#         with torch.no_grad():
#             loss = criterion(combine, labels)
#             epoch_loss += loss.item()
#
#
#         y_pre = combine.detach().cpu().numpy()
#         if y_p is None:
#             y_p = np.array(y_pre)
#         else:
#             y_p = np.vstack((y_p, np.array(y_pre)))
#
#         y_tru = y_tru.reshape((-1))
#         y_p = y_p.reshape((-1))
#     acc = show_message(y_tru, y_p)
#     epoch_loss = epoch_loss / iters_per_epoch
#     return epoch_loss, acc

def save_output(label_test_file, dataPRED, label_idx, save_file):
    label_list = label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["image"].tolist()

    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['image_name'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['image_name']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)

def acc_mol(val_loader, model):
    model.eval()
    iters_per_epoch = len(val_loader)
    pre_all = []
    y_all = []
    for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):

        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()
        labels = labels.cuda()

        _, _, _, _, outputs = model(imagesA, imagesB, imagesC)
        pre = outputs.argmax(dim=1)
        pre_all.append(pre)
        y_all.append(labels)
    return pre_all, y_all

def save_file(model,loss, kappa, acc, recall,recall0,model_save_file, epoch):
    torch.save({'state_dict': model.state_dict(), 'loss':loss,'kappa':kappa,
                'acc': acc, 'recall':recall,'recall0':recall0,'epoch': epoch + 1}, model_save_file)
    print('已保存模型至：',model_save_file)


