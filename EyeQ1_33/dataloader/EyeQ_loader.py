# encoding: utf-8
# import sys
# # sys.path.append('/home/yyang2/data/yyang2/pycharm/EyeQ_all/EyeQ1_33/dataloader')
# sys.path.append('/home/zeiss/local-repository/P3-FIQA_trainingEfficientNet/FIQA_EfficientNet_TrainingCode_20201204/EyeQ1_33/dataloader')
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import torchvision.transforms as transforms
import fundus_prep as prep




def load_eyeQ_excel(list_file):
    image_names = []

    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)
    labels = torch.zeros(img_num, 6)

    for idx in range(img_num):
        image_name = df_tmp['path'][idx]
        image_names.append(image_name)
        label = torch.tensor([int(df_tmp['fundus_image_qualified'][idx]),
                              int(df_tmp['fi_unqualified_disc-position'][idx]),
                              int(df_tmp['fi_unqualified_macular-position'][idx]),
                              int(df_tmp['fi_unqualified_focus-clearness'][idx]),
                              int(df_tmp['fi_unqualified_readable-range'][idx]),
                              int(df_tmp['fi_unqualified_others'][idx])])

        labels[idx] = label

    return image_names, labels

# 创建dataset类，读取数据
class DatasetGenerator(Dataset):
    def __init__(self, list_file, transform1=None, transform2=None, n_class=2, set_name='train'):

        image_names, labels = load_eyeQ_excel(list_file)

        self.image_names = image_names
        self.labels = labels
        self.n_class = n_class
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train' or self.set_name == 'val':
            label = self.labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), label
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)



transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])


# 获取图（包含去除黑边及对图像进行处理）
def get_image(image_name, transform_list_val1=transform_list_val1, transformList2=transformList2):

    # 对眼底图去除黑边
    img = prep.imread(image_name)
    r_img, borders, mask = prep.process_without_gb(img)
    r_img = Image.fromarray(r_img)
    # 对图像进行处理,剪裁
    image = transform_list_val1(r_img)

    # 对图像进行色彩空间转换
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    img_hsv = image.convert("HSV")
    img_lab = ImageCms.applyTransform(image, rgb2lab_transform)
    # 对图像进行数据类型转变，并变成（BWHC）格式
    img_rgb = np.asarray(image).astype('float32')
    img_hsv = np.asarray(img_hsv).astype('float32')
    img_lab = np.asarray(img_lab).astype('float32')
    img_rgb = transformList2(img_rgb).unsqueeze(0)
    img_hsv = transformList2(img_hsv).unsqueeze(0)
    img_lab = transformList2(img_lab).unsqueeze(0)
    # 返回torch.tensor 格式数据
    return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)



# 创建dataset类，读取数据
class DatasetGenerator_class(Dataset):
    def __init__(self, list_file, transform1=None, transform2=None, set_name='train'):

        image_names, labels = load_eyeQ_excel(list_file)

        image_names, labels = self.getsample(image_names, labels)
        self.image_names = image_names
        self.labels = labels
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train' or self.set_name == 'val':
            label = self.labels[index]
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), label
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)

    def getsample(self, image_names, labels):
        image_names_re = []
        label_re = []
        indxs = []
        label = labels[:, 0]
        indxs_0 = label.nonzero().reshape(-1).tolist()
        random.shuffle(indxs_0)
        # indxs += indxs_0[:800]
        indxs += indxs_0[:200]
        for indx in ([2, 3, 4, 5]):
            label = labels[:, indx]
            indxs_1 = label.nonzero().reshape(-1).tolist()
            num = 200
            random.shuffle(indxs_1)
            indxs = indxs + indxs_1[:num]


        for i in indxs:
            image_names_re.append(image_names[i])

            label_re.append(labels[i])
        return image_names_re, label_re



