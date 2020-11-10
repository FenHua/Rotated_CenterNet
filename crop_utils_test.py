import os
import sys
import gdal
import numpy as np


relate_loc = []  # 切片与大图位置关系


# 根据relate写需要测试切片的名称
def write_test_txt(relate_locs, txt_path):
    with open(txt_path, 'a+') as f:
        for relate_loc in relate_locs:
            f.write(relate_loc[0] + '\n')   # 写名称


# 切片相对大图位置信息的写
def write_relate_loc2txt(relate_locs, txt_path):
    with open(txt_path, 'a+') as f:
        for relate_loc in relate_locs:
            relate_loc = [str(i) for i in relate_loc]
            sep = ','
            f.write(sep.join(relate_loc) + '\n')  # 以sep分割进行存储


#  读取tif数据
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


# 拉伸图像  #图片的16位转8位
def uint16to8(im_data, im_geotrans, im_proj, path, lower_percent=2, higher_percent=98):
    rows, cols = im_data.shape
    out = np.zeros((3, rows, cols)).astype(np.uint8)
    a = 0
    b = 255
    print(np.min(im_data), np.max(im_data))
    # 如果切片全黑【不保存】
    if np.max(im_data) == 0:
        return None
    c = np.percentile(im_data, lower_percent)
    d = np.percentile(im_data, higher_percent)
    t = a + (im_data - c) * (b - a) / (d - c)
    # 使用2% 和 98% 防止异常影响最后的转换
    t[t < a] = a
    t[t > b] = b
    for i in range(3):
        out[i] = t.astype(np.uint8)
    return out


# 保存成tiff
def writeTiff(im_data, im_geotrans, im_proj, path):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
        new_im_data = im_data
    elif len(im_data.shape) == 2:
        # 比赛单波段（16转8）
        new_im_data = uint16to8(im_data, im_geotrans, im_proj, path)
        if new_im_data is None:
            return 0
        im_bands, im_height, im_width = new_im_data.shape
    if 'int8' in new_im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in new_im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)        # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(new_im_data[i])
    del dataset
    return 1


# 滑动窗口切图
def TifCrop(TifPath, CropSize, RepetitionRate, result_path):
    dataset_img = readTif(TifPath)
    raw_image_name, _ = os.path.basename(TifPath).split('.')
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    new_name = 0
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        crop_y1, crop_y2 = int(i * CropSize * (1 - RepetitionRate)), int(i * CropSize * (1 - RepetitionRate)) + CropSize
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            crop_x1, crop_x2 = int(j * CropSize * (1 - RepetitionRate)), int(
                j * CropSize * (1 - RepetitionRate)) + CropSize
            if (len(img.shape) == 2):
                # 单波段
                cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
            else:
                # 多波段
                cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
            image_path = os.path.join(result_path, 'images', raw_image_name + "_" + str(new_name) + ".tif")
            if writeTiff(cropped, geotrans, proj, image_path):
                relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
                new_name = new_name + 1
    # 向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        crop_y1, crop_y2 = int(i * CropSize * (1 - RepetitionRate)), int(i * CropSize * (1 - RepetitionRate)) + CropSize
        crop_x1, crop_x2 = max((width - CropSize), 0), width
        if (len(img.shape) == 2):
            cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
        else:
            cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
        # 写图像
        image_path = os.path.join(result_path, 'images', raw_image_name + "_" + str(new_name) + ".tif")
        if writeTiff(cropped, geotrans, proj, image_path):
            relate_loc.append([raw_image_name+"_"+str(new_name),crop_x1,crop_y1,crop_x2,crop_y2])
            new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        crop_y1, crop_y2 = max((height - CropSize), 0), height
        crop_x1, crop_x2 = int(j * CropSize * (1 - RepetitionRate)), int(j * CropSize * (1 - RepetitionRate)) + CropSize
        if (len(img.shape) == 2):
            cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
        else:
            cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
        image_path = os.path.join(result_path, 'images', raw_image_name + "_" + str(new_name) + '.tif')
        if writeTiff(cropped, geotrans, proj, image_path):
            relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
            new_name = new_name + 1
    #  裁剪右下角
    crop_y1, crop_y2 = max((height - CropSize), 0), height
    crop_x1, crop_x2 = max((width - CropSize), 0), width
    if (len(img.shape) == 2):
        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        cropped = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    image_path = os.path.join(result_path, 'images', raw_image_name + "_" + str(new_name) + ".tif")
    if writeTiff(cropped, geotrans, proj, image_path):
        relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
        new_name = new_name + 1


root_dir = r"datasets/test_orig"
split_size = 1000
RepetitionRate = 0.1  # 重叠率
is_all_selected = 1   # 是否需要切全部的测试集图片（需要挑选置为0）
result_path = r"datasets/test"
image_dir = os.path.join(result_path, 'images')
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
# 从验证集选10张测试
import random
select_list = random.sample(range(201),10)
for i,img_path in enumerate(os.listdir(os.path.join(root_dir,'images'))):
    if is_all_selected or (i in select_list):
        img_name,_ = img_path.split('.')
        image_path = os.path.join(root_dir,'images',img_path)
        print(image_path)
        TifCrop(image_path,split_size,RepetitionRate,result_path)
write_relate_loc2txt(relate_loc,os.path.join(result_path,"relate_test.txt"))
write_test_txt(relate_loc,os.path.join(result_path,"test.txt"))
print('Done')