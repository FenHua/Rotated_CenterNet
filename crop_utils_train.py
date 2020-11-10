import os
import sys
import gdal
import random
import numpy as np

category = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
            'ship', 'tennis-court','basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
            'harbor', 'swimming-pool', 'helicopter', "container-crane"]   # 定义类别数[比赛类别要改]
categoryToID = {category[i]: i + 1 for i in range(len(category))}         # 类别与index的映射
relate_loc = []                                                           # 用来存放大图与切片的位置关系
select_category = [2, 4, 5, 6, 8, 9, 12]                                  # 拟使用的类别


# 制作trainval.txt(用来记录哪些数据用来训练)
def write_trainval_txt(relate_locs, txt_path):
    with open(txt_path, 'a+') as f:
        for relate_loc in relate_locs:
            f.write(relate_loc[0] + '\n')   # 回车切分


# 把小图相对于大图的坐标写入relate.txt
# relate_loc = [crop_image_name_id,crop_x1,crop_y1,,crop_x2,crop_y2]
def write_relate_loc2txt(relate_locs, txt_path):
    with open(txt_path, 'a+') as f:
        for relate_loc in relate_locs:
            relate_loc = [str(i) for i in relate_loc]
            sep = ','   # 分隔符
            f.write(sep.join(relate_loc) + '\n')


# 顺时针读取DOTA数据集标注信息
def DOTA_read_txt(raw_txt_path):
    box_all = []
    no_use = 0     # 前面两行介绍信息去掉
    with open(raw_txt_path, 'r') as f:
        for line in f.readlines():
            if no_use < 2:
                no_use += 1
                continue
            lines = line.strip("\n").split(' ')                  # bbox信息
            # 判断当前bbox的类别是否是需要的类别
            if categoryToID[lines[8]] in select_category:
                # 为了后续方便，把类别转为类别id
                lines[8] = str(categoryToID[lines[8]])
                box_all.append([int(float(i)) for i in lines])   # 全部转换为数值列表
    return np.array(box_all)


# 比赛逆时针(删掉)
def Contest_read_txt(raw_txt_path):
    box_all = []
    # （目标类别标签 x1 y1 x2 y2 x3 y3 x4 y4 ）
    with open(raw_txt_path, 'r', encoding='gbk') as f:
        for line in f.readlines():
            lines = line.strip("\n").split(' ')
            lines[0] = str(categoryToID[lines[0]])
            lines[3], lines[7] = lines[7], lines[3]
            lines[4], lines[8] = lines[8], lines[4]
            # 添加difficult为0
            # DOTA[顺时针] （x1 y1 x2 y2 x3 y3 x4 y4 目标类别标签 difficult）
            box_all.append([float(lines[i]) for i in range(1, 9)] + [int(lines[0]), 0])
    return np.array(box_all)


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


# 借助gdal读取tif数据
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape  # gdal读取tif格式的图片通道在前面
        new_im_data = im_data
    elif len(im_data.shape) == 2:
        # 比赛单波段（16转8）
        new_im_data = uint16to8(im_data, im_geotrans, im_proj, path)
        if new_im_data is None:
            return 0
        im_bands, im_height, im_width = new_im_data.shape
    # 判断原始图片的格式（int8，int16，float32等）
    if 'int8' in new_im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in new_im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    # 创建tiff图片
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)        # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(new_im_data[i])
    del dataset
    return 1


# 写入切片的标签[x1,y1,...,x4,y4,class,difficult] [顺时针]
def write_txt(txt_path, box_all):
    with open(txt_path, 'w+') as f:
        for box in box_all:
            box = [str(i) for i in list(box)]
            seq = ' '
            f.write(seq.join(box[:-2]) + seq + category[int(box[-2]) - 1] + seq + box[-1] + '\n')


# 保存切片图像以及对应的标注信息（进行初步的判断和筛选）
def write_txt_and_image(image_name, box_all, cropped, im_geotrans, im_proj, crop_location, result_path):
    crop_x1, crop_x2, crop_y1, crop_y2 = crop_location    # 切片位置
    txt_path = os.path.join(result_path, 'labelTxt', image_name + ".txt")
    image_path = os.path.join(result_path, 'images', image_name + ".tif")
    new_box_all = np.zeros(box_all.shape, dtype=box_all.dtype)
    new_box_all[:, 0] = box_all[:, 0] - crop_x1
    new_box_all[:, 2] = box_all[:, 2] - crop_x1
    new_box_all[:, 4] = box_all[:, 4] - crop_x1
    new_box_all[:, 6] = box_all[:, 6] - crop_x1
    new_box_all[:, 1] = box_all[:, 1] - crop_y1
    new_box_all[:, 3] = box_all[:, 3] - crop_y1
    new_box_all[:, 5] = box_all[:, 5] - crop_y1
    new_box_all[:, 7] = box_all[:, 7] - crop_y1
    new_box_all[:, 8:] = box_all[:, 8:]
    xmin = np.min(new_box_all[:, [0, 2, 4, 6]], axis=1)
    xmax = np.max(new_box_all[:, [0, 2, 4, 6]], axis=1)
    ymin = np.min(new_box_all[:, [1, 3, 5, 7]], axis=1)
    ymax = np.max(new_box_all[:, [1, 3, 5, 7]], axis=1)
    # 长宽限制
    cond1 = np.intersect1d(np.where(xmin >= 0)[0], np.where(ymin >= 0)[0])
    cond2 = np.intersect1d(np.where(ymax <= (crop_y2 - crop_y1))[0],
                           np.where(xmax <= (crop_x2 - crop_x1))[0])
    idx = np.intersect1d(cond1, cond2)
    # 如果存在目标，保存切片
    if len(idx) > 0:
        #  写图像
        if writeTiff(cropped, im_geotrans, im_proj, image_path):
            write_txt(txt_path, new_box_all[idx])
            return 1
    else:
        # 是否保留没有标签的切片
        is_save = random.random()
        if is_save < 0.1:
            # 以0.1的概率保留无目标切片
            if writeTiff(cropped, im_geotrans, im_proj, image_path):
                with open(txt_path, 'w+') as f:
                    pass
                return 1
    return 0


# 滑窗切图
def TifCrop(TifPath, CropSize, RepetitionRate, raw_txt_path, result_path, is_DOTA=True):
    # RepetitionRate 重叠率
    if is_DOTA:
        box_all = DOTA_read_txt(raw_txt_path)
    else:
        box_all = Contest_read_txt(raw_txt_path)
    if box_all.shape[0]==0:
        return
    dataset_img = readTif(TifPath)                              # 读tiff
    raw_image_name, _ = os.path.basename(TifPath).split('.')    # 图片名称
    width = dataset_img.RasterXSize                             # 获取图片属性
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)          # 获取图片数据
    new_name = 0                                                # 当前大图切片的数量
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        # y方向的裁剪宽度
        crop_y1, crop_y2 = int(i * CropSize * (1 - RepetitionRate)), int(i * CropSize * (1 - RepetitionRate)) + CropSize
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  x方向的裁剪宽度
            crop_x1, crop_x2 = int(j * CropSize * (1 - RepetitionRate)), int(
                j * CropSize * (1 - RepetitionRate)) + CropSize
            if (len(img.shape) == 2):
                # 单通道
                cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
            else:
                # 多通道
                cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
            if write_txt_and_image(raw_image_name + "_" + str(new_name), box_all, cropped, geotrans, proj,
                                   (crop_x1, crop_x2, crop_y1, crop_y2), result_path):
                # 当前切片已经保存，则保存切片与大图之间的映射
                relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
                new_name = new_name + 1
    # 单独处理最右边
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        crop_y1, crop_y2 = int(i * CropSize * (1 - RepetitionRate)), int(i * CropSize * (1 - RepetitionRate)) + CropSize
        crop_x1, crop_x2 = max((width - CropSize), 0), width
        if (len(img.shape) == 2):
            cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
        else:
            cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
        #  写图像
        if (write_txt_and_image(raw_image_name + "_" + str(new_name), box_all, cropped, geotrans, proj,
                                (crop_x1, crop_x2, crop_y1, crop_y2), result_path)):
            relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
            new_name = new_name + 1
    #  单独处理最下边
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        crop_y1, crop_y2 = max((height - CropSize), 0), height
        crop_x1, crop_x2 = int(j * CropSize * (1 - RepetitionRate)), int(j * CropSize * (1 - RepetitionRate)) + CropSize

        if (len(img.shape) == 2):
            cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
        else:
            cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]

        if (write_txt_and_image(raw_image_name + "_" + str(new_name), box_all, cropped, geotrans, proj,
                                (crop_x1, crop_x2, crop_y1, crop_y2), result_path)):
            relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
            new_name = new_name + 1
    #  裁剪右下角 切[500*500]
    crop_y1, crop_y2 = max((height - CropSize), 0), height
    crop_x1, crop_x2 = max((width - CropSize), 0), width
    if (len(img.shape) == 2):
        cropped = img[crop_y1: crop_y2, crop_x1: crop_x2]
    else:
        cropped = img[:, crop_y1: crop_y2, crop_x1: crop_x2]
    if (write_txt_and_image(raw_image_name + "_" + str(new_name), box_all, cropped, geotrans, proj,
                            (crop_x1, crop_x2, crop_y1, crop_y2), result_path)):
        relate_loc.append([raw_image_name + "_" + str(new_name), crop_x1, crop_y1, crop_x2, crop_y2])
        new_name = new_name + 1


root_dir = r"/home/lenovo/wn2020/DOTA/train"             # DOTA根目录
split_size = 500                                         # 滑窗大小
RepetitionRate = 0.1                                     # 重叠率
result_path = r"/home/lenovo/wn2020/DOTA/split"          # 切分结果
result_image_dir = os.path.join(result_path, 'images')   # 切分切片结果
if not os.path.exists(result_image_dir):
    os.makedirs(result_image_dir)
result_txt_dir = os.path.join(result_path, 'labelTxt')
if not os.path.exists(result_txt_dir):
    os.makedirs(result_txt_dir)
for img_path in os.listdir(os.path.join(root_dir, 'images')):
    img_name, _ = img_path.split('.')
    image_path = os.path.join(root_dir, 'images', img_path)   # 大图图片
    txt_path = os.path.join(root_dir, 'labelTxt-v1.5/obb', img_name + ".txt")  # 大图标注
    # print(image_path, txt_path)
    TifCrop(image_path, split_size, RepetitionRate, txt_path, result_path)     # 滑窗切图
write_relate_loc2txt(relate_loc, os.path.join(result_path, "relate.txt"))      # 写切片和大图的映射
write_trainval_txt(relate_loc, os.path.join(result_path, "trainval.txt"))      # 写切片标注信息
print('Done')