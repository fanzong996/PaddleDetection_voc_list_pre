import os
import random
#需要根据自己的目录进行修改
trainval_percent = 0.95  # 训练集验证集总占比
train_percent = 0.9  # 训练集在trainval_percent里的train占比
xmlfilepath = r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\Annotations'
txtsavepath = r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\ImageSets\Main\trainval.txt', 'w')
ftest = open(r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\ImageSets\Main\test.txt', 'w')
ftrain = open(r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\ImageSets\Main\train.txt', 'w')
fval = open(r'data\PaddleDetection\dataset\voc\VOCdevkit\VOC2012\ImageSets\Main\val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
