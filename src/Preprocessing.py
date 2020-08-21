from PIL import Image
import numpy as np
from glob import glob
import csv

resizeRate = 0.8
digitNumber = 6
characterNumber = 10+26

def getImageSize(path):
    image = Image.open(path+"/000000.jpg")
    image = image.resize((int(np.array(image).shape[1]*resizeRate),int(np.array(image).shape[0]*resizeRate)))
    return np.array(image).shape

def getRGBData(path):
    xData = []
    fileSet = sorted(glob(path+ "/*.jpg"))
    for file in fileSet:
        image = Image.open(file)
        image = image.resize((int(np.array(image).shape[1]*resizeRate),int(np.array(image).shape[0]*resizeRate)))
        xData.append(np.array(image))
        index = fileSet.index(file)
        total = len(fileSet)
        print('\r' + '[Load training data]:[%s%s]%.2f%%' % ('█' * int(index*20/total), ' ' * (20-int(index*20/total)),float(index/total*100)), end='')
        print('\n')
    return np.array(xData)

def getLableData(path):
    yLabel = [[] for digitIndex in range(digitNumber)]
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            for characterIndex in range(len(row['code'])):
                if ord(row['code'][characterIndex]) <= 57:
                    yLabel[characterIndex].append(ord(row['code'][characterIndex])-48)
                else:
                    yLabel[characterIndex].append(ord(row['code'][characterIndex])-65+10)
            count += 1
    return yLabel

def load_data(**kw):
    return (getRGBData(kw['xTrainPath']), getLableData(kw['yTrainPath']))