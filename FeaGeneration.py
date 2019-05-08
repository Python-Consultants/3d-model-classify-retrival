import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
from scipy.special import sph_harm
from sklearn.decomposition import PCA

def ReadOff(Path):
    File = open(Path, 'rb')
    AllData = File.readlines()
    Data = np.zeros((AllData.__len__(), 3))
    for loop in range(2, AllData.__len__()):
        String = AllData[loop]
        ListString = String.split()
        if ListString.__len__() > 3:
            global FinaLength
            FinaLength = loop
            break
        String0 = float(ListString[0])
        String1 = float(ListString[1])
        String2 = float(ListString[2])
        Data[loop - 2, 0] = String0
        Data[loop - 2, 1] = String1
        Data[loop - 2, 2] = String2
    Data = np.delete(Data, np.s_[FinaLength - 2:], 0)

    return Data


def PathInfo():
    ParentPath = 'ModelNet2'

def GetSRow(Path):
    temp_raw = ReadOff(Path)
    temp = temp_raw[np.arange(start=0, stop=temp_raw.shape[0], step=int(temp_raw.shape[0]/512)), :]
    temp = temp[:512,]
    #print(temp.shape)
    theta = np.arccos(temp[:, 2] / np.sqrt(np.square(temp[:, 0]) + np.square(temp[:, 1]) + np.square(temp[:, 2])))
    phi = np.asarray_chkfinite(temp[:, 1] / np.sqrt(np.square(temp[:, 0]) + np.square(temp[:, 1])))
    del temp
    theta, phi = np.meshgrid(theta, phi)
    s = sph_harm(3, 3, theta, phi).real
    return s


def GetMainVariable(path):
    s = GetSRow(path)
    clf = PCA(n_components=3)
    clf.fit(s)
    Feature = clf.components_.reshape(1,3,512)
    #Feature = np.resize(Feature, (3, 1000))
    Label = path.split('/')[1]
    #LabelName = os.listdir('ModelNet40')
    #matches = next((loop for loop in range(Label.__len__()) if Label == LabelName[loop]))
    return Feature , Label


def GetTrainData():
    count = 0
    Feature, Label = GetMainVariable('ModelNet2/glass_box/train/glass_box_0002.off')
    deleted_path = []
    for Folder in os.listdir('ModelNet2'):
        if (Folder == '.DS_Store'):
            continue
        for SubFile in os.listdir('ModelNet2/' + Folder + '/train'):
            SubPath = 'ModelNet2/' + Folder + '/train/' + SubFile
            if (ReadOff(SubPath).shape[0]<512):
                count+=1
                deleted_path.append(SubPath)
            else:
                try:
                    _Feature, _Label = GetMainVariable(SubPath)
                    Feature = np.vstack((Feature, _Feature))
                    Label = np.hstack((Label, _Label))
                except:
                    continue
    print('missed Value:', count)
    print('deleted path:', deleted_path)
    return Feature, Label


def GetTestData():
    count = 0
    Feature, Label = GetMainVariable('ModelNet2/mantel/test/mantel_0286.off')
    deleted_path = []
    for Folder in os.listdir('ModelNet2'):
        if (Folder == '.DS_Store'):
            continue
        for SubFile in os.listdir('ModelNet2/' + Folder + '/test'):
            SubPath = 'ModelNet2/' + Folder + '/test/' + SubFile
            if (ReadOff(SubPath).shape[0]<512):
                count+=1
                deleted_path.append(SubPath)
            else:
                try:
                    _Feature, _Label = GetMainVariable(SubPath)
                    Feature = np.vstack((Feature, _Feature))
                    Label = np.hstack((Label, _Label))
                except:
                    continue
    print('missed Value:', count)
    print('deleted path:', deleted_path)
    return Feature, Label
