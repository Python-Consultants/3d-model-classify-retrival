from keras.models import model_from_json
import argparse
from FeaGeneration import GetMainVariable
from showPlot import show_plot
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def findClass4File(Path, clf):
    feature, label = GetMainVariable(Path)
    predict_label = clf.predict_classes(feature)
    le = LabelEncoder()
    le.classes_ = np.load('classes.npy')
    return(feature, le.inverse_transform(predict_label))

def findSimilarFile(Path, clf):
    input_feature, pred_label = findClass4File(Path, clf)
    pred_label = pred_label[0]
    similarity_dic = {}
    for SubFile in os.listdir('ModelNet2/' + pred_label + '/train'):
        SubPath = 'ModelNet2/' + pred_label + '/train/' + SubFile
        try:
            _Feature, _Label = GetMainVariable(SubPath)
            similarity_dic[SubFile] = euclidDistance(input_feature, _Feature)
        except:
            continue
    return(similarity_dic)

def findMostSimilarFile(Path, clf):
    sample_dict = findSimilarFile(Path, clf)
    sorted_x = sorted(sample_dict.items(), key=lambda kv: kv[1])
    return(sorted_x[:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file2Predict")
    args = parser.parse_args()
    FILEPATH = args.file2Predict
    show_plot(FILEPATH)
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    similar_result = findMostSimilarFile(FILEPATH, loaded_model)

    for i in similar_result:
        predLabel = i[0].split('_')[0]
        similarPath = 'ModelNet2/' + predLabel + '/train/' +i[0]
        show_plot(similarPath)
    return()

main()
