from bs4 import BeautifulSoup
from sklearn import metrics
import pandas as pd
import numpy as np
import os
from os import path


def get_ground_truth_label(filepath, task_name):
    with open(filepath, "r") as file:
        bf = BeautifulSoup(file.read().lower(), "xml")
        find_item_name = "aspectterm" if task_name == "atsa" else "aspectcategory"
        return [polarity2num[aspect_term['polarity']] for aspect_term in bf.find_all(find_item_name)]

def get_ernie_related_output(filepath):
    with open(filepath, "r") as file:
        return file.read().strip().split("\n")

def get_ernie_unrelated_output(filepath):
    df = pd.read_csv(filepath)
    data = df[['label_2', 'label_1', 'label_0']].as_matrix()
    preds = [np.argmax(item) for item in data]
    return preds

def merge_result(pred_res, test_path, save_path, task_name):
    with open(test_path, "r") as file:
        bf = BeautifulSoup(file.read().lower(), "xml")
        find_item_name = "aspectterm" if task_name == "ATSA" else "aspectcategory"
        for aspect_term, polarity in zip(bf.find_all(find_item_name), pred_res):
            aspect_term['polarity'] = polarity
    with open(save_path, "w", encoding='utf-8') as file:
        file.write(bf.prettify())

def f1_score(preds, labels):
    return metrics.f1_score(labels, preds, average='macro')

def to_one_hot(index):
    tmp = np.zeros((3))
    tmp[index] = 1
    return tmp


num2polarity = {0:"negative", 1:"positive", 2:"neutral"}
polarity2num = {item[1]:item[0] for item in num2polarity.items()}
voting_dir = path.abspath(path.dirname(__file__))
ernie_base_dir = voting_dir + path.sep + "output-ernie-related"
none_ernie_base_dir = voting_dir + path.sep + "output-ernie-unrelated"
save_dir = voting_dir + path.sep + "final-result"
test_dir = path.dirname(voting_dir) + path.sep + "ERNIE-unrelated" + path.sep + "data"

#==================================ATSA=============================================
taskname = "ATSA"
atsa_names = ["test_out.atsa_base_11750.tsv.0.0", "test_out.atsa_base_21750.tsv.0.0", "test_out.atsa_large_8750.tsv.0.0", "test_out.atsa_large_18250.tsv.0.0", "test_out.atsa_base_merge_11750.tsv.0.0", "test_out.atsa_base_merge_21750.tsv.0.0", "test_out.atsa_large_merge_8750.tsv.0.0", "test_out.atsa_large_merge_18250.tsv.0.0"]
weights = [1.01, 1.02, 1.11, 2.13, 1.06, 1.07, 1.16, 2.18]
vote = None
for filename, weight in zip(atsa_names, weights):
    pred_path = ernie_base_dir + path.sep + filename
    preds = get_ernie_related_output(pred_path)
    preds = np.asarray([to_one_hot(polarity2num[item]) * weight for item in preds])
    vote = preds if vote is None else (vote + preds)

filenames = os.listdir(none_ernie_base_dir + path.sep + taskname)
for filename in filenames:
    filepath = none_ernie_base_dir + path.sep + taskname + path.sep + filename
    preds = get_ernie_unrelated_output(filepath)
    preds = np.asarray([to_one_hot(item) for item in preds])
    vote = vote + preds


preds = [num2polarity[np.argmax(item)] for item in vote]
test_path = test_dir + path.sep + taskname + path.sep + "test.xml"
save_path = save_dir +  path.sep + "test_result_" +taskname+ ".xml"
merge_result(preds, test_path, save_path, taskname)


#==================================ACSA=============================================
taskname = "ACSA"
acsa_names = ["test_out.acsa_base_9000.tsv.0.0", "test_out.acsa_base_17731.tsv.0.0", "test_out.acsa_large_10750.tsv.0.0", "test_out.acsa_large_11500.tsv.0.0", "test_out.acsa_base_merge_9000.tsv.0.0", "test_out.acsa_base_merge_17731.tsv.0.0", "test_out.acsa_large_merge_10750.tsv.0.0", "test_out.acsa_large_merge_11500.tsv.0.0"]
weights = [1.01, 1.02, 1.11, 2.13, 1.06, 1.07, 1.16, 2.18]
vote = None
for filename, weight in zip(acsa_names, weights):
    pred_path = ernie_base_dir + path.sep + filename
    preds = get_ernie_related_output(pred_path)
    preds = np.asarray([to_one_hot(polarity2num[item]) * weight for item in preds])
    vote = preds if vote is None else (vote + preds)

filenames = os.listdir(none_ernie_base_dir + path.sep + taskname)
for filename in filenames:
    filepath = none_ernie_base_dir + path.sep + taskname + path.sep + filename
    preds = get_ernie_unrelated_output(filepath)
    preds = np.asarray([to_one_hot(item)  for item in preds])
    vote = vote + preds

preds = [num2polarity[np.argmax(item)] for item in vote]
test_path = test_dir + path.sep + taskname + path.sep + "test.xml"
save_path = save_dir +  path.sep + "test_result_" +taskname+ ".xml"
merge_result(preds, test_path, save_path, taskname)
