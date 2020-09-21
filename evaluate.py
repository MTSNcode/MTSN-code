import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
# import tensorflow as tf
# import tqdm
# from gensim.models.keyedvectors import Vocab
from numpy import random
from six import iteritems
from scipy.io import (savemat, loadmat)
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)

from lp_utils import *

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]
        death = ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
        return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])-1), str(int(edge[1])-1))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])-1), str(int(edge[1])-1))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)

def predict_model(file_name, H, eval_type, time_counter,node_matching):
    print('eval time stamp: {}'.format(time_counter))

    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid_{}.txt'.format(time_counter))
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test_{}.txt'.format(time_counter))

    # edge_types
    edge_types = ['1']
    edge_type_count = 1
    # edge_type_count = len(eval_type) - 1

    td = H
    final_model = {}
    try:
        if node_matching == True:
            for i in range(0,len(td)):
                final_model[str(int(td[i][0]))] = td[i][1:]
        else:
            for i in range(0,len(td)):
                final_model[str(i)]=td[i]
        print("load node nums:%d \n" %(len(td)))
    except:
        td = td.tocsr()
        if node_matching == True:
            for i in range(0,td.shape[0]):
                final_model[str(int(td[i][0]))] = td[i][1:]
        else:
            for i in range(0,td.shape[0]):
                final_model[str(i)]=td[i]
        print("load node nums:%d \n" %(td.shape[0]))

    

    valid_aucs, valid_f1s, valid_prs = [], [], []
    test_aucs, test_f1s, test_prs = [], [], []
    # for epoch in range(epochs):
    for i in range(edge_type_count):
        
        if eval_type == 'all' or edge_types[i] in eval_type.split(','):
            tmp_auc, tmp_f1, tmp_pr = evaluate(final_model, valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
            valid_aucs.append(tmp_auc)
            valid_f1s.append(tmp_f1)
            valid_prs.append(tmp_pr)
            print('valid auc: %.4f\tvalid pr: %.4f\tvalid f1: %.4f' %(tmp_auc,tmp_pr,tmp_f1))

            tmp_auc, tmp_f1, tmp_pr = evaluate(final_model, testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
            test_aucs.append(tmp_auc)
            test_f1s.append(tmp_f1)
            test_prs.append(tmp_pr)
            # print('valid auc: %.4f\tvalid pr: %.4f\tvalid f1: %.4f' %(np.mean(valid_aucs),np.mean(valid_prs),np.mean(valid_f1s)))
            

    average_auc = np.mean(test_aucs)
    average_f1 = np.mean(test_f1s)
    average_pr = np.mean(test_prs)
    # return average_auc, average_f1, average_pr
    return np.mean(valid_aucs), np.mean(valid_f1s), np.mean(valid_prs), np.mean(test_aucs), np.mean(test_f1s), np.mean(test_prs)