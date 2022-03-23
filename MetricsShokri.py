import argparse
import pandas as pd
import os
import sys
import numpy as np
import torch
from utils import computeMetricsAlt, evalThresholdAlt
from ModelShokri import DataHandler, TrainWBAttacker
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Analyse criteria obtained from different MIAs.')

parser.add_argument('--model_type', type=str, help='Model Architecture to attack.')
parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations for empirical estimation.')
parser.add_argument('--working_dir', type=str, default='./', help='Where to collect and store data.')

exp_parameters = parser.parse_args()

currdir = exp_parameters.working_dir

num_runs_for_random = exp_parameters.num_iters
model_type = exp_parameters.model_type

# Extracting intermediate outputs and gradients of the model

InterOuts_Grads0 = np.load(currdir + '/RawResults/NasrTrain0_' + model_type + '.npz')
InterOuts_Grads1 = np.load(currdir + '/RawResults/NasrTrain1_' + model_type + '.npz')
AdditionalInfo = np.load(currdir + '/RawResults/NasrAddInfo_' + model_type + '.npz')

inter_outs0 = []
inter_outs1 = []

out_size_list = AdditionalInfo['arr_0']
layer_size_list = AdditionalInfo['arr_1']
kernel_size_list = AdditionalInfo['arr_2']

n_inter_outputs = len(out_size_list)
n_layer_grads = len(kernel_size_list)

for i in range(n_inter_outputs):
    inter_outs0.append(InterOuts_Grads0['arr_' + str(i)])
    inter_outs1.append(InterOuts_Grads1['arr_' + str(i)])

lossval0 = InterOuts_Grads0['arr_' + str(n_inter_outputs)]
lossval1 = InterOuts_Grads1['arr_' + str(n_inter_outputs)]

labels1hot0 = InterOuts_Grads0['arr_' + str(n_inter_outputs + 1)]
labels1hot1 = InterOuts_Grads1['arr_' + str(n_inter_outputs + 1)]

grad_vals0 = []
grad_vals1 = []

for i in range(n_inter_outputs + 2, n_inter_outputs + 2 + n_layer_grads, 1):
    grad_vals0.append(InterOuts_Grads0['arr_' + str(i)])
    grad_vals1.append(InterOuts_Grads1['arr_' + str(i)])

# Our Analysis

FPR = np.linspace(0, 1, num=1001)

try:
    dfMetricsBalanced = pd.read_csv(currdir + '/CompleteResults/BalancedMetrics_' + model_type + '.csv')
    dfTPRBalanced = pd.read_csv(currdir + '/CompleteResults/BalancedROC_' + model_type + '.csv')
except FileNotFoundError:
    dfMetricsBalanced = pd.DataFrame(columns=['Attack Strategy',
                                              'AUROC', 'AUROC STD',
                                              'Best Accuracy', 'Best Accuracy STD',
                                              'FPR at TPR80', 'FPR at TPR80 STD',
                                              'FPR at TPR85', 'FPR at TPR85 STD',
                                              'FPR at TPR90', 'FPR at TPR90 STD',
                                              'FPR at TPR95', 'FPR at TPR95 STD'])
    dfTPRBalanced = pd.DataFrame(FPR, columns=['FPR'])

aux_list_metrics = []
aux_list_TPR = []
for k in range(num_runs_for_random):
    np.random.seed(k)
    indx_train0 = np.random.choice(lossval0.shape[0], size=4000, replace=False)
    indx_train1 = np.random.choice(lossval1.shape[0], size=4000, replace=False)

    indx_test0 = np.setdiff1d(np.arange(lossval0.shape[0]), indx_train0)
    indx_test0 = np.random.choice(indx_test0, size=6000, replace=False)
    indx_test1 = np.setdiff1d(np.arange(lossval1.shape[0]), indx_train1)
    indx_test1 = np.random.choice(indx_test1, size=6000, replace=False)

    trainingData = DataHandler(inter_outs0, inter_outs1, lossval0, lossval1, labels1hot0, labels1hot1,
                               grad_vals0, grad_vals1, indx_train0, indx_train1)

    Max = trainingData.Max
    Min = trainingData.Min

    testingData = DataHandler(inter_outs0, inter_outs1, lossval0, lossval1, labels1hot0, labels1hot1,
                              grad_vals0, grad_vals1, indx_test0, indx_test1, Max=Max, Min=Min)

    AttackerShokri = TrainWBAttacker(trainingData, testingData, out_size_list, layer_size_list, kernel_size_list)

    dataloaderEval = DataLoader(testingData, batch_size=100, shuffle=False)
    scoresEval = []
    EvalY = []
    with torch.no_grad():
        for i, batch in enumerate(dataloaderEval):
            example = batch[0]
            target = batch[1]
            scoresEval.append(AttackerShokri(*example).detach())
            EvalY.append(target.cpu().data.numpy())
    scoresEval = torch.cat(scoresEval, axis=0)
    scoresEval = torch.squeeze(scoresEval)
    scoresEval = scoresEval.cpu().data.numpy()
    EvalY = np.squeeze(np.concatenate(EvalY, axis=0))

    TPR_, metrics_ = computeMetricsAlt(scoresEval, EvalY, FPR)
    aux_list_metrics.append(metrics_)
    aux_list_TPR.append(TPR_)

metrics = np.stack(aux_list_metrics, 1)
mean_metrics = np.mean(metrics, 1)
std_metrics = np.std(metrics, 1)

new_row = {"Attack Strategy": 'Nasr White-Box',
           'AUROC': mean_metrics[0], 'AUROC STD': std_metrics[0],
           'Best Accuracy': mean_metrics[1], 'Best Accuracy STD': std_metrics[1],
           'FPR at TPR80': mean_metrics[2], 'FPR at TPR80 STD': std_metrics[2],
           'FPR at TPR85': mean_metrics[3], 'FPR at TPR85 STD': std_metrics[3],
           'FPR at TPR90': mean_metrics[4], 'FPR at TPR90 STD': std_metrics[4],
           'FPR at TPR95': mean_metrics[5], 'FPR at TPR95 STD': std_metrics[5]}

dfMetricsBalanced = dfMetricsBalanced.append(new_row, ignore_index=True)

TPR = np.stack(aux_list_TPR, 1)
mean_TPR = np.mean(TPR, 1)
std_TPR = np.std(TPR, 1)

dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1), columns=['Nasr White-Box TPR',
                                                                        'Nasr White-Box TPR STD'])
dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

# Rezaei Analysis

try:
    dfMetricsRezaei = pd.read_csv(currdir + '/CompleteResults/RezaeiMetrics_' + model_type + '.csv')
except FileNotFoundError:
    dfMetricsRezaei = pd.DataFrame(columns=['Attack Strategy',
                                            'Best Accuracy', 'Best Accuracy STD',
                                            'FPR', 'FPR STD'])

aux_list_metrics = []
for k in range(num_runs_for_random):
    np.random.seed(k)
    indx_train0 = np.random.choice(lossval0.shape[0], size=8000, replace=False)
    indx_train1 = np.random.choice(lossval1.shape[0], size=40000, replace=False)

    indx_test0 = np.setdiff1d(np.arange(lossval0.shape[0]), indx_train0)
    indx_test0 = np.random.choice(indx_test0, size=2000, replace=False)
    indx_test1 = np.setdiff1d(np.arange(lossval1.shape[0]), indx_train1)
    indx_test1 = np.random.choice(indx_test1, size=10000, replace=False)

    trainingData = DataHandler(inter_outs0, inter_outs1, lossval0, lossval1, labels1hot0, labels1hot1,
                               grad_vals0, grad_vals1, indx_train0, indx_train1)

    Max = trainingData.Max
    Min = trainingData.Min

    testingData = DataHandler(inter_outs0, inter_outs1, lossval0, lossval1, labels1hot0, labels1hot1,
                              grad_vals0, grad_vals1, indx_test0, indx_test1, Max=Max, Min=Min)

    AttackerShokri = TrainWBAttacker(trainingData, testingData, out_size_list, layer_size_list, kernel_size_list)

    dataloaderEval = DataLoader(testingData, batch_size=100, shuffle=False)
    scoresEval = []
    EvalY = []
    for i, batch in enumerate(dataloaderEval):
        example = batch[0]
        target = batch[1]
        scoresEval.append(AttackerShokri(*example))
        EvalY.append(target.cpu().data.numpy())
    scoresEval = torch.cat(scoresEval, axis=0)
    scoresEval = torch.squeeze(scoresEval)
    scoresEval = scoresEval.cpu().data.numpy()
    EvalY = np.squeeze(np.concatenate(EvalY, axis=0))

    metrics_ = evalThresholdAlt(0.5, scoresEval, EvalY)
    aux_list_metrics.append(metrics_)

metrics = np.stack(aux_list_metrics, 1)
mean_metrics = np.mean(metrics, 1)
std_metrics = np.std(metrics, 1)

new_row = {"Attack Strategy": 'Nasr White-Box',
           'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
           'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

print('Evaluation of Shokri White-Box: done')
sys.stdout.flush()
sys.stderr.flush()

if not os.path.exists(currdir + '/CompleteResults'):
    os.makedirs(currdir + '/CompleteResults')

dfMetricsBalanced.to_csv(currdir + '/CompleteResults/BalancedMetrics_' + model_type + '.csv', index=False)
dfTPRBalanced.to_csv(currdir + '/CompleteResults/BalancedROC_' + model_type + '.csv', index=False)

dfMetricsRezaei.to_csv(currdir + '/CompleteResults/RezaeiMetrics_' + model_type + '.csv', index=False)
