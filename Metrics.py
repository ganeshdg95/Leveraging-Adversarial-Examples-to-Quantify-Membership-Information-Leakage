import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils import rescale01, computeMetrics, computeBestThreshold, evalBestThreshold

parser = argparse.ArgumentParser(description='Analyse criteria obtained from different MIAs.')

parser.add_argument('--model_type', type=str, help='Model Architecture to attack.')
parser.add_argument('--num_iters', type=int, default=20, help='Number of iterations for empirical estimation.')
parser.add_argument('--mode', type=int, help='What part of the analysis to compute.')
parser.add_argument('--working_dir', type=str, default='./', help='Where to collect and store data.')

exp_parameters = parser.parse_args()

currdir = exp_parameters.working_dir
if not os.path.exists(currdir + '/CompleteResults'):
    os.makedirs(currdir + '/CompleteResults')

num_runs_for_random = exp_parameters.num_iters
model_type = exp_parameters.model_type
mode = exp_parameters.mode

scores0 = pd.read_csv(currdir + '/RawResults/scores0_' + model_type + '.csv')
scores1 = pd.read_csv(currdir + '/RawResults/scores1_' + model_type + '.csv')

FPR = np.linspace(0, 1, num=1001)

# Extracting intermediate outputs and gradients of the model

InterOuts_Grads0 = np.load(currdir + '/RawResults/NasrTrain0_' + model_type + '.npz')
InterOuts_Grads1 = np.load(currdir + '/RawResults/NasrTrain1_' + model_type + '.npz')
AdditionalInfo = np.load(currdir + '/RawResults/NasrAddInfo_' + model_type + '.npz')

out_size_list = AdditionalInfo['arr_0']

n_inter_outputs = len(out_size_list)

logits0 = InterOuts_Grads0['arr_' + str(n_inter_outputs - 1)]
logits1 = InterOuts_Grads1['arr_' + str(n_inter_outputs - 1)]

lastTwo0 = np.concatenate((InterOuts_Grads0['arr_' + str(n_inter_outputs - 1)],
                           InterOuts_Grads0['arr_' + str(n_inter_outputs - 2)]), axis=1)
lastTwo1 = np.concatenate((InterOuts_Grads1['arr_' + str(n_inter_outputs - 1)],
                           InterOuts_Grads1['arr_' + str(n_inter_outputs - 2)]), axis=1)

if mode == 1:
    # Our Evaluation of attacks using a single quantity 
    # Balanced evaluation Set

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

    for column_name in scores0:

        aux_list_metrics = []
        aux_list_TPR = []

        criteria0 = scores0[[column_name]].values
        criteria1 = scores1[[column_name]].values

        for i in range(num_runs_for_random):
            # Setting Random Seed
            np.random.seed(i)

            indx_eval0 = np.random.choice(criteria0.shape[0], size=10000, replace=False)
            indx_eval1 = np.random.choice(criteria1.shape[0], size=10000, replace=False)

            criteria0_eval = criteria0[indx_eval0]
            criteria1_eval = criteria1[indx_eval1]

            TPR_, metrics_ = computeMetrics(criteria0_eval, criteria1_eval, FPR)
            aux_list_metrics.append(metrics_)
            aux_list_TPR.append(TPR_)

        metrics = np.stack(aux_list_metrics, 1)
        mean_metrics = np.mean(metrics, 1)
        std_metrics = np.std(metrics, 1)

        new_row = {"Attack Strategy": column_name,
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

        dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                                columns=[column_name + ' TPR mean', column_name + ' TPR std'])
        dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    print('Balanced evaluation of all scores: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Our ML Attacker

    features_used_for_ML_attacker = ['Softmax Response', 'Modified Entropy', 'Adversarial Distance Linf',
                                     'Loss Value', 'Grad wrt model parameters L2', 'Grad wrt input image L2']

    list_of_train_sizes = [500, 1000, 2000, 4000]

    criteria0 = scores0[features_used_for_ML_attacker].values
    criteria1 = scores1[features_used_for_ML_attacker].values

    # Our analysis with several training set sizes

    for train_set_size in list_of_train_sizes:
        aux_list_metrics = []
        aux_list_TPR = []
        for i in range(num_runs_for_random):
            # Setting Random Seed
            np.random.seed(i)

            # Evaluation Set
            indx_eval0 = np.random.choice(criteria0.shape[0], size=6000, replace=False)
            indx_eval1 = np.random.choice(criteria1.shape[0], size=6000, replace=False)

            indx_diff0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
            indx_diff1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

            # Training Set (When needed)
            indx_training0 = np.random.choice(indx_diff0, size=train_set_size, replace=False)
            indx_training1 = np.random.choice(indx_diff1, size=train_set_size, replace=False)

            evalX0 = criteria0[indx_eval0]
            trainX0 = criteria0[indx_training0]

            evalX1 = criteria1[indx_eval1]
            trainX1 = criteria1[indx_training1]

            trainY0 = np.zeros((trainX0.shape[0]))
            trainY1 = np.ones((trainX1.shape[0]))

            trainX = np.concatenate((trainX0, trainX1), 0)
            trainY = np.concatenate((trainY0, trainY1))

            Max = np.max(trainX, axis=0)
            Min = np.min(trainX, axis=0)

            trainX = rescale01(trainX, Max, Min)
            evalX0 = rescale01(evalX0, Max, Min)
            evalX1 = rescale01(evalX1, Max, Min)

            attackModel = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                        hidden_layer_sizes=(40, 40, 20, 10), random_state=i, max_iter=300)

            attackModel.fit(trainX, trainY)

            preds0 = attackModel.predict_proba(evalX0)[:, 1]
            preds1 = attackModel.predict_proba(evalX1)[:, 1]

            TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
            aux_list_metrics.append(metrics_)
            aux_list_TPR.append(TPR_)

        metrics = np.stack(aux_list_metrics, 1)
        mean_metrics = np.mean(metrics, 1)
        std_metrics = np.std(metrics, 1)

        new_row = {"Attack Strategy": 'ML-Attacker train_set_size: ' + str(train_set_size),
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

        dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                                columns=['ML-Attacker ' + str(train_set_size) + ' TPR mean',
                                         'ML-Attacker ' + str(train_set_size) + ' TPR STD'])
        dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    # Rezaei Attacker Logits

    aux_list_metrics = []
    aux_list_TPR = []
    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(logits0.shape[0], size=6000, replace=False)
        indx_eval1 = np.random.choice(logits1.shape[0], size=6000, replace=False)

        indx_diff0 = np.setdiff1d(np.arange(logits0.shape[0]), indx_eval0)
        indx_diff1 = np.setdiff1d(np.arange(logits1.shape[0]), indx_eval1)

        # Training Set (When needed)
        indx_training0 = np.random.choice(indx_diff0, size=4000, replace=False)
        indx_training1 = np.random.choice(indx_diff1, size=4000, replace=False)

        evalX0 = logits0[indx_eval0]
        trainX0 = logits0[indx_training0]

        evalX1 = logits1[indx_eval1]
        trainX1 = logits1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                          hidden_layer_sizes=(128, 64), random_state=i, max_iter=300)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
        aux_list_metrics.append(metrics_)
        aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Logits',
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

    dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1), columns=['Rezaei Attacker Logits TPR mean',
                                                                            'Rezaei Attacker Logits TPR STD'])
    dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    # Rezaei Attacker Last two layers

    aux_list_metrics = []
    aux_list_TPR = []
    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(lastTwo0.shape[0], size=6000, replace=False)
        indx_eval1 = np.random.choice(lastTwo1.shape[0], size=6000, replace=False)

        indx_diff0 = np.setdiff1d(np.arange(lastTwo0.shape[0]), indx_eval0)
        indx_diff1 = np.setdiff1d(np.arange(lastTwo1.shape[0]), indx_eval1)

        # Training Set (When needed)
        indx_training0 = np.random.choice(indx_diff0, size=4000, replace=False)
        indx_training1 = np.random.choice(indx_diff1, size=4000, replace=False)

        evalX0 = lastTwo0[indx_eval0]
        trainX0 = lastTwo0[indx_training0]

        evalX1 = lastTwo1[indx_eval1]
        trainX1 = lastTwo1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                          hidden_layer_sizes=(128, 64), random_state=i, max_iter=300)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
        aux_list_metrics.append(metrics_)
        aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Last Layers',
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

    dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1), columns=['Rezaei Attacker Last Layers TPR mean',
                                                                            'Rezaei Attacker Last Layers TPR STD'])
    dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    # Rezaei Attacker Grad w.r.t. input images

    features_used_for_Rezaei_attacker = ['Grad wrt input image L1', 'Grad wrt input image L2',
                                         'Grad wrt input image Linf', 'Grad wrt input image Mean',
                                         'Grad wrt input image Skewness', 'Grad wrt input image Kurtosis',
                                         'Grad wrt input image Abs Min']

    aux_list_metrics = []
    aux_list_TPR = []

    criteria0 = scores0[features_used_for_Rezaei_attacker].values
    criteria1 = scores1[features_used_for_Rezaei_attacker].values

    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(criteria0.shape[0], size=6000, replace=False)
        indx_eval1 = np.random.choice(criteria1.shape[0], size=6000, replace=False)

        indx_diff0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
        indx_diff1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

        # Training Set (When needed)
        indx_training0 = np.random.choice(indx_diff0, size=4000, replace=False)
        indx_training1 = np.random.choice(indx_diff1, size=4000, replace=False)

        evalX0 = criteria0[indx_eval0]
        trainX0 = criteria0[indx_training0]

        evalX1 = criteria1[indx_eval1]
        trainX1 = criteria1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-5, random_state=i, solver='saga', max_iter=150)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
        aux_list_metrics.append(metrics_)
        aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Grad w.r.t. input',
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

    dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                            columns=['Rezaei Attacker Grad w.r.t. input TPR mean',
                                     'Rezaei Attacker Grad w.r.t. input TPR STD'])
    dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    # Rezaei Attacker Grad w.r.t. parameters

    features_used_for_Rezaei_attacker = ['Grad wrt model parameters L1', 'Grad wrt model parameters L2',
                                         'Grad wrt model parameters Linf', 'Grad wrt model parameters Mean',
                                         'Grad wrt model parameters Skewness', 'Grad wrt model parameters Kurtosis',
                                         'Grad wrt model parameters Abs Min']

    aux_list_metrics = []
    aux_list_TPR = []

    criteria0 = scores0[features_used_for_Rezaei_attacker].values
    criteria1 = scores1[features_used_for_Rezaei_attacker].values

    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(criteria0.shape[0], size=6000, replace=False)
        indx_eval1 = np.random.choice(criteria1.shape[0], size=6000, replace=False)

        indx_diff0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
        indx_diff1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

        # Training Set (When needed)
        indx_training0 = np.random.choice(indx_diff0, size=4000, replace=False)
        indx_training1 = np.random.choice(indx_diff1, size=4000, replace=False)

        evalX0 = criteria0[indx_eval0]
        trainX0 = criteria0[indx_training0]

        evalX1 = criteria1[indx_eval1]
        trainX1 = criteria1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-5, random_state=i, solver='saga', max_iter=150)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        TPR_, metrics_ = computeMetrics(preds0, preds1, FPR)
        aux_list_metrics.append(metrics_)
        aux_list_TPR.append(TPR_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Grad w.r.t. parameters',
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

    dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                            columns=['Rezaei Attacker Grad w.r.t. parameters TPR mean',
                                     'Rezaei Attacker Grad w.r.t. parameters TPR STD'])
    dfTPRBalanced = dfTPRBalanced.join(dfTPRaux)

    dfMetricsBalanced.to_csv(currdir + '/CompleteResults/BalancedMetrics_' + model_type + '.csv', index=False)
    dfTPRBalanced.to_csv(currdir + '/CompleteResults/BalancedROC_' + model_type + '.csv', index=False)

elif mode == 2:
    # Analysis from Rezaei

    try:
        dfMetricsRezaei = pd.read_csv(currdir + '/CompleteResults/RezaeiMetrics_' + model_type + '.csv')
    except FileNotFoundError:
        dfMetricsRezaei = pd.DataFrame(columns=['Attack Strategy',
                                                'Best Accuracy', 'Best Accuracy STD',
                                                'FPR', 'FPR STD'])

    for column_name in scores0:

        criteria0 = scores0[[column_name]].values
        criteria1 = scores1[[column_name]].values

        aux_list_metrics = []
        for i in range(num_runs_for_random):
            # Setting Random Seed
            np.random.seed(i)

            # Evaluation Set
            indx_eval0 = np.random.choice(criteria0.shape[0], size=2000, replace=False)
            indx_eval1 = np.random.choice(criteria1.shape[0], size=10000, replace=False)

            indx_training0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
            indx_training1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

            eval_criteria0 = criteria0[indx_eval0]
            train_criteria0 = criteria0[indx_training0]

            eval_criteria1 = criteria1[indx_eval1]
            train_criteria1 = criteria1[indx_training1]

            best_threshold = computeBestThreshold(train_criteria0, train_criteria1)

            metrics_ = evalBestThreshold(best_threshold, eval_criteria0, eval_criteria1)
            aux_list_metrics.append(metrics_)

        metrics = np.stack(aux_list_metrics, 1)
        mean_metrics = np.mean(metrics, 1)
        std_metrics = np.std(metrics, 1)

        new_row = {"Attack Strategy": column_name,
                   'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
                   'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

        dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Rezaei evaluation of all scores: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Our ML Attacker

    features_used_for_ML_attacker = ['Softmax Response', 'Modified Entropy', 'Adversarial Distance Linf',
                                     'Loss Value', 'Grad wrt model parameters L2', 'Grad wrt input image L2']

    aux_list_metrics = []

    criteria0 = scores0[features_used_for_ML_attacker].values
    criteria1 = scores1[features_used_for_ML_attacker].values

    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(criteria0.shape[0], size=2000, replace=False)
        indx_eval1 = np.random.choice(criteria1.shape[0], size=10000, replace=False)

        indx_training0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
        indx_training1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

        evalX0 = criteria0[indx_eval0]
        trainX0 = criteria0[indx_training0]

        evalX1 = criteria1[indx_eval1]
        trainX1 = criteria1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModel = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                    hidden_layer_sizes=(40, 40, 20, 10), random_state=i, max_iter=300)

        attackModel.fit(trainX, trainY)

        preds0 = attackModel.predict_proba(evalX0)[:, 1]
        preds1 = attackModel.predict_proba(evalX1)[:, 1]

        metrics_ = evalBestThreshold(0.5, preds0, preds1)
        aux_list_metrics.append(metrics_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Our ML attacker',
               'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
               'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

    dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Evaluation of our ML attacker: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Rezaei Attacker Logits

    aux_list_metrics = []
    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(logits0.shape[0], size=2000, replace=False)
        indx_eval1 = np.random.choice(logits1.shape[0], size=10000, replace=False)

        indx_training0 = np.setdiff1d(np.arange(logits0.shape[0]), indx_eval0)
        indx_training1 = np.setdiff1d(np.arange(logits1.shape[0]), indx_eval1)

        evalX0 = logits0[indx_eval0]
        trainX0 = logits0[indx_training0]

        evalX1 = logits1[indx_eval1]
        trainX1 = logits1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                          hidden_layer_sizes=(128, 64), random_state=i, max_iter=300)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        metrics_ = evalBestThreshold(0.5, preds0, preds1)
        aux_list_metrics.append(metrics_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Logits',
               'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
               'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

    dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Evaluation of Rezaei attacker Last Layer: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Rezaei Attacker Last two layers

    aux_list_metrics = []
    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(lastTwo0.shape[0], size=2000, replace=False)
        indx_eval1 = np.random.choice(lastTwo1.shape[0], size=10000, replace=False)

        indx_training0 = np.setdiff1d(np.arange(lastTwo0.shape[0]), indx_eval0)
        indx_training1 = np.setdiff1d(np.arange(lastTwo1.shape[0]), indx_eval1)

        evalX0 = lastTwo0[indx_eval0]
        trainX0 = lastTwo0[indx_training0]

        evalX1 = lastTwo1[indx_eval1]
        trainX1 = lastTwo1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', early_stopping=False,
                                          hidden_layer_sizes=(128, 64), random_state=i, max_iter=300)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        metrics_ = evalBestThreshold(0.5, preds0, preds1)
        aux_list_metrics.append(metrics_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Last Layers',
               'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
               'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

    dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Evaluation of Rezaei attacker Last Layers: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Rezaei Attacker Grad w.r.t. input images

    features_used_for_Rezaei_attacker = ['Grad wrt input image L1', 'Grad wrt input image L2',
                                         'Grad wrt input image Linf', 'Grad wrt input image Mean',
                                         'Grad wrt input image Skewness', 'Grad wrt input image Kurtosis',
                                         'Grad wrt input image Abs Min']

    aux_list_metrics = []

    criteria0 = scores0[features_used_for_Rezaei_attacker].values
    criteria1 = scores1[features_used_for_Rezaei_attacker].values

    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(criteria0.shape[0], size=2000, replace=False)
        indx_eval1 = np.random.choice(criteria1.shape[0], size=10000, replace=False)

        indx_training0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
        indx_training1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

        evalX0 = criteria0[indx_eval0]
        trainX0 = criteria0[indx_training0]

        evalX1 = criteria1[indx_eval1]
        trainX1 = criteria1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-5, random_state=i, solver='saga', max_iter=150)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        metrics_ = evalBestThreshold(0.5, preds0, preds1)
        aux_list_metrics.append(metrics_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Grad w.r.t. input',
               'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
               'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

    dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Evaluation of Rezaei attacker Grad wrt inputs: done')
    sys.stdout.flush()
    sys.stderr.flush()

    # Rezaei Attacker Grad w.r.t. parameters

    features_used_for_Rezaei_attacker = ['Grad wrt model parameters L1', 'Grad wrt model parameters L2',
                                         'Grad wrt model parameters Linf', 'Grad wrt model parameters Mean',
                                         'Grad wrt model parameters Skewness', 'Grad wrt model parameters Kurtosis',
                                         'Grad wrt model parameters Abs Min']

    aux_list_metrics = []

    criteria0 = scores0[features_used_for_Rezaei_attacker].values
    criteria1 = scores1[features_used_for_Rezaei_attacker].values

    for i in range(num_runs_for_random):
        # Setting Random Seed
        np.random.seed(i)

        # Evaluation Set
        indx_eval0 = np.random.choice(criteria0.shape[0], size=2000, replace=False)
        indx_eval1 = np.random.choice(criteria1.shape[0], size=10000, replace=False)

        indx_training0 = np.setdiff1d(np.arange(criteria0.shape[0]), indx_eval0)
        indx_training1 = np.setdiff1d(np.arange(criteria1.shape[0]), indx_eval1)

        evalX0 = criteria0[indx_eval0]
        trainX0 = criteria0[indx_training0]

        evalX1 = criteria1[indx_eval1]
        trainX1 = criteria1[indx_training1]

        trainY0 = np.zeros((trainX0.shape[0]))
        trainY1 = np.ones((trainX1.shape[0]))

        trainX = np.concatenate((trainX0, trainX1), 0)
        trainY = np.concatenate((trainY0, trainY1))

        Max = np.max(trainX, axis=0)
        Min = np.min(trainX, axis=0)

        trainX = rescale01(trainX, Max, Min)
        evalX0 = rescale01(evalX0, Max, Min)
        evalX1 = rescale01(evalX1, Max, Min)

        attackModelRezaei = LogisticRegression(penalty='l2', tol=1e-5, random_state=i, solver='saga', max_iter=150)

        attackModelRezaei.fit(trainX, trainY)

        preds0 = attackModelRezaei.predict_proba(evalX0)[:, 1]
        preds1 = attackModelRezaei.predict_proba(evalX1)[:, 1]

        metrics_ = evalBestThreshold(0.5, preds0, preds1)
        aux_list_metrics.append(metrics_)

    metrics = np.stack(aux_list_metrics, 1)
    mean_metrics = np.mean(metrics, 1)
    std_metrics = np.std(metrics, 1)

    new_row = {"Attack Strategy": 'Rezaei Attacker Grad w.r.t. parameters',
               'Best Accuracy': mean_metrics[0], 'Best Accuracy STD': std_metrics[0],
               'FPR': mean_metrics[1], 'FPR STD': std_metrics[1]}

    dfMetricsRezaei = dfMetricsRezaei.append(new_row, ignore_index=True)

    print('Evaluation of Rezaei attacker Grad wrt parameters: done')
    sys.stdout.flush()
    sys.stderr.flush()

    dfMetricsRezaei.to_csv(currdir + '/CompleteResults/RezaeiMetrics_' + model_type + '.csv', index=False)

elif mode == 3:
    # Unbalanced evaluation Set 5:1 Training:Testing

    dfMetricsUnbalanced51 = pd.DataFrame(columns=['Attack Strategy',
                                                  'AUROC',
                                                  'Best Accuracy',
                                                  'FPR at TPR80',
                                                  'FPR at TPR85',
                                                  'FPR at TPR90',
                                                  'FPR at TPR95'])

    FPR = np.linspace(0, 1, num=1001)
    dfTPRUnbalanced51 = pd.DataFrame(FPR, columns=['FPR'])

    for column_name in scores0:
        criteria0 = scores0[[column_name]].values
        criteria1 = scores1[[column_name]].values

        TPR, metrics = computeMetrics(criteria0, criteria1, FPR)

        new_row = {"Attack Strategy": column_name,
                   'AUROC': metrics[0],
                   'Best Accuracy': metrics[1],
                   'FPR at TPR80': metrics[2],
                   'FPR at TPR85': metrics[3],
                   'FPR at TPR90': metrics[4],
                   'FPR at TPR95': metrics[5]}

        dfMetricsUnbalanced51 = dfMetricsUnbalanced51.append(new_row, ignore_index=True)

        dfTPRaux = pd.DataFrame(TPR, columns=[column_name + ' TPR'])
        dfTPRUnbalanced51 = dfTPRUnbalanced51.join(dfTPRaux)

    print('Unbalanced evaluation of all scores: done')
    sys.stdout.flush()
    sys.stderr.flush()

    dfMetricsUnbalanced51.to_csv(currdir + '/CompleteResults/Unbalanced51M_' + model_type + '.csv', index=False)
    dfTPRUnbalanced51.to_csv(currdir + '/CompleteResults/Unbalanced51ROC_' + model_type + '.csv', index=False)

elif mode == 4:
    # Unbalanced evaluation Set 1:5 Training:Testing

    dfMetricsUnbalanced15 = pd.DataFrame(columns=['Attack Strategy',
                                                  'AUROC', 'AUROC STD',
                                                  'Best Accuracy', 'Best Accuracy STD',
                                                  'FPR at TPR80', 'FPR at TPR80 STD',
                                                  'FPR at TPR85', 'FPR at TPR85 STD',
                                                  'FPR at TPR90', 'FPR at TPR90 STD',
                                                  'FPR at TPR95', 'FPR at TPR95 STD'])

    FPR = np.linspace(0, 1, num=1001)
    dfTPRUnbalanced15 = pd.DataFrame(FPR, columns=['FPR'])

    for column_name in scores0:

        aux_list_metrics = []
        aux_list_TPR = []

        criteria0 = scores0[[column_name]].values
        criteria1 = scores1[[column_name]].values

        for i in range(num_runs_for_random):
            # Setting Random Seed
            np.random.seed(i)

            # Evaluation Set
            indx_eval0 = np.random.choice(criteria0.shape[0], size=10000, replace=False)
            indx_eval1 = np.random.choice(criteria1.shape[0], size=2000, replace=False)

            criteria0_eval = criteria0[indx_eval0]
            criteria1_eval = criteria1[indx_eval1]

            TPR_, metrics_ = computeMetrics(criteria0_eval, criteria1_eval, FPR)
            aux_list_metrics.append(metrics_)
            aux_list_TPR.append(TPR_)

        metrics = np.stack(aux_list_metrics, 1)
        mean_metrics = np.mean(metrics, 1)
        std_metrics = np.std(metrics, 1)

        new_row = {"Attack Strategy": column_name,
                   'AUROC': mean_metrics[0], 'AUROC STD': std_metrics[0],
                   'Best Accuracy': mean_metrics[1], 'Best Accuracy STD': std_metrics[1],
                   'FPR at TPR80': mean_metrics[2], 'FPR at TPR80 STD': std_metrics[2],
                   'FPR at TPR85': mean_metrics[3], 'FPR at TPR85 STD': std_metrics[3],
                   'FPR at TPR90': mean_metrics[4], 'FPR at TPR90 STD': std_metrics[4],
                   'FPR at TPR95': mean_metrics[5], 'FPR at TPR95 STD': std_metrics[5]}

        dfMetricsUnbalanced15 = dfMetricsUnbalanced15.append(new_row, ignore_index=True)

        TPR = np.stack(aux_list_TPR, 1)
        mean_TPR = np.mean(TPR, 1)
        std_TPR = np.std(TPR, 1)

        dfTPRaux = pd.DataFrame(np.stack((mean_TPR, std_TPR), axis=1),
                                columns=[column_name + ' TPR mean', column_name + ' TPR std'])
        dfTPRUnbalanced15 = dfTPRUnbalanced15.join(dfTPRaux)

    print('Unbalanced evaluation of all scores: done')
    sys.stdout.flush()
    sys.stderr.flush()

    dfMetricsUnbalanced15.to_csv(currdir + '/CompleteResults/Unbalanced15M_' + model_type + '.csv', index=False)
    dfTPRUnbalanced15.to_csv(currdir + '/CompleteResults/Unbalanced15ROC_' + model_type + '.csv', index=False)
