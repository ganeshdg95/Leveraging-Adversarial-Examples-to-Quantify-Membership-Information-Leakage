# Leveraging-Adversarial-Examples-to-Quantify-Membership-Information-Leakage

This repository contains the code necessary to reproduce the experiments described in "[Leveraging Adversarial Examples to Quantify Membership Information Leakage](https://arxiv.org/abs/2203.09566)".

## Set up

- Create a virtual environment with the required libraries:

```commandline
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

- Get the pretrained models [here](https://github.com/bearpaw/pytorch-classification). In the example below, the pretrained models are stored in a folder with the following structure:

```commandline
+-- trained_models
|  +-- densenet-bc-L190-k40
|  |  +-- model_best.pth.tar
|  +-- resnext-8x64d
|  |  +-- model_best.pth.tar
|  +-- resnet-110
|  |  +-- model_best.pth.tar
|  +-- alexnet
|  |  +-- model_best.pth.tar

```

## Example Pipeline

- Run Collect.py to compute the scores used to perform Membership inference attacks (detailed explanation below).

```commandline

python Collect.py --seed 0 --model_type 'AlexNet' --output_dir './RawResults' --trained_dir './trained_models'

```

- To evaluate the performance of the different attack strategies on a balanced dataset, run:

```commandline

python Metrics.py --model_type 'AlexNet' --mode 1 --working_dir './'

```

- To perform the analysis from ''[On the Difficulty of Membership Inference Attacks](https://openaccess.thecvf.com/content/CVPR2021/html/Rezaei_On_the_Difficulty_of_Membership_Inference_Attacks_CVPR_2021_paper.html)'', run:

```commandline

python Metrics.py --model_type 'AlexNet' --mode 2 --working_dir './'

python Metrics.py --model_type 'AlexNet' --mode 3 --working_dir './'

```

## Collect.py

A Membership Inference Attack (MIA) can be considered as a binary decision test, where the attacker is trying to determine if a sample belongs to the training set or not. The decision is taken by comparing a score to a threshold. If the score is higher than the threshold then the sample is predicted to be in the training set.

Collect.py computes scores according to different criteria:

- Adversarial Distance (our strategy).
- Model confidence.
- Entropy and Modified Entropy, from "[Systematic evaluation of Privacy Risks of Machine Learning Models](https://www.usenix.org/system/files/sec21fall-song.pdf)".
- Loss value.
- Statistics of the loss gradient w.r.t. model parameters.
- Statistics of the loss gradient w.r.t. the input sample.

Collect.py also computes the information necessary to train the attack models described in ''[On the Difficulty of Membership Inference Attacks](https://openaccess.thecvf.com/content/CVPR2021/html/Rezaei_On_the_Difficulty_of_Membership_Inference_Attacks_CVPR_2021_paper.html)'', "[Comprehensive Privacy Analysis of Deep Learning](https://arxiv.org/abs/1812.00910)". The scores and additional information are computed for every single sample in the dataset and stored separately depending on whether they belong to the training set or not.

## Metrics.py

Metrics.py computes the statistics that indicate the performance of the different attack strategies, e.g. AUROC score, false positive ratio when the true positive ratio 95%, and best accuracy (maximized w.r.t. the threshold). This code has different modes depending on what part of the analysis you wish to compute:

1. Mode 1 computes our analysis, using a balanced evaluation set (the number of training samples is equal to the number of non-training samples). It computes the following statistics, which are independent of the choice of threshold: AUROC score, best accuracy and, false positive rate at different true positive rates. It also computes the ROC curves for each strategy. Some strategies involve training an attack model, this is done using a balanced attacker training set. Then the same analysis is performed with the scores produced by the trained model on the evaluation set.
2. Modes 2 and 3 compute the analysis described in ''[On the Difficulty of Membership Inference Attacks](https://openaccess.thecvf.com/content/CVPR2021/html/Rezaei_On_the_Difficulty_of_Membership_Inference_Attacks_CVPR_2021_paper.html)''. An unbalanced evaluation set is used. 80% of the total dataset is given to the attacker as side information. This side information is used either to train an attack model or to find the best threshold. The accuracy and false positive rate are computed with this threshold.
3. See above.
4. The analysis described for mode 1 is computed on an unbalanced evaluation set, where  5:1 is the ratio of training to testing samples.
5. The analysis described for mode 1 is computed on an unbalanced evaluation set, where  1:5 is the ratio of training to testing samples.

## MetricsShokri.py

This code trains and evaluates the attack model described in "[Comprehensive Privacy Analysis of Deep Learning](https://arxiv.org/abs/1812.00910)".

