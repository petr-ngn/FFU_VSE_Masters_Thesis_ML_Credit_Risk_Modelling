import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import time
import math
import missingno
from itertools import combinations

from scipy.stats import chi2_contingency, ks_2samp, pointbiserialr, somersd
from imblearn.over_sampling import SMOTENC, ADASYN

from optbinning import BinningProcess

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,\
                            brier_score_loss, zero_one_loss, hamming_loss, jaccard_score,\
                            matthews_corrcoef, confusion_matrix, roc_curve, roc_auc_score
 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import shap