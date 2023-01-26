from sklearn.neural_network import MLPClassifier
import pickle
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math
import seaborn as sns
import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from copy import deepcopy
from random import sample
import mdn
import torch.nn as nn
import torch.optim as optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Diabetes')
parser.add_argument('--noise', type=float, default=0.01)
parser.add_argument('--NoG', type=int, default=3)
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--n_ensemble', type=int, default=100)
parser.add_argument('--behavior_bias_quantile', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--ra_mode', type=str, default='mse')
parser.add_argument('--eval_mode', type=str, default='pie')
parser.add_argument('--ablation', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_idx)


class uncertainty_decomposition():
    def __init__(self, X, test_X):
        self.X = X
        self.test_X = test_X
        self.predict_probs = []
        self.predict_probs_test = []

    def on_model_end(self, model):
        probabilities = model.predict_proba(self.X)
        # shape of probabilities: (n_samples, n_classes = 2: bool)
        self.predict_probs.append(probabilities)

    def on_model_end_test(self, model):
        probabilities = model.predict_proba(self.test_X)
        # shape of probabilities: (n_samples, n_classes = 2: bool)
        self.predict_probs_test.append(probabilities)

    def get_uncertainty(self):
        self.v_ep = np.var(np.asarray(self.predict_probs)[:, :, 1], axis=0)
        assert self.v_ep.shape == (self.X.shape[0],)
        self.v_al = np.mean(np.asarray(self.predict_probs)[:, :, 0] * np.asarray(self.predict_probs)[:, :, 1], axis=0)
        assert self.v_al.shape == (self.X.shape[0],)
        return self.v_ep, self.v_al

    def get_uncertainty_test(self):
        self.v_ep_test = np.var(np.asarray(self.predict_probs_test)[:, :, 1], axis=0)
        assert self.v_ep_test.shape == (self.test_X.shape[0],)
        self.v_al_test = np.mean(np.asarray(self.predict_probs_test)[:, :, 0]*np.asarray(self.predict_probs_test)[:, :, 1], axis=0)
        assert self.v_al_test.shape == (self.test_X.shape[0],)
        return self.v_ep_test, self.v_al_test


class uncertainty_decomposition_reg():
    def __init__(self, X, test_X):
        self.X = X
        self.test_X = test_X
        self.predict_pis = []
        self.predict_sigmas = []
        self.predict_mus = []
        self.predict_pis_test = []
        self.predict_sigmas_test = []
        self.predict_mus_test = []

    def on_model_end(self, model):
        pi, sigma, mu = model(torch.as_tensor(self.X).float().cuda())
        # shape of pi: (n_samples, NoG)
        # shape of sigma: (n_samples, NoG, 1)
        # shape of mu: (n_samples, NoG, 1), here 1 is dertermined by the output dimension
        pi = pi.cpu().detach().numpy()
        sigma = sigma.cpu().detach().squeeze(-1).numpy()
        mu = mu.cpu().detach().squeeze(-1).numpy()
        self.predict_pis.append(pi)
        self.predict_sigmas.append(sigma)
        self.predict_mus.append(mu)

    def on_model_end_test(self, model):
        pi, sigma, mu = model(torch.as_tensor(self.test_X).float().cuda())
        pi = pi.cpu().detach().numpy()
        sigma = sigma.cpu().detach().squeeze(-1).numpy()
        mu = mu.cpu().detach().squeeze(-1).numpy()
        self.predict_pis_test.append(pi)
        self.predict_sigmas_test.append(sigma)
        self.predict_mus_test.append(mu)

    def get_uncertainty(self):
        self.v_ep = np.var((np.asarray(self.predict_pis) * np.asarray(self.predict_mus)).mean(-1), axis=0)
        assert self.v_ep.shape == (self.X.shape[0],)
        # ref: https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        self.v_al = np.mean(
            (np.asarray(self.predict_pis) * np.asarray(self.predict_sigmas)**2).sum(-1) +
            (np.asarray(self.predict_pis) * np.asarray(self.predict_mus)**2).sum(-1) -
            ((np.asarray(self.predict_pis) * np.asarray(self.predict_mus)).sum(-1))**2, axis=0
        )
        assert self.v_al.shape == (self.X.shape[0],)
        return self.v_ep, self.v_al

    def get_uncertainty_test(self):
        self.v_ep_test = np.var((np.asarray(self.predict_pis_test) * np.asarray(self.predict_mus_test)).mean(-1), axis=0)
        assert self.v_ep_test.shape == (self.test_X.shape[0],)
        self.v_al_test = np.mean(
            (np.asarray(self.predict_pis_test) * np.asarray(self.predict_sigmas_test)**2).sum(-1) +
            (np.asarray(self.predict_pis_test) * np.asarray(self.predict_mus_test)**2).sum(-1) -
            ((np.asarray(self.predict_pis_test) * np.asarray(self.predict_mus_test)).sum(-1))**2, axis=0
        )
        assert self.v_al_test.shape == (self.test_X.shape[0],)
        return self.v_ep_test, self.v_al_test


alias = f'DCall_{args.dataset}_noise{args.noise}_bias{args.behavior_bias_quantile}_ablation{args.ablation}_repeat{args.repeat}'
alias_save = alias + f'_{args.eval_mode}_ramode{args.ra_mode}_repeat{args.repeat}'
train_dataset = np.load(f'data_dir/{alias}_x_a_ra.npy', allow_pickle=True).item()
test_dataset = np.load(f'data_dir/{alias}_x_a_ra_test.npy', allow_pickle=True).item()
train_oracle = np.load(f'data_dir/{alias}_raw_data_train.npy', allow_pickle=True).item()
test_oracle = np.load(f'data_dir/{alias}_raw_data_test.npy', allow_pickle=True).item()
if args.eval_mode == 'pie':
    pi_e = pickle.load(open(f'saved_models/{alias}.sav', 'rb'))
elif args.eval_mode == 'pib':
    pi_e = pickle.load(open(f'saved_models/{alias}_pib.sav', 'rb'))

if args.dataset in ['Boston', 'Diabetes']:
    task_type = 'regression'
elif args.dataset in ['Digits', 'Wine', 'Breast_cancer']:
    task_type = 'classification'
else:
    raise NotImplementedError

oracle_X, oracle_Y, oracle_Noisy_Y = train_oracle['X'], train_oracle['Y'], train_oracle['Noisy_Y']
oracle_X_test, oracle_Y_test, oracle_Noisy_Y_test = test_oracle['X'], test_oracle['Y'], test_oracle['Noisy_Y']
x_train, a_train, ra_train, mmse_train = train_dataset['x_train'], train_dataset['a_train'], train_dataset['r_train'], train_dataset['minus_mse']
x_test, a_test, ra_test, mmse_test = test_dataset['x_test'], test_dataset['a_test'], test_dataset['r_test'], test_dataset['minus_mse']
train_x = np.concatenate((x_train, a_train.reshape(-1, 1)), axis=1)
eval_x_train = np.concatenate((x_train, pi_e.predict(x_train).reshape(-1, 1)), axis=1)
# test_x = np.concatenate((x_test, a_test.reshape(-1, 1)), axis=1)
eval_x_test = np.concatenate((x_test, pi_e.predict(x_test).reshape(-1, 1)), axis=1)

# test_y = ra_test

if args.ra_mode == 'mse':
    ra_train = mmse_train
    ra_test = mmse_test

train_y = ra_train
hardness_training_set = []
hardness_test_set = []


ope_residual_train_dict = {}
ope_residual_test_dict = {}
for ope_method in ['ipw', 'dm', 'dr', 'rm', 'snipw', 'ipwl', 'sndr']:
    if args.ra_mode == 'mse':
        if ope_method == 'ipw':
            from ope_algos_reg import inverse_probability_weighting_mse as IPW
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = IPW(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'dm':
            from ope_algos_reg import direct_method_mse as DM
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = DM(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'dr':
            from ope_algos_reg import doubly_robust_mse as DR
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = DR(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'rm':
            from ope_algos_reg import replay_method_mse as RM
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = RM(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'snipw':
            from ope_algos_reg import sn_inverse_probability_weighting_mse as SNIPW
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = SNIPW(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'ipwl':
            from ope_algos_reg import inverse_probability_weighting_lamda_mse as IPWL
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = IPWL(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'sndr':
            from ope_algos_reg import sn_doubly_robust_mse as SNDR
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = SNDR(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        else:
            raise NotImplementedError
    else:
        if ope_method == 'ipw':
            from ope_algos_reg import inverse_probability_weighting as IPW
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = IPW(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'dm':
            from ope_algos_reg import direct_method as DM
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = DM(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'dr':
            from ope_algos_reg import doubly_robust as DR
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = DR(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'rm':
            from ope_algos_reg import replay_method as RM
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = RM(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'snipw':
            from ope_algos_reg import sn_inverse_probability_weighting as SNIPW
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = SNIPW(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'ipwl':
            from ope_algos_reg import inverse_probability_weighting_lamda as IPWL
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = IPWL(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        elif ope_method == 'sndr':
            from ope_algos_reg import sn_doubly_robust as SNDR
            value_est_train, value_est_test, value_est_train_raw, value_est_test_raw = SNDR(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test)
        else:
            raise NotImplementedError
    ope_residual_train = (-(pi_e.predict(x_train) - oracle_Y)**2 - value_est_train_raw)**2
    ope_residual_test = (-(pi_e.predict(x_test) - oracle_Y_test)**2 - value_est_test_raw)**2

    ope_residual_train_dict[ope_method] = ope_residual_train
    ope_residual_test_dict[ope_method] = ope_residual_test

if task_type == 'classification':
    datamaps = uncertainty_decomposition(X=eval_x_train, test_X=eval_x_test)
    for i in tqdm(range(args.n_ensemble)):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(64, 64), max_iter=1000)
        clf.fit(train_x, train_y)
        datamaps.on_model_end(model=clf)
        datamaps.on_model_end_test(model=clf)
else:
    datamaps = uncertainty_decomposition_reg(X=eval_x_train, test_X=eval_x_test)
    for i in tqdm(range(args.n_ensemble)):
        reg = nn.Sequential(
            nn.Linear(train_x.shape[1], 64),
            nn.Tanh(),
            mdn.MDN(64, 1, args.NoG)
        ).cuda()
        optimizer = optim.Adam(reg.parameters())
        for epoch in range(100):
            optimizer.zero_grad()
            pi, sigma, mu = reg(torch.as_tensor(train_x).float().cuda())
            loss = mdn.mdn_loss(pi, sigma, mu, torch.as_tensor(train_y.reshape(-1, 1)).cuda())
            loss.backward()
            optimizer.step()
        datamaps.on_model_end(model=reg)
        datamaps.on_model_end_test(model=reg)
datamaps.get_uncertainty()

hardness_training_set = (datamaps.v_ep, datamaps.v_al, ope_residual_train)

np.save(f'saved_results/{alias_save}_datamaps.npy', {
        'v_ep': datamaps.v_ep,
        'v_al': datamaps.v_al,
        'residual_train': ope_residual_train_dict
        })

datamaps.get_uncertainty_test()
hardness_test_set = (datamaps.v_ep_test, datamaps.v_al_test, ope_residual_test)

np.save(f'saved_results/{alias_save}_datamaps_test.npy', {
        'v_ep_test': datamaps.v_ep_test,
        'v_al_test': datamaps.v_al_test,
        'residual_test': ope_residual_test_dict
        })
