'''
off policy evaluation algorithms
- workflow:
    - input:
        - policy
        - dataset: consists both training and testing
            (D_train, D_test) = (x_train, a_train, ra_train, x_test, a_test, ra_test)
    - output:
        - estimated policy value on training set and test set.
        - real policy value on training dataset/ test dataset

'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def log_prob_gaussian(selected_action, mu, log_std=0.01):
    action_mean = mu
    std = np.exp(log_std)
    log_prob = -0.5 * ((selected_action - action_mean) / std)**2 - log_std - np.log(np.sqrt(2 * np.pi))
    return log_prob


class NeuralNetReg(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetReg, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.logstd = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_mean = self.fc3(x)
        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def sample(self, x, return_logprob=True):
        action_mean, action_logstd = self.forward(x)
        action_std = action_logstd.exp()
        action = torch.normal(action_mean, action_std)
        if return_logprob:
            log_prob = self.log_prob(x, action)
            return action, log_prob
        return action

    def log_prob(self, x, action):
        action_mean, action_logstd = self.forward(x)
        action_std = action_logstd.exp()
        log_prob = -0.5 * ((action - action_mean) / action_std).pow(2) - action_logstd - np.log(np.sqrt(2 * np.pi))
        return log_prob.sum(1)


'''
direct method

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[q_i]
where q_i is the estimated Q value of state x_i and action a_i
'''


def direct_method_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test):
    # build classification model for reward prediction
    train_x = np.concatenate((x_train, a_train.reshape(-1, 1)), axis=1)
    test_x = np.concatenate((x_test, a_test.reshape(-1, 1)), axis=1)
    train_y = ra_train
    test_y = ra_test
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(64, 64), max_iter=1000)
    clf.fit(train_x, train_y)
    a_train_pi_e = pi_e.predict(x_train)
    a_test_pi_e = pi_e.predict(x_test)

    value_est_train_raw = clf.predict(np.concatenate((x_train, a_train_pi_e.reshape(-1, 1)), axis=1))
    value_est_test_raw = clf.predict(np.concatenate((x_test, a_test_pi_e.reshape(-1, 1)), axis=1))
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
replay method

math: \hat{V}(\pi_e, \mathcal{D}) =
\frac{\mathbb{E} \mathbb{I}(pi_e(x_i) = a_i)*r_i}{\mathbb{I}(pi_e(x_i) = a_i)}
'''


def continuous_approx_equal(var_a, var_b, thres=1):
    return (var_a - var_b)**2 <= thres


def replay_method_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test):
    value_est_train_raw = (continuous_approx_equal(pi_e.predict(x_train), a_train) * ra_train) / np.mean(continuous_approx_equal(pi_e.predict(x_train), a_train))
    value_est_test_raw = (continuous_approx_equal(pi_e.predict(x_test), a_test) * ra_test) / np.mean(continuous_approx_equal(pi_e.predict(x_test), a_test))
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
inverse probability weighting

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[w(x_i, a_i)*r_i],
where w(x_i, a_i) = \frac{\pi_e(x_i, a_i)}{\pi_b(x_i, a_i)}
'''


def inverse_probability_weighting_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test):
    from sklearn.neural_network import MLPRegressor

    '''
    gaussian regression model
    '''
    x_train_torch = torch.from_numpy(x_train).float().cuda()
    y_train_torch = torch.from_numpy(a_train).float().cuda()
    model = NeuralNetReg(x_train_torch.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        loss = -model.log_prob(x_train_torch, y_train_torch).mean()
        loss.backward()
        optimizer.step()

    pi_b_proba_train = model.log_prob(x_train_torch, y_train_torch).cpu().detach().numpy()
    pi_b_proba_test = model.log_prob(torch.as_tensor(x_test).float().cuda(), torch.as_tensor(a_test).float().cuda()).cpu().detach().numpy()
    weight_train = continuous_approx_equal(pi_e.predict(x_train), a_train)/np.exp(pi_b_proba_train.clip(-1, 0.5))
    weight_test = continuous_approx_equal(pi_e.predict(x_test), a_test)/np.exp(pi_b_proba_test.clip(-1, 0.5))
    # weight_train = np.exp(log_prob_gaussian(pi_e.predict(x_train), a_train, 1.0).clip(-1000, 1000)-(pi_b_proba_train))
    # weight_test = np.exp(log_prob_gaussian(pi_e.predict(x_test), a_test, 1.0).clip(-1000, 1000)-(pi_b_proba_test))
    print(weight_train.mean(), weight_test.mean())
    value_est_train_raw = weight_train * ra_train
    value_est_test_raw = weight_test * ra_test
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
self-normalized inverse probability weighting

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[w(x_i, a_i)*r_i / \mathbb{w(x_i, a_i)}],
where w(x_i, a_i) = \frac{\pi_e(x_i, a_i)}{\pi_b(x_i, a_i)}
'''


def sn_inverse_probability_weighting_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test):
    from sklearn.neural_network import MLPRegressor

    '''
    gaussian regression model
    '''
    x_train_torch = torch.from_numpy(x_train).float().cuda()
    y_train_torch = torch.from_numpy(a_train).float().cuda()
    model = NeuralNetReg(x_train_torch.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        loss = -model.log_prob(x_train_torch, y_train_torch).mean()
        loss.backward()
        optimizer.step()

    pi_b_proba_train = model.log_prob(x_train_torch, y_train_torch).cpu().detach().numpy()
    pi_b_proba_test = model.log_prob(torch.as_tensor(x_test).float().cuda(), torch.as_tensor(a_test).float().cuda()).cpu().detach().numpy()
    # weight_train = continuous_approx_equal(pi_e.predict(x_train), a_train)/np.exp(pi_b_proba_train.clip(-100, 0))
    # weight_test = continuous_approx_equal(pi_e.predict(x_test), a_test)/np.exp(pi_b_proba_test.clip(-100, 0))

    weight_train = np.exp(log_prob_gaussian(pi_e.predict(x_train), a_train)-(pi_b_proba_train))
    weight_test = np.exp(log_prob_gaussian(pi_e.predict(x_test), a_test)-(pi_b_proba_test))

    value_est_train_raw = weight_train * ra_train / (np.mean(weight_train) + 1e-7)
    value_est_test_raw = weight_test * ra_test / (np.mean(weight_test) + 1e-7)
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
inverse probability weighting - lambda

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[\mathrm{Clip}(w(x_i, a_i), -1.0, 1.0)*r_i],
where w(x_i, a_i) = \frac{\pi_e(x_i, a_i)}{\pi_b(x_i, a_i)}
'''


def inverse_probability_weighting_lamda_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test, lamda=0.2):
    from sklearn.neural_network import MLPRegressor

    '''
    gaussian regression model
    '''
    x_train_torch = torch.from_numpy(x_train).float().cuda()
    y_train_torch = torch.from_numpy(a_train).float().cuda()
    model = NeuralNetReg(x_train_torch.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        loss = -model.log_prob(x_train_torch, y_train_torch).mean()
        loss.backward()
        optimizer.step()

    pi_b_proba_train = model.log_prob(x_train_torch, y_train_torch).cpu().detach().numpy()
    pi_b_proba_test = model.log_prob(torch.as_tensor(x_test).float().cuda(), torch.as_tensor(a_test).float().cuda()).cpu().detach().numpy()
    # weight_train = continuous_approx_equal(pi_e.predict(x_train), a_train)/np.exp(pi_b_proba_train.clip(-100, 0))
    # weight_test = continuous_approx_equal(pi_e.predict(x_test), a_test)/np.exp(pi_b_proba_test.clip(-100, 0))
    weight_train = np.exp(log_prob_gaussian(pi_e.predict(x_train), a_train)-(pi_b_proba_train))
    weight_test = np.exp(log_prob_gaussian(pi_e.predict(x_test), a_test)-(pi_b_proba_test))

    weight_train = np.clip(weight_train, 1-lamda, 1+lamda)
    weight_test = np.clip(weight_test, 1-lamda, 1+lamda)

    value_est_train_raw = weight_train * ra_train
    value_est_test_raw = weight_test * ra_test
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
doubly robust

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[q(x_i, \pi_e(x_i)) + w(x_i, a_i)(r_i - q(x_i, \pi_e(x_i)))]
'''


def doubly_robust_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test, lamda=0.5):
    # build classification model for reward prediction
    train_x = np.concatenate((x_train, a_train.reshape(-1, 1)), axis=1)
    test_x = np.concatenate((x_test, a_test.reshape(-1, 1)), axis=1)
    train_y = ra_train
    test_y = ra_test
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(64, 64), max_iter=1000)
    clf.fit(train_x, train_y)
    a_train_pi_e = pi_e.predict(x_train)
    a_test_pi_e = pi_e.predict(x_test)
    q_est_train = clf.predict(np.concatenate((x_train, a_train_pi_e.reshape(-1, 1)), axis=1))  # q(x_i, \pi_e(x_i))
    q_est_test = clf.predict(np.concatenate((x_test, a_test_pi_e.reshape(-1, 1)), axis=1))

    # build propensity estimator to calculate w(x_i, a_i)
    from sklearn.neural_network import MLPRegressor

    '''
    gaussian regression model
    '''
    x_train_torch = torch.from_numpy(x_train).float().cuda()
    y_train_torch = torch.from_numpy(a_train).float().cuda()
    model = NeuralNetReg(x_train_torch.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        loss = -model.log_prob(x_train_torch, y_train_torch).mean()
        loss.backward()
        optimizer.step()

    pi_b_proba_train = model.log_prob(x_train_torch, y_train_torch).cpu().detach().numpy()
    pi_b_proba_test = model.log_prob(torch.as_tensor(x_test).float().cuda(), torch.as_tensor(a_test).float().cuda()).cpu().detach().numpy()
    # weight_train = continuous_approx_equal(pi_e.predict(x_train), a_train)/np.exp(pi_b_proba_train.clip(-100, 0))
    # weight_test = continuous_approx_equal(pi_e.predict(x_test), a_test)/np.exp(pi_b_proba_test.clip(-100, 0))
    weight_train = np.exp(log_prob_gaussian(pi_e.predict(x_train), a_train)-(pi_b_proba_train))
    weight_test = np.exp(log_prob_gaussian(pi_e.predict(x_test), a_test)-(pi_b_proba_test))

    weight_train = np.clip(weight_train, 1-lamda, 1+lamda)
    weight_test = np.clip(weight_test, 1-lamda, 1+lamda)

    value_est_train_raw = q_est_train + weight_train * (ra_train - q_est_train)
    value_est_test_raw = q_est_test + weight_test * (ra_test - q_est_test)
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw


'''
self-normalized doubly robust

math: \hat{V}(\pi_e, \mathcal{D}) = \mathbb{E}[q(x_i, \pi_e(x_i)) + \frac{w(x_i, a_i)(r_i - q(x_i, \pi_e(x_i)))}{\mathbb{E}[w(x_i, a_i)]}]
'''


def sn_doubly_robust_mse(pi_e, x_train, a_train, ra_train, x_test, a_test, ra_test, lamda=0.2):
    # build classification model for reward prediction
    train_x = np.concatenate((x_train, a_train.reshape(-1, 1)), axis=1)
    test_x = np.concatenate((x_test, a_test.reshape(-1, 1)), axis=1)
    train_y = ra_train
    test_y = ra_test
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(64, 64), max_iter=1000)
    clf.fit(train_x, train_y)
    a_train_pi_e = pi_e.predict(x_train)
    a_test_pi_e = pi_e.predict(x_test)
    q_est_train = clf.predict(np.concatenate((x_train, a_train_pi_e.reshape(-1, 1)), axis=1))  # q(x_i, \pi_e(x_i))
    q_est_test = clf.predict(np.concatenate((x_test, a_test_pi_e.reshape(-1, 1)), axis=1))

    # build propensity estimator to calculate w(x_i, a_i)
    from sklearn.neural_network import MLPRegressor

    '''
    gaussian regression model
    '''
    x_train_torch = torch.from_numpy(x_train).float().cuda()
    y_train_torch = torch.from_numpy(a_train).float().cuda()
    model = NeuralNetReg(x_train_torch.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(2000):
        optimizer.zero_grad()
        loss = -model.log_prob(x_train_torch, y_train_torch).mean()
        loss.backward()
        optimizer.step()

    pi_b_proba_train = model.log_prob(x_train_torch, y_train_torch).cpu().detach().numpy()
    pi_b_proba_test = model.log_prob(torch.as_tensor(x_test).float().cuda(), torch.as_tensor(a_test).float().cuda()).cpu().detach().numpy()
    # weight_train = continuous_approx_equal(pi_e.predict(x_train), a_train)/np.exp(pi_b_proba_train.clip(-100, 0))
    # weight_test = continuous_approx_equal(pi_e.predict(x_test), a_test)/np.exp(pi_b_proba_test.clip(-100, 0))
    weight_train = np.exp(log_prob_gaussian(pi_e.predict(x_train), a_train)-(pi_b_proba_train))
    weight_test = np.exp(log_prob_gaussian(pi_e.predict(x_test), a_test)-(pi_b_proba_test))

    weight_train = np.clip(weight_train, 1-lamda, 1+lamda)
    weight_test = np.clip(weight_test, 1-lamda, 1+lamda)
    print(np.mean(weight_train), np.mean(weight_test), (ra_train - q_est_train).mean())
    value_est_train_raw = q_est_train + weight_train * (ra_train - q_est_train) / np.mean(weight_train)
    value_est_test_raw = q_est_test + weight_test * (ra_test - q_est_test) / np.mean(weight_test)
    value_est_train = np.mean(value_est_train_raw)
    value_est_test = np.mean(value_est_test_raw)
    return value_est_train, value_est_test, value_est_train_raw, value_est_test_raw
