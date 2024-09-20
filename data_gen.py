import argparse
import os
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_boston, load_diabetes, load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data_dir')
parser.add_argument('--dataset', type=str, default='Diabetes')
parser.add_argument('--train_proportion', type=float, default=0.8)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--behavior_bias_quantile', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=0)
parser.add_argument('--ablation', type=int, default=0)
parser.add_argument('--gpu_idx', type=int, default=0)
args = parser.parse_args()


def digits_loader():
    X = load_digits().data
    Y = load_digits().target
    return X, Y


def wine_loader():
    X = load_wine().data
    Y = load_wine().target
    return X, Y


def breast_cancer_loader():
    X = load_breast_cancer().data
    Y = load_breast_cancer().target
    return X, Y


def boston_loader():
    X = load_boston().data
    Y = load_boston().target
    return X, Y


def diabetes_loader():
    X = load_diabetes().data
    Y = load_diabetes().target
    return X, Y

if args.dataset == 'Digits':
    task_type = 'classification'
    X, Y = digits_loader()

elif args.dataset == 'Wine':
    task_type = 'classification'
    X, Y = wine_loader()

elif args.dataset == 'Breast_cancer':
    task_type = 'classification'
    X, Y = breast_cancer_loader()

elif args.dataset == 'Boston':
    task_type = 'regression'
    X, Y = boston_loader()

elif args.dataset == 'Diabetes':
    task_type = 'regression'
    X, Y = diabetes_loader()

else:
    raise ValueError('Dataset not found')


if __name__ == '__main__':
    alias = f'DCall_{args.dataset}_noise{args.noise}_bias{args.behavior_bias_quantile}_ablation{args.ablation}_repeat{args.repeat}'
    scaler = StandardScaler()

    XY = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
    np.random.shuffle(XY)
    X = scaler.fit_transform(XY[:, :-1])
    if task_type == 'regression':
        # 0-1 normalize the regression target
        Y = (XY[:, -1] - np.min(XY[:, -1])) / (np.max(XY[:, -1]) - np.min(XY[:, -1]))
    else:
        Y = XY[:, -1]
    if task_type == 'regression':
        Noisy_Y = Y + np.random.normal(0, args.noise, size=Y.shape)
    elif task_type == 'classification':
        sampled_bn_train = np.random.binomial(1, args.noise, Y.shape)
        Noisy_Y = Y * (1-sampled_bn_train) + np.random.randint(0, len(set(Y)), Y.shape) * sampled_bn_train

    print('dataset size,', X.shape)
    ndata = X.shape[0]

    if args.behavior_bias_quantile < 1.0:
        if args.ablation == 0:
            if args.dataset == 'Breast_cancer':
                thres = np.quantile(X[:, 27], args.behavior_bias_quantile)
                pi_b_training_idx = X[:, 27] < thres
            elif args.dataset == 'Diabetes':
                thres = np.quantile(X[:, 8], args.behavior_bias_quantile)
                pi_b_training_idx = X[:, 8] < thres
            elif args.dataset == 'Boston':
                thres = np.quantile(X[:, 12], args.behavior_bias_quantile)
                pi_b_training_idx = X[:, 12] < thres
            elif args.dataset == 'Wine':
                thres = np.quantile(X[:, 6], args.behavior_bias_quantile)
                pi_b_training_idx = X[:, 6] < thres
            elif args.dataset == 'Digits':
                thres = np.quantile(X[:, 52], args.behavior_bias_quantile)
                pi_b_training_idx = X[:, 52] < thres
            pi_b_training_idx = pi_b_training_idx & (np.random.binomial(1, 0.5, size=pi_b_training_idx.shape) == 1)
            X_pi_b = X[pi_b_training_idx]
            Y_pi_b = Noisy_Y[pi_b_training_idx]
        else:
            if args.dataset == 'Breast_cancer':
                thres = np.quantile(X[:, 27], 0.5)
                pi_b_training_idx = X[:, 27] < thres
            elif args.dataset == 'Diabetes':
                thres = np.quantile(X[:, 8], 0.5)
                pi_b_training_idx = X[:, 8] < thres
            elif args.dataset == 'Boston':
                thres = np.quantile(X[:, 12], 0.5)
                pi_b_training_idx = X[:, 12] < thres
            elif args.dataset == 'Wine':
                thres = np.quantile(X[:, 6], 0.5)
                pi_b_training_idx = X[:, 6] < thres
            elif args.dataset == 'Digits':
                thres = np.quantile(X[:, 52], 0.5)
                pi_b_training_idx = X[:, 52] < thres

            pi_b_training_idx = pi_b_training_idx & (np.random.binomial(1, args.behavior_bias_quantile, size=pi_b_training_idx.shape) == 1)
            X_pi_b = X[pi_b_training_idx]
            Y_pi_b = Noisy_Y[pi_b_training_idx]
    else:
        X_pi_b = X
        Y_pi_b = Noisy_Y

    ndata_pi_b = X_pi_b.shape[0]

    '''building behavior models pi_b'''
    # linear regression/ classification
    if task_type == 'regression':
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        pi_b = LinearRegression()

        pi_b.fit(X_pi_b[:int(ndata_pi_b * args.train_proportion)], Y_pi_b[:int(ndata_pi_b * args.train_proportion)])
        pi_e = MLPRegressor(max_iter=2000)
        pi_e.fit(X[:int(ndata * args.train_proportion)], Noisy_Y[:int(ndata * args.train_proportion)])
    elif task_type == 'classification':
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.neural_network import MLPClassifier
        pi_b = LogisticRegression()
        pi_b.fit(X_pi_b[:int(ndata_pi_b * args.train_proportion)], Y_pi_b[:int(ndata_pi_b * args.train_proportion)])

        pi_e = MLPClassifier(max_iter=2000)
        pi_e.fit(X[:int(ndata * args.train_proportion)], Noisy_Y[:int(ndata * args.train_proportion)])

    pi_b_train_eval = pi_b.score(X[:int(ndata * args.train_proportion)], Y[:int(ndata * args.train_proportion)])
    pi_b_test_eval = pi_b.score(X[int(ndata * args.train_proportion):], Y[int(ndata * args.train_proportion):])
    pi_e_test_eval = pi_e.score(X[int(ndata * args.train_proportion):], Y[int(ndata * args.train_proportion):])
    pi_e_train_eval = pi_e.score(X[:int(ndata * args.train_proportion)], Y[:int(ndata * args.train_proportion)])

    print('pi_b training set score', pi_b_train_eval)
    print('pi_b test set score', pi_b_test_eval)
    print('pi_e training set score', pi_e_train_eval)
    print('pi_e test set score', pi_e_test_eval)

    # generate (x,a,r^a) tuples
    # x: input
    # a: action
    # r^a: reward for action a

    x_train = X[:int(ndata * args.train_proportion)]
    a_train = pi_b.predict(x_train)
    x_test = X[int(ndata * args.train_proportion):]
    a_test = pi_b.predict(x_test)  # those actions are from the evaluation policy pi_e
    #  save data
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if task_type == 'regression':
        u_train = (Y[:int(ndata * args.train_proportion)] - a_train) ** 2
        v_train = (Y[:int(ndata * args.train_proportion)] - Y[:int(ndata * args.train_proportion)].mean()) ** 2
        r_train = - u_train/v_train.sum()

        u_test = (Y[int(ndata * args.train_proportion):] - a_test) ** 2
        v_test = (Y[int(ndata * args.train_proportion):] - Y[int(ndata * args.train_proportion):].mean()) ** 2
        r_test = - u_test/v_test.sum()

        print('avg performance (denoised training set of pi_b)', 1 + r_train.sum())
        print('avg performance (denoised held-out for pi_b)', 1 + r_test.sum())
        print('u train, u test', -u_train.mean(), -u_test.mean())
        np.save(os.path.join(args.data_dir, f'{alias}_x_a_ra.npy'), {'x_train': x_train, 'a_train': a_train, 'r_train': r_train, 'minus_mse': -u_train})
        np.save(os.path.join(args.data_dir, f'{alias}_x_a_ra_test.npy'), {'x_test': x_test, 'a_test': a_test, 'r_test': r_test, 'minus_mse': -u_test})
    elif task_type == 'classification':
        r_train = (Y[:int(ndata * args.train_proportion)] == a_train).astype(float)
        r_test = (Y[int(ndata * args.train_proportion):] == a_test).astype(float)
        print('avg performance (denoised training set of pi_b)', r_train.mean())
        print('avg performance (denoised held-out for pi_b)', r_test.mean())
        np.save(os.path.join(args.data_dir, f'{alias}_x_a_ra.npy'), {'x_train': x_train, 'a_train': a_train, 'r_train': r_train})
        np.save(os.path.join(args.data_dir, f'{alias}_x_a_ra_test.npy'), {'x_test': x_test, 'a_test': a_test, 'r_test': r_test})

    np.save(os.path.join(args.data_dir, f'{alias}_raw_data_train.npy'), {'X': X[:int(ndata * args.train_proportion)],
            'Y': Y[:int(ndata * args.train_proportion)], 'Noisy_Y': Noisy_Y[:int(ndata * args.train_proportion)]})
    np.save(os.path.join(args.data_dir, f'{alias}_raw_data_test.npy'), {'X': X[int(ndata * args.train_proportion):],
            'Y': Y[int(ndata * args.train_proportion):], 'Noisy_Y': Noisy_Y[int(ndata * args.train_proportion):]})
    os.makedirs('saved_models', exist_ok=True)
    pickle.dump(pi_e, open(f'saved_models/{alias}.sav', 'wb'))
    pickle.dump(pi_b, open(f'saved_models/{alias}_pib.sav', 'wb'))
    np.save(os.path.join(args.data_dir, f'{alias}_pi_e_eval.npy'), {'train_eval': pi_e_train_eval, 'test_eval': pi_e_test_eval})
    np.save(os.path.join(args.data_dir, f'{alias}_pi_b_eval.npy'), {'train_eval': pi_b_train_eval, 'test_eval': pi_b_test_eval})
