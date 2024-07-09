import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import argparse
import pickle5
import missingpy
import os
from sklearn import ensemble, model_selection
from copy import deepcopy
# the following import will break unless you have set your PYTHONPATH as stated in the README
import learning_utils
'''
The script that runs the ML regression random forest algorithm.

This script operates in a few steps:
1. Read in input features + output features
2. Split data into train + test sets
3. On the training set, perform an exhaustive grid search through algorithm
   hyperparameters, using 5-fold cross-validation to score each param combo
4. Using the highest-scoring set of hyperparameters, train algorithm with the
   full training set, and score with the test set
5. Plot learning curves
6. Attempt to quantify performance of "random guessing" algorithm by shuffling
   output classes a bunch of times and looking at scores
   - Do this twice: with test set shuffled on its own, and with train/test sets
     shuffled together
7. Save relevant outputs to a pickle file

Input args:
- wafer (int or None): Which wafer to use for the dataset. If an int, will use a
  single wafer; if None, will use all three wafers.
- exclude_wafer (int or None): If int, will train on all wafers except this one,
  which will be set aside as the test set
- avg_double_bolos (bool): Whether to average the input features for doubly-imaged
  bolometeres
- seed (int): Seed for random number generation
- save_dir (str): Directory in which to save outputs
- retrain (bool): By default, the algorithm is not re-trained when performing the
  shuffling to test "random guessing" performance. Set this flag if you want to
  re-train every time
- shuffle_input_feats (bool): Whether to shuffle the input features along with the
  output features during the test to estimate "random guessing" performance. Only
  interesting in the case where train+test sets are shuffled together.
'''
p = argparse.ArgumentParser()
p.add_argument('-w', '--wafer', default=None)
p.add_argument('--exclude-wafer', default=None)
p.add_argument('--avg-double-bolos', action='store_true', default=False)
p.add_argument('-s', '--seed', type=int, default=0)
p.add_argument('--save-dir', type=str, default='/home/kferguson/')
p.add_argument('-r', '--retrain', action='store_true', default=False)
p.add_argument('--shuffle-input-feats', action='store_true', default=False)
args = p.parse_args()

if args.avg_double_bolos:
    tag = 'averaged'
else:
    tag = 'kept'

all_wafers = [148, 162, 187]
if args.wafer is None:
    if args.exclude_wafer is None:
        fname_out = 'RandomForestRegressor_output_all_wafers_double_bolos_%s.pkl'%(tag)
        wafer = deepcopy(all_wafers)
    else:
        args.exclude_wafer = int(args.exclude_wafer)
        fname_out = 'RandomForestRegressor_output_exclude_w%d_double_bolos_%s.pkl'%(args.exclude_wafer, tag)
        wafer = []
        for w in all_wafers:
            if w != args.exclude_wafer:
                wafer.append(w)
else:
    wafer = int(args.wafer)
    fname_out = 'RandomForestRegressor_output_w%d_double_bolos_%s.pkl'%(wafer, tag)

im_out = 'plots_' + fname_out.split('.')[0]
np.random.seed(args.seed)
out_dict = {}

tree_nums = [10, 50, 100, 500, 1000]
depths = [1, 2, 3, 4, 5, 6]
split_samps = [2, 3, 4]
#max_feats = ['sqrt', 'log2']
max_feats = ['sqrt'] # both give nearly same answer, which is rounded to the exact same int with our 14 input features
max_samples = [None, 0.1, 0.25, 0.5]

features_in, features_out, nums, wafer_labels = learning_utils.get_wafer_characteristics(wafer, impute_input_nans=True,
                                                        impute_output_nans=True, average_double_bolos=args.avg_double_bolos,
                                                        return_wafer_nums=True)
print(features_in.shape)
print(features_out.shape)
if args.wafer is None:
    nums = np.sum(np.array(nums), axis=0)
print('%d passed because no readout name'%(nums[0]))
print('%d passed because no R(T) data'%(nums[1]))
print('%d passed because outlier Rnormal'%(nums[2]))

if np.any(np.isnan(features_in)):
    raise ValueError('%d nans in input after MissForest imputation'%(np.sum(np.isnan(features_in))))

# split data
# first train/test split
if args.exclude_wafer is None:
    s = np.random.randint(low=0, high=2**32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features_in, features_out, test_size=0.2, random_state=s)
    _, _, wafers_train, wafers_test = model_selection.train_test_split(features_in, wafer_labels, test_size=0.2, random_state=s)
else:
    X_train = deepcopy(features_in); y_train = deepcopy(features_out); wafers_train = deepcopy(wafer_labels)
    X_test, y_test, nums2, wafers_test = learning_utils.get_wafer_characteristics(args.exclude_wafer, impute_input_nans=True,
                                                        impute_output_nans=True, average_double_bolos=args.avg_double_bolos,
                                                        return_wafer_nums=True)
    print('test %d passed because no readout name'%(nums2[0]))
    print('test %d passed because no R(T) data'%(nums2[1]))
    print('test %d passed because outlier Rnormal'%(nums2[2]))

# then split train set further for k-fold cross-validation
k = 5
kfold = model_selection.KFold(n_splits=k, shuffle=True, random_state=np.random.randint(low=0, high=2**32))
mean_scores = []
param_combo = []

for n in tree_nums:
    for d in depths:
        for s in split_samps:
            for method in max_feats:
                for m in max_samples:

                    scores = []
                    for train_ix, test_ix in kfold.split(X_train):
                        # select rows
                        train_X, test_X = X_train[train_ix], X_train[test_ix]
                        train_y, test_y = y_train[train_ix], y_train[test_ix]

                        clf = ensemble.RandomForestRegressor(n_estimators = n,
                                                             max_depth = d,
                                                             min_samples_split = s,
                                                             max_features = method,
                                                             max_samples = m,
                                                             random_state=np.random.randint(low=0, high=2**32))
                        clf.fit(train_X, train_y)
                        scores.append(clf.score(test_X, test_y))
                    mean_scores.append(np.mean(scores))
                    param_combo.append((n, d, s, method, m))
        print('d = %s'%d)
    print('n = %s'%n)

out_dict['grid search scores'] = mean_scores
out_dict['hyper params'] = param_combo

n, d, s, method, m = param_combo[np.argmax(mean_scores)]
clf_final = ensemble.RandomForestRegressor(n_estimators = n,
                                           max_depth = d,
                                           min_samples_split = s,
                                           max_features = method,
                                           max_samples = m,
                                           random_state=np.random.randint(low=0, high=2**32))
clf_final.fit(X_train, y_train)
score = clf_final.score(X_test, y_test)
print('final score: %.3f'%(score))
out_dict['final score'] = score
out_dict['RandomForestRegressor'] = clf_final

# recombine train and test sets
if args.exclude_wafer is not None:
    y = deepcopy(y_train)
    y = list(y)
    y.extend(y_test)
    y = np.array(y)
    x = deepcopy(X_train)
    x = list(x)
    x.extend(X_test)
    x = np.array(x)
    test_size = y_test.shape[0]
else:
    x = deepcopy(features_in)
    y = deepcopy(features_out)
    test_size = 0.2

# plot learning curve
plot_dir = args.save_dir + im_out + '/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
clf = ensemble.RandomForestRegressor(
    n_estimators = n, max_depth = d, min_samples_split = s, max_features = method,
    max_samples = m, random_state=np.random.randint(low=0, high=2**32))
cv = model_selection.ShuffleSplit(n_splits=50, test_size=test_size, random_state=np.random.randint(low=0, high=2**32))
fig = learning_utils.plot_learning_curve(
    clf, fname_out.split('.')[0], x, y, cv=cv, n_jobs=4)
plt.savefig(plot_dir + 'learning_curve.png')
plt.close()

# histogram residuals
preds = clf_final.predict(X_test)
feats_out = ['rnormal', 'rpar', 'tc_mid', '1sigup', '1sigdown', 'has nan']

for i in range(y_test.shape[1]):
    plt.figure()
    plt.plot(np.linspace(np.min([np.min(preds[:,i]), np.min(y_test[:,i])]), np.max([np.max(preds[:,i]), np.max(y_test[:,i])])),
             np.linspace(np.min([np.min(preds[:,i]), np.min(y_test[:,i])]), np.max([np.max(preds[:,i]), np.max(y_test[:,i])])), 'k')
    for j, w in enumerate(np.unique(wafers_test)):
        plt.plot(preds[:,i][wafers_test==w], y_test[:,i][wafers_test==w], 'C%d.'%(j), label='W%d'%(int(w)))
    plt.title('%s'%(feats_out[i]))
    plt.xlabel('random forest predictions')
    plt.ylabel('true feature values')
    plt.legend()
    plt.savefig(plot_dir + '%s_prediction_accuracy.png'%(feats_out[i].replace(' ', '_')))
    plt.close()

# shuffle scores, test and train separately
random_scores = []
for i in range(100):
    np.random.shuffle(y_train)
    np.random.shuffle(y_test)
    if args.shuffle_input_feats:
        np.random.shuffle(X_train)
        np.random.shuffle(X_test)
    if args.retrain:
        clf = ensemble.RandomForestRegressor(n_estimators = n,
                                             max_depth = d,
                                             min_samples_split = s,
                                             max_features = method,
                                             max_samples = m,
                                             random_state=np.random.randint(low=0, high=2**32))
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    else:
        score = clf_final.score(X_test, y_test)
    random_scores.append(score)

out_dict['shuffled scores, shuffle test/train separately'] = random_scores
print('random score: %.3f $\pm$ %.3f'%(np.mean(random_scores), np.std(random_scores)/np.sqrt(len(random_scores))))

# shuffle scores, test and train together
random_scores = []
for i in range(100):
    np.random.shuffle(y)
    if args.shuffle_input_feats:
        np.random.shuffle(x)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size,
                                                                 random_state=np.random.randint(low=0, high=2**32))
    else:
        _, _, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size,
                                                                 random_state=np.random.randint(low=0, high=2**32))
    if args.retrain:
        clf = ensemble.RandomForestRegressor(n_estimators = n,
                                             max_depth = d,
                                             min_samples_split = s,
                                             max_features = method,
                                             max_samples = m,
                                             random_state=np.random.randint(low=0, high=2**32))
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    else:
        score = clf_final.score(X_test, y_test)
    random_scores.append(score)
out_dict['shuffled scores, shuffle test/train together'] = random_scores

pk.dump(out_dict, open(args.save_dir + fname_out, 'wb'), protocol=pk.HIGHEST_PROTOCOL)

print('\noptimal param combo:')
print('num trees:  %.2f'%(n))
if d is None:
    print('tree depth: None')
else:
    print('tree depth: %.2f'%(d))
print('split:      %.2f'%(s))
print('method:     %s'%(method))
if m is None:
    print('max samps: None')
else:
    print('max samps:  %s'%(m))

