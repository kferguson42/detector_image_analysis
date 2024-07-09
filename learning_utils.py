import pickle as pk, numpy as np, sys
# for the following import to work, you must add the repository to your PYTHONPATH (see README)
from tes_analysis import tes_mapping_utils

def get_wafer_classes(wafer_num, impute_input_nans=True, average_double_bolos=False,
                      data_dir='/home/kferguson/spt3g_wafer_ML/data/'):
    '''
    Do all of the loading in and massaging of input/output feature arrays.

    Input args:
    - wafer_num (int or list of ints): The wafer number to get an array for. Can also
      be a list of wafer numbers, in which case the function returns an array
      containing all of the listed wafers.
    - impute_input_nans (boolean): Whether to use MissForest to impute values for
      all nans in the input array. If False, will leave the nans in, so be wary.
    - average_double_bolos (boolean): What to do with bolos for which we have two
      sets of input features. False will leave the doubles in, and True will average
      them (this may get weird for features like the angle or other geometrically-
      dependent features).
    - data_dir (string): The directory where the data live.

    Returns:
    - features: Input features for ML algorithm
    - classes: Output features for ML algorithm
    '''
    def get_wafer_classes_inner(wafer_num, impute_input_nans=True, average_double_bolos=True):
        try:
            pk_in = pk.load(open(data_dir + 'w%d_input_features.pkl'%(wafer_num), 'rb'))
            pk_out = pk.load(open(data_dir + 'w%d_TES_classes_%d_GofT.pkl'%(wafer_num, wafer_to_date[wafer_num]), 'rb'))
        except:
            pk_in = pickle5.load(open(data_dir + 'w%d_input_features.pkl'%(wafer_num), 'rb'))
            pk_out = pickle5.load(open(data_dir + 'w%d_TES_classes_%d_GofT.pkl'%(wafer_num, wafer_to_date[wafer_num]), 'rb'))

        features = pk_in['features']
        classes = []
        double_inds = {}

        if impute_input_nans:
            # do this before averaging/removing zero-classified bolos
            import missingpy
            mf = missingpy.MissForest()
            features = mf.fit_transform(features)

        if average_double_bolos:
            # get indicies of repeated bolos
            for i, b in enumerate(pk_in['bolos']):
                if b in pk_in['bolos'][:i]:
                    double_inds[b] = np.where(np.asarray(pk_in['bolos']) == b)

        feats_good = []
        for i, b in enumerate(pk_in['bolos']):
            # only get bolo info for which we have an image
            if average_double_bolos and b in double_inds.keys():
                if i == double_inds[b][0][0]:
                    # only append class first time it appears
                    classes.append(pk_out[b])
                    # average input features
                    i1 = double_inds[b][0][0]
                    i2 = double_inds[b][0][1]
                    feats_good.append(np.mean([features[i1,:], features[i2,:]], axis=0))
            else:
                classes.append(pk_out[b])
                feats_good.append(features[i,:])
        features = np.array(feats_good)
        classes = np.array(classes)

        # get rid of "0" class because it's just bolos that aren't in HWM and shouldn't correlate with any features
        inds = np.where(classes != 0)
        features = features[inds]
        classes = classes[inds]

        return features, classes

    wafer_to_date = {148: 20210929,
                     162: 20210801,
                     187: 20211112}

    if type(wafer_num) is list:
        features = []; classes = []
        for w in wafer_num:
            feats, classs = get_wafer_classes_inner(w, impute_input_nans=impute_input_nans, average_double_bolos=average_double_bolos)
            features.extend(feats)
            classes.extend(classs)
        features = np.array(features)
        classes = np.array(classes)
    else:
        features, classes = get_wafer_classes_inner(wafer_num, impute_input_nans=impute_input_nans, average_double_bolos=average_double_bolos)

    return features, classes


def get_wafer_characteristics(wafer_num, impute_input_nans=True, impute_output_nans=True, average_double_bolos=True,
                              data_dir = '/home/kferguson/spt3g_wafer_ML/data/', return_wafer_nums=False):
    '''
    Do all of the loading in and massaging of input/output feature arrays.

    Input args:
    - wafer_num (int or list of ints): The wafer number to get an array for. Can also
      be a list of wafer numbers, in which case the function returns an array
      containing all of the listed wafers.
    - impute_input_nans (boolean): Whether to use MissForest to impute values for
      all nans in the input array. If False, will leave the nans in, so be wary.
    - impute_output_nans (boolean): Whether to use MissForest to impute values for
      all nans in the output array. If False, will leave the nans in, so be wary.
    - average_double_bolos (boolean): What to do with bolos for which we have two
      sets of input features. False will leave the doubles in, and True will average
      them (this may get weird for features like the angle or other geometrically-
      dependent features).
    - data_dir (string): The directory where the data live.
    - return_wafer_nums (boolean): Whether to return an array of the wafer each bolo
      is on (in addition to the two usual outputs).

    Returns:
    - features: Input features for ML algorithm
    - characteristics: Output features for ML algorithm
    '''
    
    def get_wafer_characteristics_inner(wafer_num, impute_input_nans=True, impute_output_nans=True, average_double_bolos=True):
        # First, read in data and futz with everything so the inputs and outputs match
        try:
            pk_in = pk.load(open(data_dir + 'w%d_input_features.pkl'%(wafer_num), 'rb'))
            pk_out = pk.load(open(data_dir + 'w%d_resistance_features.pkl'%(wafer_num), 'rb'))
        except:
            pk_in = pickle5.load(open(data_dir + 'w%d_input_features.pkl'%(wafer_num), 'rb'))
            pk_out = pickle5.load(open(data_dir + 'w%d_resistance_features.pkl'%(wafer_num), 'rb'))
        wafer_hwm_file = data_dir + 'w%d_wafer_hwm.csv'%(wafer_num)

        readout_names = np.array(pk_out['bolos'])
        features_in = pk_in['features']
        features_out = []
        double_inds = {}

        if impute_input_nans:
            # do this before averaging/removing outlier bolos
            import missingpy
            mf = missingpy.MissForest()
            features_in = mf.fit_transform(features_in)

        if average_double_bolos:
            # get indicies of repeated bolos
            for i, b in enumerate(pk_in['bolos']):
                if b in pk_in['bolos'][:i]:
                    double_inds[b] = np.where(np.asarray(pk_in['bolos']) == b)
        
        r, p, o = (0, 0, 0)

        feats_good = []
        for i, b in enumerate(pk_in['bolos']):
            # only get bolo info for which we have an image
            readout_name = tes_mapping_utils.image_num_to_readout_bolo_name(b, wafer_hwm_file)
            if readout_name is None:
                # bolo imaged but not in HWM, can't do anything with it
                r += 1
                continue
            ind = np.where(readout_names == readout_name)[0]
            if len(ind) == 0:
                # bolo imaged but no R(T) data, can't do anything with it
                p += 1
                continue
            else:
                ind = ind[0]
            if pk_out['features'][ind,0] > 4:
                # remove outliers (may want to get rid of this)
                o += 1
                continue

            if average_double_bolos and b in double_inds.keys():
                if i == double_inds[b][0][0]:
                    # only append output features first time bolo appears
                    features_out.append(pk_out['features'][ind,:])
                    #features_out.append(pk_out['features'][ind,0])  # only use r_normal

                    # average input features
                    i1 = double_inds[b][0][0]
                    i2 = double_inds[b][0][1]
                    feats_good.append(np.mean([features_in[i1,:], features_in[i2,:]], axis=0))
            else:
                features_out.append(pk_out['features'][ind,:])
                feats_good.append(features_in[i,:])
        features_in = np.array(feats_good)
        features_out = np.array(features_out)

        if impute_output_nans:
            import missingpy
            mf = missingpy.MissForest()
            features_out = mf.fit_transform(features_out)
            
        return features_in, features_out, [r, p, o]

    if type(wafer_num) is list:
        features = []; chars = []; nums = []; wafer_labels = []
        for w in wafer_num:
            feats, charss, acct = get_wafer_characteristics_inner(w, impute_input_nans=impute_input_nans,
                                                                  impute_output_nans=impute_output_nans,
                                                                  average_double_bolos=average_double_bolos)
            features.extend(feats)
            chars.extend(charss)
            nums.append(acct)
            wafer_labels.extend(w*np.ones(feats.shape[0]))
        features = np.array(features)
        chars = np.array(chars)
        wafer_labels = np.array(wafer_labels)
    else:
        features, chars, nums = get_wafer_characteristics_inner(wafer_num, impute_input_nans=impute_input_nans,
                                                                impute_output_nans=impute_output_nans,
                                                                average_double_bolos=average_double_bolos)
        wafer_labels = wafer_num * np.ones(features.shape[0])

    if return_wafer_nums:
        return features, chars, nums, wafer_labels
    else:
        return features, chars, nums


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    NOTE: Adapted from https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html

    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.use('Agg')
    from sklearn.model_selection import learning_curve
    
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return fig

def plot_precision_recall_curve(estimator, X, y_in, classes=[1,2,3], random_state=None):
    '''
    NOTE: Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

    Plots the precision-recall curve for a classification estimator. In cases with more than
    two classes, uses a one-vs-rest scheme and plots curves for each split.
    '''
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
    from sklearn.model_selection import train_test_split
    import matplotlib as mpl, matplotlib.pyplot as plt
    #from itertools import cycle
    mpl.use('Agg')

    Y = label_binarize(y_in, classes=classes)
    n_classes = Y.shape[1]

    # Split into training and test
    if random_state is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    classifier = OneVsRestClassifier(estimator)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, c in enumerate(classes):
        precision[c], recall[c], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[c] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    # setup plot details
    colors = ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]

    fig, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, c in enumerate(classes):
        color = colors[i]
        display = PrecisionRecallDisplay(
            recall=recall[c],
            precision=precision[c],
            average_precision=average_precision[c],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {c}", color=color)
        # plot no-skill line
        these_y = y_test[:,i]
        positive = these_y[these_y == 1]
        negative = these_y[these_y == 0]
        ax.axhline(positive.shape[0] / (positive.shape[0] + negative.shape[0]), color=color,
                   linestyle='--', label=f"No-skill line for class {c}", alpha=0.6)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Precision-Recall curves")

    return fig


