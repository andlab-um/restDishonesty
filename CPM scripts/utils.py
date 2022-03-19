import os
import json
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore") 

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

from nilearn import input_data, datasets

import seaborn as sns
import matplotlib.pylab as plt


# Calculate overlap between neurosynth ROIs and 264ROIs
def extract_roi(network2path, network2rois_path):
    """Extract overlapped ROIs between neurosynth network and 264ROIs template

    Parameters
    ----------
    network2path : dict
        Mapping from neurosynth network name to corresponding file path.
    network2rois_path : str
        Path to save the dict mapping from network name and overlapped ROIs (with 264 template) in the network.

    Returns
    -------
    network2rois: dict
       Mapping from network name and overlapped ROIs (with 264 template) in it.
    """
    # read 264 ROIs and create mask
    power = datasets.fetch_coords_power_2011()
    coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coords, radius=5)
    # get mapping from network to roi indexs
    signals = list()
    for network, path in network2path.items():
        # extract network activation from 264ROIs mask to calculate overlap
        signals.append(spheres_masker.fit_transform(path).reshape(-1,))
    signals = abs(np.array(signals))
    # get index of network with max value for each roi
    argmax = np.argmax(signals, axis=0)
    # set nonactivation roi to -1
    argmax[np.argwhere(signals[argmax, range(signals.shape[1])] <= 0).reshape(-1,)] = -1
    network2rois = {network: np.argwhere(argmax == i).reshape(-1,).tolist() for i, network in enumerate(network2path.keys())}
    # save index dict
    with open(network2rois_path, "w") as f:
        json.dump(network2rois, f, indent=4)
    return network2rois

# Extract FC matrix based on the overlap area from individual files
def get_input(in_path, exclude_sub, outfile, select_rois):
    """Construct the data(features and target value) from individual fc files

    Parameters
    ----------
    in_path : str
        Individual FC files path
    exclude_sub : list
        subject id to exclude.
    outfile : str
        Filename of the final data output
    select_rois : list
        ROI index, here correspond to the overlapped ROIs between 264ROIs and neurosynth ROIs

    Returns
    -------
    array-like, shape (n_samples, n_features + 1)
        Final data.
    """
    joint_fc = []
    fc_path = os.path.join(in_path, "fc")
    beh_path = os.path.join(in_path, f"{os.path.basename(in_path)[:2]}_beh.txt")
    for sub in os.listdir(fc_path):
        # skip subjects
        if os.path.splitext(sub)[0].split("_")[-1] in exclude_sub:
            continue
        # read functional connectivity matrix
        raw_fc = np.loadtxt(os.path.join(fc_path, sub))
        # only select overlaped ROIs, see above
        roi_fc = raw_fc[np.ix_(select_rois, select_rois)]
        # get upper triangular and convert to (n, 1)
        single_side = roi_fc[np.triu_indices(roi_fc.shape[0], k=1)]
        joint_fc.append(single_side)
    # stack fc(X) and behavioral result(y)
    input = np.hstack([np.array(joint_fc), np.loadtxt(beh_path).reshape(-1, 1)])
    np.save(outfile, input)
    return input

# regression model
def regression(X_train, y_train, X_test, y_test, feature_index):
    """Building an regression model and get predictions
    Returns
    -------
    dict
        containing target values, prediction values, and feature_index(index of selected fc edges)
    """
    # select feature based on p value
    if feature_index is None:
        feature_index = np.argwhere(np.array([stats.pearsonr(X_train[:, i], y_train)[1]
            for i in range(X_train.shape[1])]) < 0.01).reshape(-1, )
    # get train&&test data
    X_train, X_test = X_train[:, feature_index], X_test[:, feature_index]
    
    # CPM (pos + neg -> y)
    feature_beh_corr = np.array([stats.pearsonr(X_train[:, i], y_train)[1] for i in range(X_train.shape[1])])
    X_train = np.hstack((X_train[:, feature_beh_corr > 0].sum(axis=1, keepdims=True), X_train[:, feature_beh_corr < 0].sum(axis=1, keepdims=True)))
    X_test = np.hstack((X_test[:, feature_beh_corr > 0].sum(axis=1, keepdims=True), X_test[:, feature_beh_corr < 0].sum(axis=1, keepdims=True)))
    model = LinearRegression()

    # model fit and predict
    model.fit(X_train, y_train)
    y_p = model.predict(X_test)
    # add predict value and ground truth
    return dict(y_t=y_test.tolist(), y_p=y_p.tolist(), model_coef=model.coef_.tolist(), f_index=feature_index)
    
# prediction pipeline
def prediction(data1_X, data1_y, data2_X, data2_y, cv="loocv", feature_index=None, validation="single", seed=9873):
    """Get prediction

    Parameters
    ----------
    data1_X : array-like, shape (n_samples, n_features)
        Features of dataset1
    data1_y : array-like, shape (n_samples,)
        Target of dataset1
    data2_X : array-like, shape (m_samples, m_features)
        Features of dataset2
    data2_y : array-like, shape (m_samples,)
        Target of dataset2
    cv : str or int, optional
        loocv or k-fold, by default "loocv", can also be 5, 10 if using k-fold
    seed : int, optional
        random seed, by default 9873
    feature_index : list or 1D array-like, optional
        index to select significant functional connectivity edges, by default None
    validation : str, optional
        validation approach, can be "single" or "kfold", by default "single".
        "single" means train on data1, test on data2
        "kfold" means train on data1, test on data2 with k-fold
    Returns
    -------
    y_actual, y_predict, _, _
        the last two can be ignored.
    """
    # save predict and actual value
    y_actual, y_predict, model_coef = list(), list(), list()
    feature_per, feature_freq = list(), np.zeros(data1_X.shape[1])
    # test external data on internal model
    if validation == "single":
        result = [regression(data1_X, data1_y, data2_X, data2_y, feature_index)]
    # train and test on the same dataset, here data1 is the same as data2
    elif validation == "kfold":
        # define kfold
        if cv == "loocv":
            kfold, repeat = data1_X.shape[0], 1
        else:
            kfold, repeat = cv, cv
        rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeat, random_state=seed)
        result = Parallel(n_jobs=32)(delayed(regression)(data1_X[train_index], data1_y[train_index], data2_X[test_index],
                data2_y[test_index], feature_index) for train_index, test_index in rkf.split(data1_X))
    else:
        print("Invalid validation parameter")
    # save all predict value and actual value and feature_index
    for r in result:
        y_actual.extend(r["y_t"])
        y_predict.extend(r["y_p"])
        model_coef.append(r["model_coef"])
        feature_per.append(len(r["f_index"]))
        feature_freq[r["f_index"]] += 1
    return y_actual, y_predict, model_coef, feature_per, feature_freq

def permutation(d1_X, d1_y, d2_X, d2_y, true_coef, feature_index=None, validation="single", permutation_n=1000):
    """Run permutation test

    Parameters
    ----------
    data1_X : array-like, shape (n_samples, n_features)
        Features of dataset1
    data1_y : array-like, shape (n_samples,)
        Target of dataset1
    data2_X : array-like, shape (m_samples, m_features)
        Features of dataset2
    data2_y : array-like, shape (m_samples,)
        Target of dataset2
    true_coef : float
        Coefficient performance of true model, used as threshold 
    feature_index : list or array-like
        Index of selected features, by default None
    validation : str, optional
        Training methods, by default "single", see code for more information
    permutation_n : int, optional
        Permutation times, by default 1000

    Returns
    -------
    coefs: list or array-like
        Coefficients of every permutation.
    """
    # run permutation
    coefs, pvalues = np.zeros(permutation_n), np.zeros(permutation_n)
    for i in trange(permutation_n):
        # shuffle y
        d2_y_shuffle = np.random.RandomState(seed=i).permutation(d2_y)
        y_actual, y_predict, _, _, _ = prediction(d1_X, d1_y, d2_X, d2_y_shuffle, seed=i, feature_index=feature_index, validation=validation)
        # get correlation coefficient
        coefs[i], pvalues[i] = stats.spearmanr(y_actual, y_predict)
    # get p_value of permutation
    permutation_p = (coefs > true_coef).mean()
    print(f"permutation pvalue: {permutation_p}, feature number: {len(feature_index)}")
    return coefs

# recovery fc matrix based on features and value(or feature coefficients)
def recovery_fc(select_rois, feature_index, value, rois_n=264):
    """Recovery the FC matrix based on selected feature indexs.

    Parameters
    ----------
    select_rois : array-like
        Selected ROI indexs
    feature_index : array-like
        Selected feature indexs
    value : array-like
        Feature coefficients
    rois_n : int, optional
        Full ROI numbers, by default 264

    Returns
    -------
    fc: 2-D array, shape (rois_n, rois_n)
        Full functional edge coefficients
    """
    # recovery small fc matrix first
    fc_select = np.zeros((len(select_rois), len(select_rois)))
    triu_index = np.triu_indices(len(select_rois), k=1)
    tmp = np.zeros_like(fc_select[triu_index])
    tmp[feature_index] = value
    fc_select[triu_index] = tmp
    fc_select = fc_select + fc_select.T
    # recovery full fc matrix
    fc = np.zeros((rois_n, rois_n))
    fc[np.ix_(select_rois, select_rois)] = fc_select
    return fc

# plot correlation between target value and prediction
def plot_corr(y_true, y_pred, save_path=None, color="green"):
    # get correlation
    coef, p = stats.spearmanr(y_true, y_pred)
    print(f'MSE: {mean_squared_error(y_true, y_pred)}')
    # regression plot
    plt.close()
    plt.figure(figsize=(6, 5))
    # set x, y range
    plt.xlim(-0.02, 0.53)
    plt.ylim(0.05, 0.63)
    p_label = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    sns.regplot(x=y_true, y=y_pred, line_kws=dict(color=color, label=fr"$\rho$ = {coef:.3f}" + f"\np {p_label}"),
                scatter_kws=dict(color=color, s=20), truncate=False)
    plt.legend(loc="upper left")
    plt.xlabel("")
    plt.ylabel("")
    # remove top and right spines
    sns.despine()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# plot permutation result
def plot_permutation(threshold, coefs, save_path=None, kdecolor="green"):
    # plot hist
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.xlim(-0.75, 0.75)
    plt.ylim(0, 2.4)
    ax = sns.histplot(coefs, stat="density", kde=True, color=kdecolor, edgecolor="white")
    # sns.kdeplot(coefs, color="darkcyan")
    plt.vlines([threshold], *ax.get_ylim(), colors="blue", linestyles="dashed")
    sns.despine()
    # remove labels
    plt.ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# plot heatmap based on fc
def plot_conn(name2index, full_matrix, save_path=None, fontsize=12, filter=None, colormap=plt.cm.Blues, annot=False):
    # filter networks with valid value
    fil_name2index = {name: index for name, index in name2index.items() if abs(np.sum(full_matrix[index])) > 0}
    # matrix value filter
    tmp_matrix = full_matrix.copy()
    if filter == "abs":
        tmp_matrix = abs(tmp_matrix)
    elif filter == "pos":
        tmp_matrix[tmp_matrix < 0] = 0
    elif filter == "neg":
        tmp_matrix[tmp_matrix > 0] = 0
    print(f"Coefficients sum: {np.sum(tmp_matrix)}")
    # construct plotting fc matrix
    fc = np.zeros((len(fil_name2index), len(fil_name2index)))
    # get summarized fc matrix
    for i, name_1 in enumerate(fil_name2index.keys()):
        for j, name_2 in enumerate(fil_name2index.keys()):
            fc[i, j] = np.sum(tmp_matrix[np.ix_(fil_name2index[name_1], fil_name2index[name_2])])
    # plot
    mask = np.zeros_like(fc, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    plt.figure(figsize=(10, 8))
    with plt.style.context({"axes.labelsize": fontsize, "xtick.labelsize": fontsize, "ytick.labelsize": fontsize}):
        g = sns.heatmap(fc, mask=mask, annot=annot, annot_kws=dict(size=fontsize), fmt=".0f", cmap=colormap, linewidths=0.5, square=True, xticklabels=fil_name2index.keys(), yticklabels=fil_name2index.keys())
        g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment="right")
        g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment="right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
