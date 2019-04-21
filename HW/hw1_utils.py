"""hw1_utils.py

Utilities for assignment 1.

"""

from collections import defaultdict
import numpy as np
import tqdm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import time

from psych253.metrics import get_confusion_matrix


def create_random_split(data, num_trials=None, seed=0):
    """Split trial data for a single nueron.

    Parameters
    ----------
    d: np.array
        Shape (trials, images, neurons)
    num_trials: int
        Number of trials to consider.
    seed: int
        Seed for random number generator.

    Returns
    -------
    tuple
        Tuple of trial indices.

    """
    # Just take total trials if no amount is entered
    if num_trials is None:
        num_trials = data.shape[0]
    random_number_generator = np.random.RandomState(seed=seed)
    perm = random_number_generator.permutation(num_trials)
    first_half_of_trial_indices = perm[:int(num_trials / 2)]
    second_half_of_trial_indices = perm[int(num_trials / 2): num_trials]
    return first_half_of_trial_indices, second_half_of_trial_indices


def get_neuron_reliability(data, neuron_id, num_trials=10, seed=0, type_='regular'):
    """Get neuron reliability for a single split.

    Parameters
    ----------
    data: np.array (num_trials, num_images, num_neurons)
        Ventral input data.

    Returns
    -------
    float
        Single correlation measure.

    """
    assert len(data.shape) == 3  # Check dims
    first_half, second_half = create_random_split(data, num_trials, seed)
    if type_ == 'efficient':
        d_first = np.expand_dims(np.array(data)[first_half, :, neuron_id].mean(0), 0)
        d_second = np.expand_dims(np.array(data)[second_half, :, neuron_id].mean(0), 0)
        d_comb = np.stack([d_first, d_second]).squeeze(1)
        return np.corrcoef(d_comb)[0][1]
    d_first = np.array(data)[first_half, :, neuron_id].mean(0)
    d_second = np.array(data)[second_half, :, neuron_id].mean(0)
    return stats.pearsonr(d_first, d_second)[0]


def get_neuron_reliabilities(data, neuron_id, num_splits=100, num_trials=10, type_='regular'):
    """Get (multiple) reliabilities data for a single neuron.

    Parameters
    ----------
    data: np.array (num_trials, num_images, num_neurons)
        Input neural data.
    neron_id: int
        Neuron identifier.
    num_splits: int
        Number of split-half splits to make.
    num_trials: int
        Split size is num_trials / 2.

    Returns
    -------
    np.array (num_splits, )
        Split-half correlations ofr num_splits.

    """
    res = []
    for split_num in range(num_splits):
        # Note split_num here for random seed
        res.append(
            get_neuron_reliability(data, neuron_id, num_trials, split_num, type_))
    return np.array(res)


def get_all_neuron_reliabilities(data, num_splits=400, num_trials=40, type_='regular'):
    """
    We concluded that `having num_splits ~ 10 * num_trials
    is adequate so we use these as our default params.

    Parameters
    ----------
    data: np.array (num_trials, num_neurons)
        Input neural data averaged over images.

    Returns
    -------
    np.array (num_neurons, num_splits)
        Correlation measures for each neuron.

    """
    assert len(data.shape) == 3

    res = []
    for neuron_idx in tqdm.tqdm(range(data.shape[2])):
        res.append(
            get_neuron_reliabilities(data, neuron_idx, num_splits, num_trials))
    return np.stack(res)


def get_reliability_by_variation_level(data, level=0, num_splits=20,
                                       num_trials=10, type_='regular'):
    """Get neuron reliability by variation level."""
    d = data['variation_level_{}'.format(level)]
    reliabilities = \
        get_all_neuron_reliabilities(d, num_splits, num_trials, type_)
    return np.mean(reliabilities, axis=1), np.std(reliabilities,
                                                  axis=1), reliabilities


def run_models_over_single_split(models, X_train, y_train, X_test, y_test,
                                 verbose=False):
    """Run a set of models over a single train/test split.

    Parameters
    ----------
    models: list[BaseEsimator]
        List of sklearn models.
    X_train: np.array (num_examples, num_features)
        Training features.
    y_train: np.array (num_examples, )
        Training supervision.
    X_test: np.array (num_examples, num_features)
        Test features.
    y_train: np.array (num_examples, )
        Test supervision.
    verbose: bool
        Print current model run.

    Returns
        tuple (dict, list)
            First dictionary contains classification results.
            List contains classifier run-times.

    """
    create_model_name = lambda x: '{}_{}'.format(x.penalty, x.C)
    model_names = map(create_model_name, models)
    labels = set(y_test)
    # Make sure we don't have any duplicate models
    assert len(set(model_names)) == len(models)

    results = {}
    run_times = []
    for m, n in zip(models, model_names):
        if verbose:
            print("Running {} model".format(n))
        t0 = time.time()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        cm = get_confusion_matrix(preds, y_test, labels)
        run_times.append(time.time() - t0)
        results[n] = cm
    return results, run_times


def run_models_over_n_splits(models, n_splits, X, y, test_size=0.25,
                             verbose=True):
    """Run a set of models of n train/test splits.

    Parameters
    ----------
    models: list[BaseEsimator]
        List of sklearn models.
    n_splits: int
        Number of train/test splits to make.
    X: np.array (num_examples, num_features)
        Data to split into train/test.
    y: np.array (num_examples, )
        Klasses.
    test_size: float [Default: 0.25]
        Test split size.
    random_state: int
        Random state initializer.
    verbose: bool
        Print current model run.

    Returns
        tuple (dict, dict)
            First dictionary contains classification results.
            Second dictionary contains classifier run-times.

    """
    if verbose:
        print(
            "Running {} models over {} splits.".format(len(models), n_splits))
    results = []
    times = []
    pbar = tqdm.tqdm(total=n_splits)
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size,
                             random_state=i)
        res, tim = run_models_over_single_split(models, X_train, y_train,
                                                X_test, y_test)
        results.append(res)
        times.append(tim)
        if verbose:
            pbar.update()
    pbar.close()
    model_results = defaultdict(list)
    model_run_times = defaultdict(list)
    for result, r_times in zip(results, times):
        for m_name, m_time in zip(result, r_times):
            model_results[m_name].append(result[m_name])
            model_run_times[m_name].append(m_time)
    return model_results, model_run_times

