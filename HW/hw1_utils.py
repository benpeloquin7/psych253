"""hw1_utils.py

Utilities for assignment 1.

"""

import numpy as np
import tqdm
import scipy.stats as stats


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


def get_neuron_reliability(data, neuron_id, num_trials=10, seed=0):
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
    d_first = np.array(data)[first_half, :, neuron_id].mean(0)
    d_second = np.array(data)[second_half, :, neuron_id].mean(0)
    return stats.pearsonr(d_first, d_second)[0]


def get_neuron_reliabilities(data, neuron_id, num_splits=100, num_trials=10):
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
            get_neuron_reliability(data, neuron_id, num_trials, split_num))
    return np.array(res)


def get_all_neuron_reliabilities(data, num_splits=400, num_trials=40):
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
                                       num_trials=10):
    d = data['variation_level_{}'.format(level)]
    reliabilities = get_all_neuron_reliabilities(d, num_splits, num_trials)
    return np.mean(reliabilities, axis=1), np.std(reliabilities,
                                                  axis=1), reliabilities
