# -*- coding: utf-8 -*-



#------------------------------------------------------------------------------
# Imports """Cross-correlograms."""
# THIS CODE IS FROM CORTEX-LAB GITHUB, USED FOR PHYS VISUALIZER
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from six import string_types, integer_types
import numba

_ACCEPTED_ARRAY_DTYPES = (np.float, np.float32, np.float64,
                          np.int, np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          np.bool)


@numba.jit
def numba_corr(data1, data2):
    ''' Faster pearson correlation using compiled numba code '''
    M = data1.size

    sum1 = 0.
    sum2 = 0.
    for i in range(M):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in range(M):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / M) ** .5
    std2 = (var_sum2 / M) ** .5
    cross_mean = cross_sum / M

    return (cross_mean - mean1 * mean2) / (std1 * std2)

def palplot_legend(pal, titles, size=1):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot
    """
    if len(pal) != len(titles):
        raise('Titles and colors must match')
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels(list(titles))
    ax.set_yticklabels([])


def reorder_matrix(mat, order):
    # Will reorder the given matrix mat such that it is now in new order
    # THis can be used for example to reorder a correlation matrix such
    # that instead of having the order be determined by probe number, it
    # can be determined by location
    reordered_mat = np.copy(mat)
    reordered_mat = reordered_mat[order, :]  # Reorder rows
    reordered_mat = reordered_mat[:, order]  # Reorder columns
    return reordered_mat



#------------------------------------------------------------------------------
# Cross-correlograms
#------------------------------------------------------------------------------

def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitely assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))

    return tmp[arr]

def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.
    Avoid a copy if possible."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out

def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _as_array(arr)
    indices = _as_array(indices)
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = _as_array(arr)
    return arr[steps:] - arr[:len(arr) - steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1),
                    dtype=np.int32)


def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                      correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def correlograms(spike_times,
                 spike_clusters,
                 cluster_ids=None,
                 sample_rate=1.,
                 bin_size=None,
                 window_size=None,
                 symmetrize=True,
                 ):
    """Compute all pairwise cross-correlograms among the clusters appearing
    in `spike_clusters`.
    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    spike_clusters : array-like
        Spike-cluster mapping.
    cluster_ids : array-like
        The list of unique clusters, in any order. That order will be used
        in the output array.
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise
        CCGs.
    """
    # Sort to ascending
    idx = np.argsort(spike_times)
    spike_times = spike_times[idx]
    spike_clusters = spike_clusters[idx]

    assert sample_rate > 0.
    assert np.all(np.diff(spike_times) >= 0), ("The spike times must be "
                                               "increasing.")

    # Get the spike samples.
    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_samples = (spike_times * sample_rate).astype(np.int64)

    spike_clusters = _as_array(spike_clusters)

    assert spike_samples.ndim == 1
    assert spike_samples.shape == spike_clusters.shape

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds
    binsize = int(sample_rate * bin_size)  # in samples
    assert binsize >= 1

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds
    winsize_bins = 2 * int(.5 * window_size / bin_size) + 1

    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    # Take the cluster oder into account.
    if cluster_ids is None:
        clusters = np.unique(spike_clusters)
    else:
        clusters = _as_array(cluster_ids)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters = spike_clusters.astype(int)
    print(clusters)
    print(spike_clusters)
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=np.bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # # Update the masks given the clusters to update.
        # m0 = np.in1d(spike_clusters[:-shift], clusters)
        # m = m & m0
        # d = spike_diff_b[m]
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index((spike_clusters_i[:-shift][m],
                                        spike_clusters_i[+shift:][m],
                                        d),
                                       correlograms.shape)
        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        print(str(shift), end=" ")
        shift += 1

    # Remove ACG peaks.
    correlograms[np.arange(n_clusters),
                 np.arange(n_clusters),
                 0] = 0

    if symmetrize:
        return _symmetrize_correlograms(correlograms)
    else:
        return correlograms