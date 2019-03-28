import glob
import sys

import h5py
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.colors import Normalize
import seaborn as sns

# Visualization arm of SCRAPC


def plot_image_with_alpha(im, mask, thresh = 0.4, vmin = -np.pi, vmax = np.pi):
    '''Plots an image with alpha values'''

    # Make the RGBA array
    cmap = plt.cm.gist_rainbow
    colors = Normalize(vmin, vmax, clip=True)(im); colors = cmap(colors)

    # Normalize mask and set A value to the mask value
    mask = mask - np.min(mask); mask = mask/np.max(mask);
    mask[mask>thresh] = 1;
    colors[..., -1] = mask;

    # Plot a black bacground and masked image
    plt.imshow(np.ones_like(im), vmin = 0, vmax = 1, cmap= 'binary')
    plt.imshow(colors)

def plot_raster(st, cl, cluster_numbers = range(10), trange = [5000., 5050.], labels = None):
    ''' Plots a raster of the given cluster numbers in the given range trange'''
    lo, hi = trange
    colors = sns.color_palette("hls", len(cluster_numbers))
    for neuron_id, cc in enumerate(cluster_numbers):
        tmp = st[cl==cc]
        sub = tmp[(tmp>=lo)*(tmp<=hi)]
        current_label = str(cc) if labels == None else labels[neuron_id]
        plt.eventplot(sub, lineoffsets=neuron_id, colors = colors[neuron_id], label = current_label) 
        #print(len(sub), end=" ")

