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
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row, gridplot
from bokeh.plotting import figure, show, output_file


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

def plot_raster(st, cl, cluster_numbers = range(10), trange = [5000., 5050.], labels = None, use_bokeh = False):
    ''' Plots a raster of the given cluster numbers in the given range trange'''
    
    if labels != None:
        assert len(cluster_numbers) == len(labels)
    
    lo, hi = trange
    colors = sns.color_palette("hls", len(cluster_numbers))
    if use_bokeh:
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
        p = figure(title="Raster", tools=TOOLS, width = 1000, height = 300)
        p.background_fill_color = "black"

    for neuron_id, cc in enumerate(cluster_numbers):
        tmp = st[cl==cc]
        sub = tmp[(tmp>=lo)*(tmp<=hi)]
        current_label = str(cc) if labels == None else labels[neuron_id]
        if use_bokeh == True:
            # Make a list of lists for each spike and add as a tick mark
            xs = [[i, i] for i in sub]
            ys = [[neuron_id, neuron_id+0.75] for i in sub]
            p.multi_line(xs, ys, line_color = colors.as_hex()[neuron_id], legend = current_label)
            
        else: # Use matplotlib
            plt.eventplot(sub, lineoffsets=neuron_id, colors = colors[neuron_id], 
            label = current_label) 
        #print(len(sub), end=" ")

    if use_bokeh:
        show(p)
    else:
        plt.legend()
