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
from sklearn.decomposition import FastICA
from matplotlib.colors import Normalize
import scrapc_viz as scviz

def upsample_timestamps(ts, factor):
    """Code for upsampling the timestamps in ts by a given factor, useful for experiments where you have only received a ttl every N frames
    
    Arguments:
        ts {array} -- 1d array of timestamps
        factor {int} -- Factor by which to upsample
    
    Returns:
        out {array} -- Interpolation of timestamps
    """
    out = np.interp(np.linspace(0, 1, len(ts)*factor), np.linspace(0, 1, len(ts)), ts)
    return out

def parse_timestamps(signal, timestamps, thresh = 2.5, interval_between_experiments = 2, min_isi = 0.01, min_number_exp = 20):
    '''
    Arguments:
        signal {array}                      -- 1-d signal of voltages to parse
        timestamps {array}                  -- 1-d array of timestamps, same length as signal
    
    Keyword Arguments:
        thresh {float}                      -- Threshold for voltage crossing (default: {2.5})
        interval_between_experiments {int}  -- Minimum interval between experimetns (seconds) (default: {2})
        min_isi {float}                     -- Minimum inter-stim-interval to count as separate events (default: {0.01})
        min_number_exp {int}                -- Minimum number of events in a given experiment, discard experiemnts with less (default: {20})
    
    Returns:
        out_list [list]                     -- List of experiments, each entry has timestamps of all events
        times [array]                       -- 1d array of timestamps of all events that were considered valid in the parsing
    '''

    # Find timestamps of all stimulus frames..
    times = timestamps[1:][np.logical_and(signal[1:]>thresh, signal[:-1]<=thresh)]
    # Throw out ones that dont pass a mini isi
    discard_idx = np.where(np.diff(times) < min_isi)[0] + 1;
    times = np.delete(times, discard_idx)
    
    # Parse experiments
    cuts = np.where(np.diff(times) > interval_between_experiments)[0]; 

    # Anywhere it is inter-interval greater than interval represents a new experiment
    if len(cuts) > 0: # Condition where there is more than one experiment
        exp_list = []; 
        exp_list.append(times[:cuts[0]])
        for i in range(len(cuts)):
            if i == len(cuts)-1:
                exp_list.append(times[cuts[i]+1: ] )
            else:
                exp_list.append(times[cuts[i]+1:cuts[i+1]+1 ] )

        # Lastly throw out experiments that dont have the minimum number
        out_list = [s for s in exp_list if len(s) >= min_number_exp]      
    else:
        out_list = []; out_list.append(times)
	
    return out_list, times

def resample_xyt(data, oldx, newx,  dims = {'x':1, 'y':2, 't':0}):
    """Code for resamplying in the time dimension an xyt array, useful for resampling the fUS recording at the time of stimulus frames
    
    Arguments:
        data {3-d array}                    -- xyt 3d array
        oldx {array}                        -- timestamps corresponding to time dimension
        newx {array}                        -- timestamps of the new interpolated positions
    
    Keyword Arguments:
        dims {dict}                         -- x, y, and t dimensions in the 3d array (default: {'x':2, 'y':1, 't':0} )
    
    Returns:
        data_resampled {3-d array}          -- Returns the data in t,y,x form
    """

    data_reshaped = np.transpose(data, [dims['t'], dims['y'], dims['x']]); # Transpose to t, y, x
    sz = data_reshaped.shape
    data_resampled = np.zeros([len(newx), sz[1], sz[2]])

    print(data_reshaped.shape)
    print(data_resampled.shape)

    for xx in range(sz[1]):
        for yy in range(sz[2]):
            data_resampled[:,xx,yy] = np.interp(newx, oldx, data_reshaped[:,xx, yy])

    return data_resampled

def gaussian_filter_xyt(data, sigmas = [0,1,1], dims = {'x':2, 'y':1, 't':0}):
    """Gaussian filter the 3d array in x,y, and t dimensions, useful for pre-processing/cleaning fUS images
    
    Arguments:
        data {3-d array}            -- Data that will be filtered
    
    Keyword Arguments:
        sigmas {3-element list}     -- 3-element list containing the sigmas in each of the t, x, y dimensions (default: {[0,1,1]})
        dims {dict}                 -- The dimensions corresponding to x, y, and t (default: {{'x':2, 'y':1, 't':0}})
    
    Returns:
        [type] -- [description]
    """

    data_reshaped = np.transpose(data, [dims['t'], dims['y'], dims['x']]); # Transpose to t, y, x

    sz = data_reshaped.shape
    gauss_out = np.zeros_like(data_reshaped)
    # Filter in x and y
    for tt in range(sz[0]):
        gauss_out[tt, :, :] = gaussian_filter(data_reshaped[tt, :, :], sigmas[1:], truncate = 2)
    
    # Filter in t if necessary
    if sigmas[0] != 0: # Just perform xy gaussian smoothing
        for yy in range(sz[1]):
            for xx in range(sz[2]):
                gauss_out[:, yy, xx] = gaussian_filter1d(gauss_out[:, yy, xx], sigma = sigmas[0], truncate = 2)

    return gauss_out


def perform_ica(data, num_comps = 5, dims = {'x':2, 'y':1, 't':0}):
    """Performs an ica on the timedomain
    
    Arguments:
        data {3-d array}        -- Input of xyt, can be any order as long as it is accompanied with appropriate dims argument
    
    Keyword Arguments:
        num_comps {int}         -- Number of ICs to extract (default: {5})
        dims {dict}             -- The dimensions corresponding to x, y, and t (default: {{'x':2, 'y':1, 't':0}})
    
    Returns:
        comps_out {3-d array} -- Array of num_comps x X x Y
    """

    '''Performs ica on data input that is of the for (t,y,x)'''

    ica = FastICA(n_components=num_comps)

    data_reshaped = np.transpose(data, [dims['t'], dims['y'], dims['x']]); # Transpose to t, y, x

    sz = data_reshaped.shape
    data_re = data_reshaped.reshape([sz[0], sz[1]*sz[2]])
    # Make it zero mean and common std
    for ii in range(sz[1]*sz[2]):
        data_re[:, ii] = (data_re[:, ii] - np.mean(data_re[:, ii]))/np.std(data_re[:,ii])
    # Now perform ica
    print('fitting model...')
    ica.fit(data_re)
    print('done')
    comps_out = ica.components_.reshape([num_comps, sz[1], sz[2]])

    return comps_out

def compute_fft(data , dims = {'x':2, 'y':1, 't':0}, doPlot = False, mask_plot = True):
    ''' Compute the fourier transform and plots the first 100 power and phase maps
    
    dims = [x_idx, y_idx, t_idx]; Usually [1,2,0] as it is time-leading from david
    '''

    data_reshaped = np.transpose(data, [dims['t'], dims['y'], dims['x']]); # Transpose to t, y, x
    out = np.fft.fft(data_reshaped, axis = 0); # take along time
    
    power_map = np.abs(out[1, :, :])
    phase_map = np.angle(out[1, :, :])

    if doPlot:
        plt.figure(figsize = [10, 4])
        plt.subplot(1,2,1); plt.imshow(power_map, cmap = 'binary'); plt.title('power'); plt.colorbar()
        if mask_plot: # Add an alpha mask to the plot
            plt.subplot(1,2,2)
            scviz.plot_image_with_alpha(phase_map, gaussian_filter(power_map, 6), vmin = -np.pi, vmax = np.pi)
        else:
            plt.subplot(1,2,2); plt.imshow(phase_map, cmap = 'gist_rainbow'); plt.title('phase'); plt.colorbar()

    return phase_map, power_map

def bin_2d(data, factor, dims = {'x':1, 'y':2, 't':0}):
    '''Takes as input a 3d array T x w x h and bins by a scalar factor
    Note: Returns it as t,y,x regardelss of input...
    '''
    data_reshaped = np.transpose(data, [dims['t'], dims['y'], dims['x']]); # Transpose to t, y, x
    t, y, x = data_reshaped.shape;
    print(t,y,x)

    if (y % factor != 0) or (x % factor != 0):
        if (y % factor != 0) and (x % factor != 0):
            data_reshaped = data_reshaped[:, :-(y % factor) , : -(x % factor)]
        elif (y % factor != 0):
            data_reshaped = data_reshaped[:, :-(y % factor) , :]
        elif (x % factor != 0):
            data_reshaped = data_reshaped[:, : , : -(x % factor)]
        t, y, x = data_reshaped.shape;

    out = data_reshaped.reshape([t, int(y/factor), factor, int(x/factor), factor]).mean(-1).mean(2)
    
    return out

def compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size = 1, doPlot = True):
    '''Compute field sign map from the phase maps of azimuth and elevation
    
    Arguments:
        ph_az {[type]} -- phase map azimuth
        ph_ev {[type]} -- phase map elevation
    '''
    #[dXev, dYev]= np.gradient(gaussian_filter(ph_ev, filt_size) )
    #[dXaz, dYaz]= np.gradient(gaussian_filter(ph_az, filt_size) )

    [dXev, dYev]= np.gradient(ph_ev)
    [dXaz, dYaz]= np.gradient(ph_az )

    angleEV = (dXev < 0) * np.pi + np.arctan(dYev / dXev);
    angleAZ = (dXaz < 0) * np.pi + np.arctan(dYaz / dXaz);

    field_sign = np.sin(angleEV - angleAZ);


    if doPlot == True:
        plt.figure(figsize = [10, 10])

        # Make a new double sided colormap.....
        colors1 = plt.cm.gist_rainbow(np.linspace(0., 1, 128))
        colors2 = plt.cm.gist_rainbow_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))  # combine them and build a new colormap
        double_side_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

        plt.subplot(3,2,1); plt.imshow(ph_az, cmap = double_side_cmap); plt.title('azimuth 2_sided_cmap'); plt.colorbar()
        plt.subplot(3,2,2); 
        scviz.plot_image_with_alpha(ph_az, gaussian_filter(pw_az, 6), vmin = -np.pi, vmax = np.pi); 
        plt.title('azimuth'); plt.colorbar()
        
        plt.subplot(3,2,3); 
        scviz.plot_image_with_alpha(ph_ev, gaussian_filter(pw_ev, 6), vmin = -np.pi, vmax = np.pi); 
        plt.title('elevation'); plt.colorbar()

        plt.subplot(3,2,4); plt.imshow(field_sign, cmap = 'bwr'); plt.title('field sign')

        plt.subplot(3,2,5); plt.imshow(pw_az); plt.title('power azimuth')
        plt.subplot(3,2,6); plt.imshow(pw_ev); plt.title('power elevation')


    return field_sign
