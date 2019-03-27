import glob
import sys

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile
from scipy.ndimage import gaussian_filter

# TODO: Make something that imports the data from individual tiffs
# Make something that imports the timeline files from mat
# Make something that computes the fourier and plots the retinoptic maps
# Maybe reuse the allen institute data...
# Put all of this under io

class matloader:

    def __init__(self):
        self.data = {}
        self.filename = None;
        self.loaded = False


    def _add_dtype_name(self, f, name):
        """Keep track of all dtypes and names in the HDF5 file using it."""
        global dtypes
        #print(dtypes)
        dtype = f.dtype
        if dtype.name in dtypes.keys():
            dtypes[dtype.name].add(name)
        else:
            dtypes[dtype.name] = set([name])
        return
        # if dtypes.has_key(dtype.name):
        #     dtypes[dtype.name].add(name)
        # else:
        #     dtypes[dtype.name] = set([name])
        # return

    def _string(self, seq):
        """Convert a sequence of integers into a single string."""
        out = ''.join([chr(a) for a in seq])
        return out

    def _recursive_dict(self, f, root=None, name='root'):
        """This function recursively navigates the HDF5 structure from
        node 'f' and tries to unpack the data structure by guessing their
        content from dtype, shape etc.. It returns a dictionary of
        strings, arrays and some leftovers. 'root' is the root node of the
        HDF5 structure, i.e. what h5py.File() returns.
        Note that this function works well on the Matlab7.3 datasets on
        which it was tested, but in general it might be wrong and it might
        crash. The motivation is that it has to guess the content of
        substructures so it might fail. One source of headache seems to be
        Matlab7.3 format that represents strings as array of 'uint16' so
        not using the string datatype. For this reason it is not possible
        to discriminate strings from arrays of integers without using
        heuristics.
        """
        
        global global_excluded_variables

        if root is None: root = f
        if hasattr(f, 'keys'):
            a = dict(f)
            if u'#refs#' in a.keys(): # we don't want to keep this
                del(a[u'#refs#'])
            for k in a.keys():
                #print k
                # FL: I added this in to skip moviedata to make loading faster...
                if k in global_excluded_variables:
                    continue
                else:
                    a[k] = self._recursive_dict(f[k], root, name=name+'->'+k)
            return a
        
        elif hasattr(f, 'shape'):
            if f.dtype.name not in ['object', 'uint16']: # this is a numpy array
                # Check shape to assess whether it can fit in memory
                # or not. If not recast to a smaller dtype!
                self._add_dtype_name(f, name)
                dtype = f.dtype
                if (np.prod(f.shape)*f.dtype.itemsize) > 2e9:
                    print("WARNING: The array", name, "requires > 2Gb")
                    if f.dtype.char=='d':
                        print("\t Recasting", dtype, "to float32")
                        dtype = np.float32
    #				else:
    #					raise MemoryError
                return np.array(f, dtype=dtype).squeeze()
            elif f.dtype.name in ['uint16']: # this may be a string for Matlab
                self._add_dtype_name(f, name)
                try:
                    return self._string(f)
                except ValueError: # it wasn't...
                    print("WARNING:", name, ":")
                    print("\t", f)
                    print("\t CONVERSION TO STRING FAILED, USING ARRAY!")
                    tmp = np.array(f).squeeze()
                    print("\t", tmp)
                    return tmp
                pass
            elif f.dtype.name=='object': # this is a 2D array of HDF5 object references or just objects
                self._add_dtype_name(f, name)
                container = []
                for i in range(f.shape[0]):
                    for j in range(f.shape[1]):
                        if str(f[i][j])=='<HDF5 object reference>': # reference follow it:
                            container.append(self._recursive_dict(root[f[i][j]], root, name=name))
                        else:
                            container.append(np.array(f[i][j]).squeeze())
                try:
                    return np.array(container).squeeze()
                except ValueError:
                    print("WARNING:", name, ":")
                    print("\t", container)
                    print("\t CANNOT CONVERT INTO NON-OBJECT ARRAY")
                    return np.array(container, dtype=np.object).squeeze()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return

    def loadmat_h5(self, filename, excluded_variables = []):
        '''
        'Master code for loading int the .mat filename'
        Can not load certain variables by including a list under the keyword argument, excluded)_variables, useful when certain variables are very large and take too long to load
        '''

        print('Loading mat data from %s' %filename)

        f = h5py.File(filename, mode='r')
        global dtypes, global_excluded_variables
        global_excluded_variables = excluded_variables
        dtypes = {}; # Reset dtypes each time...
        
        #print outputName
        
        # Output everything from the structure
        self.data = self._recursive_dict(f)
        self.filename = filename
        self.loaded = True
        print('Finished Loading.')

    def summary(self):
        '''Lists a summary of the given outputs'''
        #print("+++++++++++++++++++++++++++++++++++++++")
        #%print("Variable name                     Type")
        print('\n'); print('Summary of loaded data'); print('\n')
        print('{:<40s}{:<30s}{:<0s}'.format('Variable name','Type', 'Shape'))
        print('+'*100);
        for x in self.data.keys():

            if isinstance(self.data[x], np.ndarray):
                print('{:<40s}{:<30s}{:<0s}'.format(str(x),str(type(self.data[x])),  str(self.data[x].shape) ) )
            else:
                print('{:<40s}{:<30s}'.format(str(x),str(type(self.data[x])) ) )

            #print('%s   %s' % (x, type(self.data[x]) ) )

def export_tiffs(data, outDir='', dims = {'x':2, 'y':1, 't':0} ):
    ''' dims takes as input the x y t dimensions, e.g. if input is T x Y x X, then it is [2, 1, 0]'''

    if outDir == '':
        raise NotADirectoryError
    else:
        tifffile.imsave(outDir, np.transpose(data, [dims['t'], dims['y'], dims['x']]).astype('single') )
