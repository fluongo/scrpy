# A class for examining neural networks
''' A class for instantiatign and passing data through 
'''

import keras
from keras import applications
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

class base:
    ''' Class for instantiating a given network then getting the activations of a given layer'''
    def __init__(self, network_name):
        self.data = []
        self.imgs = [] 
        self.img_size = []
        self.n_images = 0
        self.nn = network_name
        self.model = []
        self.outputs = []

    def reset_img(self):
        '''Resets the image list'''
        self.imgs = []

    def add_img(self, img):
        '''Write image into the queue'''

        if len(img.shape) == 2: # monocrhome but we will treat as color by makign 3 identical channels
            img = np.dstack([img for i in range(3)])

        if len(self.imgs) == 0:
            self.img_size = img.shape
        
        if img.shape == self.img_size:
            self.imgs.append(img)
        else:
            raise('You are adding images of different sizes, should be same size')
        
        self.n_images = len(self.imgs)

    def resize_images(self, ds_factor = 2):
        """resize_images [summary]
        
        Keyword Arguments:
            ds_factor {int} -- scaling factor to downsample by (default: {2})
        """
        # Do a downsample of the images
        new_imgs = []

        # Only resize for the first 2 dimensions...
        self.img_size = (int(self.img_size[0]/float(ds_factor)), int(self.img_size[1]/float(ds_factor)), self.img_size[2])
        
        for xx in self.imgs:
            new_imgs.append(scipy.misc.imresize(xx, self.img_size))
        self.imgs = new_imgs

    def instantiate_network(self):
        if self.nn == 'VGG16':
            self.model = applications.VGG16(weights='imagenet', include_top=True, input_shape=(self.img_size[0], self.img_size[0], 3))

    def list_layers(self):
        """list_layers [summary]
        
        Returns:
            [list] -- list of the layer names
        """

        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        return [output.name for output in outputs]

    def get_activations(self, layer_name=None):
        """get_activations [summary]
        
        Keyword Arguments:
            layer_name {string} -- Which layer to compute the activations from in the NN
        """

        step_size = 100
        nSteps = np.ceil(self.n_images/float(step_size)).astype('int')

        print('.......Computing activations for %d images.......' % self.n_images)
        print('.......Performing in %d steps so as to not run out of memory....................' % nSteps)

        act_list = []

        for kk in range(nSteps):
            print('.............On step %d out of %d' % (kk+1, nSteps))
            activations = []
            inp = self.model.input

            model_multi_inputs_cond = True
            if not isinstance(inp, list):
                # only one input ... wrap it in a list.
                inp = [inp]
                model_multi_inputs_cond = False

            outputs = [layer.output for layer in self.model.layers if
                    layer.name == layer_name or layer_name is None]  # all layer outputs
            
            dict_names = [layer.name for layer in self.model.layers if
                    layer.name == layer_name or layer_name is None]  # all layer outputs

            #print(outputs)

            # we remove the placeholders (Inputs node in Keras). Not the most elegant though..
            outputs = [output for output in outputs if 'input_' not in output.name]

            funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
            
            if model_multi_inputs_cond:
                list_inputs = []
                list_inputs.extend(self.imgs[ kk*step_size: min( (kk+1)*step_size , self.n_images )])
                list_inputs.append(0.)
            else:
                list_inputs = [self.imgs[ kk*step_size: min( (kk+1)*step_size , self.n_images ) ], 0.]

            # Learning phase. 0 = Test mode (no dropout or batch normalization)
            # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]        
            act_list.append([func(list_inputs)[0] for func in funcs])

        # Recombine the activations
        if len(act_list) == 1: # Only one iteration
            activations = act_list[0]
            #print(len(activations))
        else:
            activations = []
            for layer_no in range(len(outputs)): # Each layer
                activations.append(np.vstack(act_list[i][layer_no] for i in range(nSteps)))

        #layer_names = [output.name for output in outputs]

        # Zip them with corresponding names....
        self.activations = dict(zip(dict_names, activations))

        # Do this to clear off the funcs on GPU so as to be able to use the same names next time...
        K.clear_session()

        print('................DONE COMPUTING ACTIVATIONS...............')        

    def display_activations(self, layer_name = None, image_number = None, same_colorscale=False):
        """Code for visualizing the activations of a given channel across the various layers"""

        if (layer_name == None) or (image_number == None):
            print(self.activations.keys())
            raise('Must specify which layer or image you want to plot...')

        [n_images, xx_lim, yy_lim, n_channels] = self.activations[layer_name].shape

        # Determine the size of the block for the n_channels by 
        grid_size = np.ceil(np.sqrt(n_channels))


        if same_colorscale:
            vmax = np.percentile(self.activations[layer_name][image_number, :, :, :], 99)

        plt.figure(figsize = [10, 10])
        for i in range(n_channels):
            plt.subplot(grid_size, grid_size, i+1)
            if same_colorscale:
                plt.imshow(self.activations[layer_name][image_number, :, :, i], vmin = 0, vmax = vmax);  
            else:
                plt.imshow(self.activations[layer_name][image_number, :, :, i], vmin = 0); 
            plt.title(str(i))
            plt.axis('off')

        plt.suptitle(layer_name)
        plt.show()