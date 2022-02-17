# ml4s.py
# useful scripts and utilities for our course

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow,theme

# --------------------------------------------------------------------------
def draw_feed_forward(ax, num_node_list, node_labels=None, weights=None,biases=None, 
                      zero_index=True, weight_thickness=False, feed_forward=True,annotate=True):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): number of nodes in each layer.
    '''
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]
    y_list = 1.5 * np.arange(len(num_node_list))
    
    theme.NODE_THEME_DICT['nn.input'] = ["#E65933","circle","none"]
    theme.NODE_THEME_DICT['nn.hidden'] = ["#B9E1E2","circle","none"]
    theme.NODE_THEME_DICT['nn.output'] = ["#579584","circle","none"]
    
    shift = not zero_index

    if weight_thickness and (weights is None or weights == []):
        print("Can't show weight thicknesses if weights is empty!")
        weight_thickness = False

    # Determine if we will annotate the network
    if not annotate:
        node_labels = []
        weights = []
        biases = []

    # generate some default node labels
    if node_labels is None:
        node_labels = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ == 0:
                node_labels.append([f'$x_{j+shift}$' for j in range(nℓ)])
            else:
                node_labels.append([r'$a^{' + f'{ℓ}' + r'}_{' + f'{j+shift}' + r'}$' for j in range(nℓ)])

    # generate default bias labels
    if weights is None:
        weights = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ > 0:
                nℓm1 = num_node_list[ℓ-1]
                w_lab = np.zeros([nℓm1,nℓ],dtype='<U32')
                for k in range(nℓm1):
                    for j in range(nℓ):
                        w_lab[k,j] = r'$w^{' + f'{ℓ}' + r'}_{' + f'{k+shift}{j+shift}' + r'}$'
                weights.append(w_lab)
                    
    # generate some default weight labels
    if biases is None:
        biases = []
        for ℓ,nℓ in enumerate(num_node_list):
            if ℓ > 0:
                biases.append([r'$b^{' + f'{ℓ}' + r'}_{' + f'{j+shift}' + r'}$' for j in range(nℓ)])

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))

    # add labels
    if node_labels:
        for i,st in enumerate(seq_list):
            for j,node in enumerate(st):
                lab = node_labels[i][j]
                if isinstance(lab, float):
                    lab = f'{lab:.2f}'
                node.text(f'{lab}',fontsize=8)

    # add biases
    if biases:
        for i,st in enumerate(seq_list[1:]):
            for j,node in enumerate(st):
                x,y = node.pin(direction='right')
                lab = biases[i][j]
                if isinstance(lab, np.floating) or isinstance(lab,float):
                    lab = f'{lab:.2f}'
                ax.text(x+0.05,y,lab,fontsize=6)
    
    # do we want feed forward connections?
    if feed_forward:
        connection = '-->'
    else:
        connection = '--'
    eb = EdgeBrush(connection, ax,color='#58595b')
    
    ℓ = 0
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        
        if not weight_thickness or not annotate:
            c = connecta2a(st, et, eb)
            if len(weights)>0:

                if isinstance(weights[ℓ], list):
                    w = np.array(weights[ℓ])
                else:
                    w = weights[ℓ]

                if isinstance(w,np.ndarray):
                    w = w.flatten()

                for k,cc in enumerate(c):

                    factor = 1

                    # get the input and output neuron indices
                    #idx = np.unravel_index(k,weights[ℓ].shape)
                    idx = np.unravel_index(k,w.shape)
                    if idx[0]%2:
                        factor = -1

                    lab = w[k]
                    if isinstance(lab, np.floating) or isinstance(lab,float):
                        lab = f'{lab:.2f}' 

                    wtext = cc.text(lab,fontsize=6,text_offset=0.08*factor, position='top')
                    wtext.set_path_effects([path_effects.withSimplePatchShadow(offset=(0.5, -0.5),shadow_rgbFace='white', alpha=1)])
        
        else:
            # this is to plot individual edges with a thickness dependent on their weight
            # useful for convolutional networks where many weights are "zero"
            for i,cst in enumerate(st):
                for j,cet in enumerate(et):
                    if weights:
                        w = weights[ℓ]
                        
                        if np.abs(w[i,j]) > 1E-2:
                            eb = EdgeBrush(connection, ax,color='#58595b', lw=np.abs(w[i,j]))
                            e12 = eb >> (cst, cet)
                        
                            factor = 1
                            if i%2:
                                factor = -1
                            wtext = e12.text(f'{w[i,j]:.2f}',fontsize=6,text_offset=0.08*factor, position='top')
                            wtext.set_path_effects([path_effects.withSimplePatchShadow(offset=(0.5, -0.5),shadow_rgbFace='white', alpha=1)])

        ℓ += 1
        
# --------------------------------------------------------------------------
def draw_network(num_node_list,node_labels=None,weights=None,biases=None,zero_index=True, 
             weight_thickness=False, feed_forward=True, annotate=True):
    fig = plt.figure()
    ax = fig.gca()
    draw_feed_forward(ax, num_node_list=num_node_list, node_labels=node_labels,weights=weights, biases=biases, zero_index=zero_index, weight_thickness=weight_thickness, feed_forward=feed_forward,annotate=annotate)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.show()

# --------------------------------------------------------------------------
from IPython.core.display import HTML
def set_css_style(css_file_path):
   """
   Read the custom CSS file and load it into Jupyter.
   Pass the file path to the CSS file.
   """

   styles = open(css_file_path, "r").read()
   s = '<style>%s</style>' % styles     
   return HTML(s)

# --------------------------------------------------------------------------
def get_linear_colors(cmap,num_colors,reverse=False):
    '''Return num_colors colors in hex from the colormap cmap.'''
    
    from matplotlib import cm
    from matplotlib import colors as mplcolors

    cmap = cm.get_cmap(cmap)

    colors_ = []
    for n in np.linspace(0,1.0,num_colors):
        colors_.append(mplcolors.to_hex(cmap(n)))

    if reverse:
        colors_ = colors_[::-1]
    return colors_

# --------------------------------------------------------------------------
def random_psd_matrix(size,seed=None):
    '''Return a random positive semi-definite matrix with unit norm.'''
    
    np.random.seed(seed)
    
    A = np.random.randn(*size)
    A = A.T @ A
    A = A.T @ A
    A = A / np.linalg.norm(A)
    return A


# --------------------------------------------------------------------------
def feed_forward(aₒ,w,b,ffprime):
    '''Propagate an input vector x = aₒ through 
       a network with weights (w) and biases (b).
       Return: activations (a) and derivatives f'(z).'''
    
    a,df = [aₒ],[]
    for wℓ,bℓ in zip(w,b):
        zℓ = np.dot(a[-1],wℓ) + bℓ
        _a,_df = ffprime(zℓ)
        a.append(_a)
        df.append(_df)
        
    return a,df

# --------------------------------------------------------------------------
def backpropagation(y,a,w,b,df): 
    '''Inputs: results of a forward pass
       Targets     y: dim(y)  = batch_size ⨯ nL
       Activations a: dim(a)  = L ⨯ batch_size ⨯ nℓ
       Weights     w: dim(w)  = L-1 ⨯ nℓ₋₁ ⨯ nℓ
       Biases      b: dim(b)  = L-1 ⨯ nℓ
       f'(z)      df: dim(df) = L-1 ⨯ batch_size ⨯ nℓ
       
       Outputs: returns mini-batch averaged gradients of the cost function w.r.t. w and b
       dC_dw: dim(dC_dw) = dim(w)
       dC_db: dim(dC_db) = dim(b)
    '''
    
    num_layers = len(w)
    L = num_layers-1        
    batch_size = len(y)
    
    # initialize empty lists to store the derivatives of the cost functions
    dC_dw = [None]*num_layers
    dC_db = [None]*num_layers
    Δ = [None]*num_layers
    
    # perform the backpropagation
    for ℓ in reversed(range(num_layers)):
        
        # treat the last layer differently
        if ℓ == L:
            Δ[ℓ] = (a[ℓ] - y)*df[ℓ]
        else: 
            Δ[ℓ] = (Δ[ℓ+1] @ w[ℓ+1].T) * df[ℓ]
            
        dC_dw[ℓ] = (a[ℓ-1].T @ Δ[ℓ]) / batch_size
        dC_db[ℓ] = np.average(Δ[ℓ],axis=0)
        
    return dC_dw,dC_db

# --------------------------------------------------------------------------
def gradient_step(η,w,b,dC_dw,dC_db):
    '''Update the weights and biases as per gradient descent.'''
    
    for ℓ in range(len(w)):
        w[ℓ] -= η*dC_dw[ℓ]
        b[ℓ] -= η*dC_db[ℓ]
    return w,b

# --------------------------------------------------------------------------
def train_network(x,y,w,b,η,ffprime):
    '''Train a deep neural network via feed forward and back propagation.
       Inputs:
       Input         x: dim(x) = batch_size ⨯ n₁
       Target        y: dim(y) = batch_size ⨯ nL
       Weights       w: dim(w)  = L-1 ⨯ nℓ₋₁ ⨯ nℓ
       Biases        b: dim(b)  = L-1 ⨯ nℓ
       Learning rate η
       
       Outputs: the least squared cost between the network output and the targets.
       '''
    
    a,df = feed_forward(x,w,b,ffprime)
    
    # we pass a cycled a by 1 layer for ease of indexing
    dC_dw,dC_db = backpropagation(y,a[1:]+[a[0]],w,b,df)
    
    w,b = gradient_step(η,w,b,dC_dw,dC_db)
    
    return 0.5*np.average((y-a[-1])**2)

# --------------------------------------------------------------------------
def make_batch(n,batch_size,extent,func):
    '''Create a mini-batch from our inputs and outputs.
    Inputs:
    n0        : number of neurons in each layer
    batch_size: the desired number of samples in the mini-batch
    extent    : [min(xₒ),max(xₒ), min(x₁),max(x₁),…,min(x_{n[0]-1}),max(x_{n[0]-1})]
    func:     : the desired target function.
    
    Outputs: returns the desired mini-batch of inputs and targets.
    '''
    
    x = np.zeros([batch_size,n[0]])
    for i in range(n[0]):
        x[:,i] = np.random.uniform(low=extent[2*i],high=extent[2*i+1],size=[batch_size])

    y = func(*[x[:,j] for j in range(n[0])]).reshape(-1,n[-1])
    
    return x,y 

# --------------------------------------------------------------------------
#  tsne.py
# --------------------------------------------------------------------------

#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

#  Code Usage Notice on website: https://lvdmaaten.github.io/tsne/
#  You are free to use, modify, or redistribute this software in any way you want, 
#  but only for non-commercial purposes. The use of the software is at your own risk; 
#  the authors are not responsible for any damage as a result from errors in 
#  the software.

#  2021-04-20 Plotting code added, modified from F. Marquardt
#  https://github.com/FlorianMarquardt/machine-learning-for-physicists
from IPython.display import clear_output

# --------------------------------------------------------------------------
def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

# --------------------------------------------------------------------------
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

# --------------------------------------------------------------------------
def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

# --------------------------------------------------------------------------
def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, 
         do_animation=False, animation_skip_steps=10, max_iter = 1000):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        
        Added by F. Marquardt: do_animation==True will give you a graphical animation of
        the progress, use animation_skip_steps to control how often this will
        be plotted; max_iter controls the total number of gradient descent steps        
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)
    
    if do_animation: # added by FM/AGD
        costs = np.zeros(max_iter) # to store the cost values
        
    # Run iterations
    for c_iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if c_iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        
        if not do_animation: # added by FM/AGD: do not print if we are animating!
            # Compute current value of cost function
            if (c_iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                # modified to overwrite line
                print("Iteration %d: error is %f" % (c_iter + 1, C), end="           \r") 

        # Stop lying about P-values
        if c_iter == 100:
            P = P / 4.
            
        if do_animation:  # added by FM/AGD
            C = np.sum(P * np.log(P / Q)) # compute for every step, to store it in 'costs'
            costs[c_iter] = C
            if c_iter % animation_skip_steps ==0 :
                clear_output(wait=True)
                fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(10,5))
                ax[0].plot(costs)
                ax[1].scatter(Y[:,0],Y[:,1],color="#5E4Fa2")
                plt.show()            

    # Return solution
    return Y
