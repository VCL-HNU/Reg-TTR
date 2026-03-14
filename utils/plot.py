import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_deformation(deformation, z_slice, step, scale, opt):
    # assume we plot the xy plane
    X, Y, Z = np.meshgrid(np.arange(deformation.shape[0]), np.arange(deformation.shape[1]), np.arange(deformation.shape[2]), indexing='ij')
    X_slice = X[::step, ::step, z_slice]
    Y_slice = Y[::step, ::step, z_slice]

    X_def_slice = X_slice + scale*deformation[::step, ::step, z_slice,0]
    Y_def_slice = Y_slice + scale*deformation[::step, ::step, z_slice,1]

    plt.figure(figsize=(6, 6))
    ax = plt.axes()

    # Plot deformed grid
    ax.plot(Y_def_slice.T, X_def_slice.T, 'r', alpha=0.5)
    ax.plot(Y_def_slice, X_def_slice, 'r', alpha=0.5)
    ax.axis('off')

    os.makedirs(os.path.join(os.getcwd(), opt['dataset'],opt['model']), exist_ok=True)
    ax.get_figure().savefig(os.path.join(os.getcwd(), opt['dataset'],opt['model'], 'deformation.png'))
    plt.close()

def plot_img(img, z_slice, opt, filename):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    os.makedirs(os.path.join(os.getcwd(), opt['dataset'],opt['model']), exist_ok=True)
    
    plt.imsave(os.path.join(os.getcwd(), opt['dataset'],opt['model'], filename), img[:, :, z_slice], cmap='gray', format='png')
    plt.close()