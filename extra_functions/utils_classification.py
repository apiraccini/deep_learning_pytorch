# imports
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import torch

torch.set_printoptions(edgeitems=2)
torch.manual_seed(42)


def show(imgs):
    
    '''Show image from torch tensor batch (or slice of)'''

    #mean = m
    #std = s
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(figsize = (10,10), ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = img.numpy().transpose((1, 2, 0))
        #img = std * img + mean
        img = np.clip(img, 0, 1)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def check_params(model, verbose=False):

    '''Inspect model parameters'''

    modname = model.__class__.__name__
    print(f'Model: {modname}')

    tot = []
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            tot.append(param.numel())
            if verbose:
                print(f'\t{name:<14}--> size: {param.numel():6}, shape: {param.shape}')
    
    print(f'Total number of parameters: {sum(tot)}')


def plot_history(log, fig_height=6):

    '''Given metric history for a model, plots metrics on epochs'''

    fig, axs = plt.subplots(ncols=2, figsize=(1.618*fig_height*2, fig_height))

    for i, ax in enumerate(axs.ravel()):

        if i==0:
            metric='loss'
            metric_name = 'cross entropy'
            yt = np.array(log[metric]['train'])
            yv = np.array(log[metric]['val'])
        else:
            metric='acc'
            metric_name = 'accuracy'
            yt = np.stack([h.cpu().numpy() for h in log[metric]['train']])
            yv = np.stack([h.cpu().numpy() for h in log[metric]['val']])
        
        x = list(range(1, len(yv)+1))
        plotdata = pd.melt(pd.DataFrame({'epoch':x, 'train':yt, 'val':yv}), id_vars='epoch', value_name=metric, var_name='phase')
        
        sns.lineplot(ax=ax, x='epoch', y=metric, hue='phase', data=plotdata, palette=sns.color_palette('Dark2')[0:2])
        ax.set_xlabel('epochs',size=15)
        ax.set_ylabel(metric_name ,size=15)

    #fig.suptitle(f'{model.__class__.__name__}', fontsize=18)


def plot_models(models_dict, phase='val', metric='acc', fig_height = 6):

    '''
    Given a model dictionary with models and logs, plot metric of interest
    on epochs for every model
    '''

    d = pd.DataFrame()
    for k, v in models_dict.items():
        if metric=='acc':
            d[str(k)] = np.stack([h.cpu().numpy() for h in v[1][metric][phase]])
        else:
            d[str(k)] = v[1][metric][phase]

    d['epoch'] = list(range(1, len(d)+1))
    metric_name = 'cross entropy' if metric=='loss' else 'accuracy'
    plotdata = pd.melt(d, id_vars='epoch', value_name=metric_name, var_name='model')
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.618*fig_height, fig_height))
    sns.lineplot(ax=ax, x='epoch', y=metric_name,
                 hue='model', data=plotdata,
                 palette=sns.color_palette('Dark2')[0:(len(d.columns)-1)])
    ax.set_xlabel('epochs',size=15)
    ax.set_ylabel(metric_name ,size=15)
    ax.set_title('Model comparison',size=18)