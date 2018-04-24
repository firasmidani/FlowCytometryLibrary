    #!/usr/bin/env python

# Library of functions for analysis of flow cytometry data

##fitting 2D guassian (single component)

import os
import imp
import sys
import itertools
import scipy as sp
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import FlowCytometryTools as fct

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from FlowCytometryTools import FCMeasurement, ThresholdGate
from scipy import linalg
from sklearn import mixture

def makeContourPlot(data,fit):
        
    Z = {}
    
    mins_ = data.min().values - 10; #print mins_
    maxs_ = data.max().values + 10; #print maxs_
        
    delta=1;
    x = np.arange(mins_[0],maxs_[0],delta)
    y = np.arange(mins_[1],maxs_[1],delta)
    X, Y = np.meshgrid(x,y)
    
    #fit = GaussianMixture(n_components=1).fit(data.values)
    
    for ii in range(6):
        covar_diag = fit.covariances_[ii][1,0]; #print covar
        if covar_diag<0:
            covar_diag = 0
        Z[ii] = mlab.bivariate_normal(X = X,
                                      Y = Y,
                                      mux = fit.means_[ii,0],
                                      muy = fit.means_[ii,1],
                                      sigmax = np.sqrt(fit.covariances_[ii][0,0]), 
                                      sigmay = np.sqrt(fit.covariances_[ii][1,1]),
                                      sigmaxy = covar_diag
                                     )
    return X,Y,Z

##fitting 2D guassian (single component)

def fitGuassianComponent(data):
        
    mins_ = data.min().values - 10; #print mins_
    maxs_ = data.max().values + 10; #print maxs_
        
    delta=1;
    x = np.arange(mins_[0],maxs_[0],delta)
    y = np.arange(mins_[1],maxs_[1],delta)
    X, Y = np.meshgrid(x,y)
    
    fit = GaussianMixture(n_components=1).fit(data.values)
    
    covar_diag = fit.covariances_[0][1][0];
    if covar_diag<0:
            covar_diag=0;
    
    Z = mlab.bivariate_normal(X=X,
                              Y=Y,
                              mux=fit.means_[0][0],
                              muy=fit.means_[0][1],
                              sigmax=np.sqrt(fit.covariances_[0][0][0]),
                              sigmay=np.sqrt(fit.covariances_[0][1][1]),
                              sigmaxy=covar_diag)
    return X,Y,Z,fit

def fitGuassianComponent_ND(data):
    
    n_features = data.shape[1];
    
    mins_ = data.min().values - 10; #print mins_
    maxs_ = data.max().values + 10; #print maxs_
        
    delta=1;
    x = np.arange(mins_[0],maxs_[0],delta)
    y = np.arange(mins_[1],maxs_[1],delta)
    X, Y = np.meshgrid(x,y)
    
    fit = GaussianMixture(n_components=1).fit(data.values)
    
    covar_diag = fit.covariances_[0][1][0];
    if covar_diag<0:
            covar_diag=0;
    
    Z = mlab.bivariate_normal(X=X,
                              Y=Y,
                              mux=fit.means_[0][0],
                              muy=fit.means_[0][1],
                              sigmax=np.sqrt(fit.covariances_[0][0][0]),
                              sigmay=np.sqrt(fit.covariances_[0][1][1]),
                              sigmaxy=covar_diag)
    return X,Y,Z,fit


def SummaryPlot(wellid,data_dict,mapping,colors=(0,0,0,0.75),
                dtype='H',
                fit=False,
                channel_default=True,
                subsample=False):
 
    if channel_default:
        rfp_ch = 'RFP';
        gfp_ch = 'GFP';
    else:
        rfp_ch = 'dsRed/txRed'
        gfp_ch = 'GFP/FITC'


    fig,axes = plt.subplots(1,3,figsize=[15,4])

    fit_dict = {};
    
    if isinstance(wellid,str):
        wellid = [wellid];
        
    if isinstance(wellid,str):
        color = [colors];
        
    if isinstance(dtype,str):
        dtype = [dtype]

    varbs = np.ravel([['%s-%s' % (rfp_ch,dt),'%s-%s' % (gfp_ch,dt), 'FSC-%s' % dt, 'SSC-%s' % dt] for dt in dtype]); #print varbs
    varbs = np.ravel([['%s-%s' % (rfp_ch,dt),'%s-%s' % (gfp_ch,dt)] for dt in dtype]); #print varbs
    #varbs = np.ravel(['dsRed/txRed-A','GFP/FITC-A','FSC-A','SSC-A']); #print varbs

    print varbs

    cnt = 0;
    all_data = []
    for wid in wellid:
    
        #wname = mapping.loc[wid,'Name'];
        #data = data_dict[wid].loc[:,['dsRed/txRed-%s' % dtype,'GFP/FITC-%s' % dtype]];
        if subsample:
            data = data_dict[wid].sample(n=10000,replace=False).loc[:,varbs]
        else:
            data = data_dict[wid].loc[:,varbs];
        #data = data[(data>0).all(1)]
        rfp = data.loc[:,'%s-%s' % (rfp_ch,dtype[0])];
        gfp = data.loc[:,'%s-%s' % (gfp_ch,dtype[0])]
        color = colors[cnt]
    
        ## rfp histogram
    
        ax = axes[1];
    
        ax.hist(rfp,bins=100,lw=0,color=color[0:4],histtype='stepfilled',label=wid) #(1,0,0,0.75)
        ax.set_xlabel('%s-%s' % ('RFP',dtype[0]),fontsize=16);
    
        ## gfp histogram
    
        ax = axes[0];
        ax.hist(gfp,bins=100,lw=0,color=color[0:4],histtype='stepfilled',label=wid) #(0,0.5,0,0.75)
        ax.set_xlabel('%s-%s' % ('GFP',dtype[0]),fontsize=16);
    
        ## scatter plot
    
        #ax = axes[2];
        #sns.kdeplot(rfp,gfp, ax=axes[2])
        #ax.set_ylabel('%s-%s' % (dtype[0],'GFP'),fontsize=16);
        #ax.set_xlabel('%s-%s' % (dtype[0],'RFP'),fontsize=16);
        
        #X,Y,Z,fit_dict[wid] = fitGuassianComponent_ND(data);
        if fit:
            X,Y,Z,fit_dict[wid] = fitGuassianComponent(data);
        
        #ax.contour(X,Y,Z,cmap=color[-1])
        axes[0].set_ylim([0,3000])
        axes[1].set_ylim([0,3000])

        print data.shape
        all_data.append(data)

        cnt+=1;

        #data = pd.concat(all_data); print data.shape
        if fit:
            sns.kdeplot(data.loc[:,'%s-%s' % (rfp_ch,dtype[0])].values,data.loc[:,'%s-%s' % (gfp_ch,dtype[0])].values,
                        kind="kde",cmap=color[-1],shade=False,gridsize=100,ax=axes[2])
            axes[2].set_ylabel('GFP-A',fontsize=16)
            axes[2].set_xlabel('RFP-A',fontsize=16)
            axes[2].set_ylim([0,1000])
            axes[2].set_xlim([0,1000])

    ## shared figure parameters
    
    [[ii.set(fontsize=16) for ii in ax.get_xticklabels()+ax.get_yticklabels()] for ax in axes]
    [ax.set_ylabel('Count',fontsize=16) for ax in axes[0:2]];
    
    axes[0].legend(loc=2,fontsize=13,frameon=True,
        bbox_to_anchor=(15,3000),
        bbox_transform=axes[0].transData)
    #axes[0].set_ylim([0,3500])
    axes[0].set_xlim([0,1000])

    axes[1].legend(loc=2,fontsize=13,frameon=True,
        bbox_to_anchor=(15,3000),
        bbox_transform=axes[1].transData)
    #axes[1].set_ylim([0,3500])
    axes[1].set_xlim([0,1000])

    plt.subplots_adjust(wspace=0.4)
    
    return ax,fit_dict