#!/usr/bin/env python

# Firas Said Midani
# Start date: 2018-04-24
# Final date: 2018-04-24

# DESCRIPTION Library of functions for analysis of flow cytometry data

# TABLE OF CONTENTS
#
#|-- Plotting & Visualization
#    |-- prettyJointPlot
#
#|-- Data transformations
#    |-- addPseudoCount
#    |-- joint MinVoltageFilter
#    |-- minVoltageFilter
#
#|-- Syntax reductions
#    |-- conjuction
#

# IMPORT NECESSARY LIBRARIES

import functools
import numpy as np
import pandas as pd
import seaborn as sns

# SET PARAMETERS & STYLES

sns.set_style('whitegrid');

# FUNCTIONS

def addPseudoCount(df,pc=1e-3):
    '''
    addPseudoCount sets a non-zero floor to all values in a dataframe.

    Keyword arguments:
    df -- pandas.dataframe where rows are events and columns are flow cytometry variables (e.g. channels). Values should be int or float.
    pc -- Pseudo-count. Default is 0.001

    Returns pandas.DataFrame
    '''

    df = df.applymap(lambda x: [pc if x<=0 else x][0])

    return df

def conjunction(*conditions):
    '''
    conjuncation conjuncts multiple (more than two) conditions (input arguments)

    Keyword arguments (specific use but can be generalizable):
    condition -- pandas.Series of True or False values

    Returns pandas.DataFrame where True indicates all conditions met for sample in row

    Notes: 
    np.logical_and can conjugate two arguments only. 
    Functools enables us to handle more than two. 
    See https://stackoverflow.com/questions/13611065/efficient-way-to-apply-multiple-filters-to-pandas-dataframe-or-series
    '''
    return functools.reduce(np.logical_and,conditions)

def jointMinVoltageFilter(df,min_dict):
    '''
    jointMinVoltageFilter removes events based on joint minimal bounds for desired channels. 
    
    Kewyord arguments:
    df -- pandas.dataframe where rows are events and columns are flow cytometry variables (e.g. channels)
    min_dict -- Dictionary of channels as keys and minima as values.

    Returns pandas.DataFrame.
    '''

    conditions = [df[df[channel]>minimum] for channel,medium in min_dict.iteritems()];

    return df[conjunction(*conditions)]
        
def minVoltageFilter(df,min_dict):
    '''
    minVoltageFinder removes events based on minimum bounds for desired channels. 
    
    Kewyord arguments:
    df -- pandas.dataframe where rows are events and columns are flow cytometry variables (e.g. channels)
    min_dict -- Dictionary of channels as keys and minima as values.

    Returns pandas.DataFrame.
    '''

    for channel,minimum in min_dict.iteritems():
        
        df = df[df[channel]>minimum];
        
    return df

def prettyJointPlot(df):
    '''
    prettyJointPlot draws a plot of two variables with bivariate core and adjoining univariate histograms.

    Keyword arguments:
    df -- pandas.dataframe where rows are events and columns are *two* flow cytometry variables (e.g. channels)

    Returns seaborn plot.
    '''

    x = df.iloc[:,0]; 
    y = df.iloc[:,1];

    jp = sns.jointplot(x=x,y=y,
                       kind="kde",stat_func=None,
                       size=7,ratio=3,space=9,color="black");

    if (np.max(np.max(df))<1000) and (np.min(np.min(df))>0): 
        
        jp.ax_joint.set_xlim([0,1000]);
        jp.ax_joint.set_xlim([0,1000]);

    jp.ax_joint.set_xlabel(xy[0],fontsize=30);
    jp.ax_joint.set_ylabel(xy[1],fontsize=30);
    jp.ax_joint.tick_params(labelsize=20);

    jp.ax_joint.collections[0].set_alpha(0); # what is this?

    plt.close(jp.fig) # do not display figure

    return jp

