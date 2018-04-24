#!/usr/bin/env python

# Library of functions for analysis of flow cytometry data

import functools
import numpy as np
import pandas as pd

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

    for channel,minimum in min_dict.iteritems():
        
        df = df[df[channel]>minimum];
        
    return df

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