#!/usr/bin/env python

# Firas Said Midani
# Start date: 2018-04-24
# Final date: 2018-05-03

# DESCRIPTION Library of functions for analysis of flow cytometry data

# TABLE OF CONTENTS
#
#
#|-- Data transformations
#    |-- addPseudoCount
#    |-- jointMinVoltageFilter
#    |-- minVoltageFilter
#
#|-- FCS Data Processing
#    |-- dataFromFCS
#    |-- readFCS
#
#|-- Plotting & Visualization
#    |-- prettyJointPlot

#|-- Syntax reductions
#    |-- conjuction
#
#|-- System organization
#    |-- listFiles
#    |-- sampleNumber
#

# IMPORT NECESSARY LIBRARIES

import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from FlowCytometryTools import FCMeasurement

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

    Questions:
    would one or two conditions be fine?
    '''
    return functools.reduce(np.logical_and,conditions)

def dataFromFCS(fcs,ZeroFloor=True):
    '''
    dataFromFCS 

    Keyword arguments:
    fcs -- data container with format of FlowCytometryTools.core.containers.FCMeasurement
    ZeroFloor -- Boovelean on whether to conver negative infinity values to zero.

    Returns 
    fcs -- pandas.DataFrame where rows are flow events and columsn are flow channels
    '''

    # david lab uses MACSQuant VYB flow-cytometery machine
    vyb_channel_dict = {
        'dsRed/txRed-A':'RFP-A',
        'dsRed/txRed-H':'RFP-H',
        'dsRed/txRed-W':'RFP-W',
        'GFP/FITC-A':'GFP-A',
        'GFP/FITC-H':'GFP-H',
        'GFP/FITC-W':'GFP-W',
        'PI/LSS-mKate-A':'PI-A',
        'PI/LSS-mKate-H':'PI-H',
        'PI/LSS-mKate-W':'PI-W'
    }

    reporter_channel_dict = {
        'Y2-A':'RFP-A',
        'Y2-H':'RFP-H',
        'Y2-W':'RFP-W',
        'B1-A':'GFP-A',
        'B1-H':'GFP-H',
        'B1-W':'GFP-W',
        'B2-A':'PI-A',
        'B2-H':'PI-H',
        'B2-W':'PI-W'
    }

    if 'Y2-A' in fcs.data.keys():

        columns_dict = reporter_channel_dict;

    elif 'dsRed/txRed-A' in fcs.data.keys():

        columns_dict = vyb_channel_dict


    dataFromFCS = fcs.data

    if ZeroFloor: 
        dataFromFCS = dataFromFCS.replace(-np.inf,0)

    dataFromFCS = dataFromFCS.rename(columns=columns_dict);

    return dataFromFCS

def jointMinVoltageFilter(df,min_dict):
    '''
    jointMinVoltageFilter removes events based on joint minimal bounds for desired channels. 
    
    Kewyord arguments:
    df -- pandas.dataframe where rows are events and columns are flow cytometry variables (e.g. channels)
    min_dict -- Dictionary of channels as keys and minima as values.

    Returns pandas.DataFrame.
 
    Questions:
    would one or two conditions be fine?
   '''

    conditions = [df[channel]>minimum for channel,minimum in min_dict.iteritems()];

    return df[conjunction(*conditions)]

def listFiles(directory,suffix='.fcs',removeSuffix=True):
    '''
    listFiles identifies files in a directory with a certain suffix and can removes suffix

    Keyword arguments
    directory -- string of directory with either absolute or relative path
    suffix -- string for suffix in filenames 
    removeSuffix -- strips suffix from filenames in list

    Returns dictionary with filenames as keys and filepaths as vlaues
    '''

    list_files = os.listdir(directory);
    list_files = [ff for ff in list_files if ff.endswith(suffix)]

    if removeSuffix:
        list_files = [lf.strip('.fcs') for lf in list_files];
        list_paths = ['%s/%s%s' % (directory,ff,suffix) for ff in list_files]
    else:
        list_paths = ['%s/%s' % (directory,ff) for ff in list_files]

    out_list = dict(zip(list_files,list_paths))

    return out_list
 
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
                       size=7,ratio=3,space=0,color="black");

    if (np.max(np.max(df))<1000) and (np.min(np.min(df))>0): 
        
        jp.ax_joint.set_xlim([0,1000]);
        jp.ax_joint.set_xlim([0,1000]);

    jp.ax_joint.set_xlabel(x.name,fontsize=30);
    jp.ax_joint.set_ylabel(y.name,fontsize=30);
    jp.ax_joint.tick_params(labelsize=20);

    jp.ax_joint.collections[0].set_alpha(0); # what is this?

    plt.close(jp.fig) # do not display figure

    return jp

def readFCS(filepath):
    '''
    readFCS uses FlowCytometryTools package to open an FCS file

    Keyword arguments:
    filepath -- filename including absolute or relative path. 

    Returns 
    fcs -- data container with format of FlowCytometryTools.core.containers.FCMeasurement
    '''

    # extract filename from path
    if '/' in filepath:
        filename = filepath.split('/')[-1][:-4]
    else: 
        filename = filepath

    fcs = FCMeasurement(ID=filepath,datafile=filepath)

    return fcs


def sampleNumber(filename):
    '''
    sampleNumber extracts flow cytometry sample number from filenmae

    filename -- generic file name by MACSQuant VYB

    Returns integer
    '''

    return int(filename.split('.')[-1])
