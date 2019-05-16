#!/usr/bin/env python

# Firas Said Midani
# Start date: 2018-04-24
# Final date: 2019-01-17

# DESCRIPTION Library of functions for analysis of flow cytometry data

# TABLE OF CONTENTS
#
#
#|-- Data transformations
#    |-- addPseudoCount
#    |-- countEvents
#    |-- jointMinVoltageFilter
#    |-- minVoltageFilter
#    |-- relativeAbundance
#    |-- sampleData
#
#
#|-- Array mmanipulations
#    |-- findOverlap
#    |-- findIntersection
#    |-- getSignalPDF
#
#
#|-- FCS Data Processing
#    |-- dataFromFCS
#    |-- readFCS
#    |-- getEvents
#    |-- getGates
# 
#
#|-- Plotting & Visualization
#    |-- annotatePlot
#    |-- initializeGrid
#    |-- plotGates
#    |-- plotMixedTimeSeries
#    |-- prettyJointPlot
#    |-- prettyKDEPlot
#
#
#|-- Syntax reductions
#    |-- conjuction
#
#
#|-- System organization
#    |-- listFiles
#    |-- sampleNumber
#    |-- getFormattedTime
#
#

# IMPORT NECESSARY LIBRARIES

import os
import functools
import itertools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde as gkde
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

def annotatePlot(ax,abundance):
    '''
    annotatePlot adds species and relative abundance indicator text to each quadrant

    Keyword arguments:
    ax -- matplot.axes._subplots.AxesSubplot
    abundance -- list of relativea abundanc values (size of four)

    Returns None
    '''

    ax.text(x=100,y=180,s='Bo',ha='center',va='top',fontsize=20,color='navy')
    ax.text(x=900,y=180,s='Bf',ha='center',va='top',fontsize=20,color='red')
    ax.text(x=100,y=980,s='Bt',ha='center',va='top',fontsize=20,color='green')
    ax.text(x=900,y=980,s='Bv',ha='center',va='top',fontsize=20,color='goldenrod')

    ax.text(x=100,y=20,s='%0.2f' % abundance[0],ha='center',va='bottom',fontsize=20,color='navy')
    ax.text(x=900,y=20,s='%0.2f' % abundance[1],ha='center',va='bottom',fontsize=20,color='red')
    ax.text(x=100,y=820,s='%0.2f' % abundance[2],ha='center',va='bottom',fontsize=20,color='green')
    ax.text(x=900,y=820,s='%0.2f' % abundance[3],ha='center',va='bottom',fontsize=20,color='goldenrod')

    return None

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

def countEvents(df,rfp=525,gfp=525):
    '''
    countEvents computes number of events in each quadrant of a 2-dimensional flow cytometry plot 
    based on two user-defined gates

    Keyword arguments:
    df -- pandas.DataFrame where variables include RFP-A and GFP-A
    rfp -- INT voltage gate for RFP-A signal
    gfp -- INT volutage gate for GFP-A signal

    Returns
    count -- list of event counts for BO, BF, BT, and BV, respectively
    '''    
    
    count = [float(df[(df['RFP-A']<rfp) & (df['GFP-A']<gfp)].shape[0]), 
             float(df[(df['RFP-A']>=rfp) & (df['GFP-A']<gfp)].shape[0]),
             float(df[(df['RFP-A']<rfp) & (df['GFP-A']>=gfp)].shape[0]),
             float(df[(df['RFP-A']>=rfp) & (df['GFP-A']>=gfp)].shape[0])]
    
    return count


def countEventsDynamic(df,gates):
    '''

    Args

    df -- pandas.DataFrame with variables of RFP-A & GFP-A
    gates -- two-layered dictionary with first layer rof GFP/RFP and second layer is species pairs (total of 4 values)

    '''

    h_1 = gates['GFP'][('BO','BT')];
    h_2 = gates['GFP'][('BF','BV')];
    v_1 = gates['RFP'][('BO','BF')];
    v_2 = gates['RFP'][('BT','BV')];

    if ((v_2 > v_1) & (h_2 > h_1)): 

        BO = float(df[(df['RFP-A'] < v_1) & (df['GFP-A'] < h_1)].shape[0]);
        BV = float(df[(df['RFP-A'] > v_2) & (df['GFP-A'] > h_2)].shape[0]);

        BF_BT = float(df[((df['RFP-A']>v_1) & (df['RFP-A']<v_2)) & ((df['GFP-A']>h_1) & (df['GFP-A']<h_2))].shape[0]);

        BF = float(df[(df['RFP-A'] > v_1) & (df['GFP-A'] < h_2)].shape[0]) - BF_BT
        BT = float(df[(df['RFP-A'] < v_2) & (df['GFP-A'] > h_1)].shape[0]) - BF_BT

        BO_BV = 0;
        BX = 0;

    elif ((v_1 > v_2) & (h_1 > h_2)):
        
        BF = float(df[(df['RFP-A'] > v_1) & (df['GFP-A'] < h_2)].shape[0]);
        BT = float(df[(df['RFP-A'] < v_2) & (df['GFP-A'] > h_1)].shape[0]);

        BO_BV = float(df[((df['RFP-A']>v_2) & (df['RFP-A']<v_1)) & ((df['GFP-A']>h_2) & (df['GFP-A']<h_1))].shape[0]);

        BO = float(df[(df['RFP-A'] < v_1) & (df['GFP-A'] < h_1)].shape[0]) - BO_BV
        BV = float(df[(df['RFP-A'] > v_2) & (df['GFP-A'] > h_2)].shape[0]) - BO_BV

        BF_BT = 0;
        BX = 0;

    elif ((v_2 > v_1) & (h_1 > h_2)):

        BO = float(df[(df['RFP-A'] < v_1) & (df['GFP-A'] < h_1)].shape[0]);
        BF = float(df[(df['RFP-A'] > v_1) & (df['GFP-A'] < h_2)].shape[0]);
        BT = float(df[(df['RFP-A'] < v_2) & (df['GFP-A'] > h_1)].shape[0]);
        BV = float(df[(df['RFP-A'] > v_2) & (df['GFP-A'] > h_2)].shape[0]);

        BF_BT = 0;
        BO_BV = 0;

        BX = float(df[((df['RFP-A']>v_1) & (df['RFP-A']<v_2)) & ((df['GFP-A']<h_1) & (df['GFP-A']>h_2))].shape[0]);

    elif ((v_1 > v_2) & (h_2 > h_1)):

        BO = float(df[(df['RFP-A'] < v_1) & (df['GFP-A'] < h_1)].shape[0]);
        BF = float(df[(df['RFP-A'] > v_1) & (df['GFP-A'] < h_2)].shape[0]);
        BT = float(df[(df['RFP-A'] < v_2) & (df['GFP-A'] > h_1)].shape[0]);
        BV = float(df[(df['RFP-A'] > v_2) & (df['GFP-A'] > h_2)].shape[0]);

        BF_BT = 0;
        BO_BV = 0;

        BX = float(df[((df['RFP-A']>v_1) & (df['RFP-A']<v_2)) & ((df['GFP-A']>h_1) & (df['GFP-A']<h_2))].shape[0]);

    count = [BO,BF,BT,BV,BO_BV,BF_BT,BX]

    count = pd.DataFrame(count, index=['BO','BF','BT','BV','BO_BV','BF_BT','BX'])

    return count

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
        'PI/LSS-mKate-W':'PI-W',
        'CFP/VioBlue-A':'SYTO-A',
        'CFP/VioBlue-H':'SYTO-H',
        'CFP/VioBlue-W':'SYTO-W'
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
        'B2-W':'PI-W',
        'V1-A':'SYTO-A',
        'V1-H':'SYTO-H',
        'V1-W':'SYTO-W'
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

def findIntersection(x_1,x_2,y_1,y_2,interval=1e-4):
    '''
    fundIntersection find the region of overlap between two numerical arrays, 
    creates a new array with x-values at an interval that user-specified,
    interpolates both arrays using the new x-values, then 
    finds the intersection between those interpolated and x-synchronized arrays.

    Keyword arguments:
    x_1 -- numpy.array of x-values for first array
    x_2 -- numpy.array of x-values for second arrayter
    y_1 -- numpy.array of y-values for first array
    y_2 -- numpy.array of y-values for second array

    Returns two INT for x- and y-coordinate of interesection, respectively
    '''

    l_x, r_x = findOverlap(x_1,x_2); #print l_x,r_x
    
    new_x = np.arange(l_x,r_x,interval)

    f1 = sp.interpolate.interp1d(x_1,y_1);
    f2 = sp.interpolate.interp1d(x_2,y_2);

    y_1_new = f1(new_x)
    y_2_new = f2(new_x)

    idx = np.argwhere(np.diff(np.sign(y_1_new-y_2_new))).flatten()
    
    return new_x[idx],y_1_new[idx]

def findOverlap(arr_1,arr_2):
    '''
    findOverlay identifies overlap between arrays. For example if arr_1 is [1,2,3,4,5] and arr_2 is [4,5,6,7]. 
    The left boundary (low end) is 4 and the right boundary (high end) is 5. 

    Keyword arguments:
    arr_1 -- numpy.array
    arr_2 -- numpy.array


    Notes:
    * order of arrays does not matter. Function checks for which array has a lower start. 

    Potential improvements:
    * what if one array is longer than the other (i.e. it has a lower start and later end), will this cause a malfunciton? 

    Returns two INT for the left and right boundary respectively. 
    '''

    # identify which array is left-aligned and which is right-aligned
    if arr_1[0] < arr_2[0]:
        
        x_l_arr = arr_1; # left array
        x_r_arr = arr_2; # array
        
    else:

        x_l_arr = arr_2;
        x_r_arr = arr_1;
        
    # find left boundary
    for x_l in x_l_arr:

        if x_l > x_r_arr[0]:

            left_bound = x_l; break

    # find right boundary
    for x_r in x_r_arr[::-1]:

        if x_r < x_l_arr[-1]:

            right_bound = x_r; break

    return left_bound,right_bound

def getEvents(df,sugar,species,tp,N=1000,SYTO=400):
    '''
    getEvents extracts the flow cytometry data for a treatment of interest, 
    samples N events that passes a user-specific SYTO gate. There may be multiple data sets 
    (e.g. technical replicates) so function may return multiple results. 
    
    Keyword arguments:
    df -- pandas.dataframe where index is FCS file name, and variables inlcude species, sugar, & timepoint.
    sugar -- carbon source (eg. "Arabinose") as STR
    species -- STR of either BO, BF, BT, or BV
    tp -- INT time point of either 24, 48, 72, or 96

    Dependencies:
    * parent directory must contain a folder "data_derived" where each file is a flow cytometry-derived text file. 
      See /home/lad44/davidlab/users/fsm/bacteroides/code/processFlowCytometryData.py for more details. 
        
    Returns
    events_list -- LIST where each item is a pandas.DataFrame with variables of RFP-A, GFP-A, and SYTO-A and index is flow event identifier
    tabund_list -- LIST where each item is an INT with absolute number of events
    '''
    
    events_list = [];
    tabund_list = [];
    
    hits = df[df.isin({'Sugar':[sugar],'TimePoint':[tp],'Species':[species]}).sum(1)==3];
    
    for idx in hits.index:
        
        idx_data = pd.read_csv('../data_derived/%s.txt' % idx, sep='\t', header=0, index_col=0);
        idx_data,ta = sampleData(idx_data,N=N,SYTO=SYTO);
        
        events_list.append(idx_data);
        tabund_list.append(ta);

    return events_list,tabund_list

def getFormattedTime():
    '''
    Constructs time stamp formatted as Year-Month-Day-Hour_Minute_Second

    e.g. '2018-10-10-13-51-12'
    '''

    ts = time.localtime()
    ts = time.strftime("%Y-%m-%d-%H-%M-%S",ts)

    return ts

def getGates(df,sugar,tp):
    '''
    getGates determines gates that divvy up flow cytometry into four quadrants
    based on flow cytometry of mono-cultures at user-defined time-point on user-defined sugar.

    Returns
    dictionary with two keyes for gfp and rfp, and values are gates [low, high]
    '''

    gates = {};

    species_pairs = {'RFP':[('BO','BF'),('BT','BV')],
                     'GFP':[('BO','BT'),('BF','BV')]}

    samples_pairs = list(itertools.product(range(2),repeat=2))

    for band in ['GFP','RFP']:

        gates[band] = {}; # two for gfp, two for rfp. lower gate first.

        for zz,species in enumerate(species_pairs[band]):

            gates[band][species] = [];

            for cnt,samples in enumerate(samples_pairs):

                xx_coords = [];
                yy_coords = [];

                for ii,jj in zip(species,samples):

                    # ii for species and jj for technical replicate
                    
                    events,total_abundance = getEvents(df,sugar,ii,tp,N=1000,SYTO=400)

                    xx_temp,yy_temp = getSignalPDF(events[jj]['%s-A' % band]);

                    xx_coords.append(xx_temp);
                    yy_coords.append([float(total_abundance[jj])*gg for gg in yy_temp]);

                x_1,y_1 = xx_coords[0],yy_coords[0]
                x_2,y_2 = xx_coords[1],yy_coords[1]

                xx,yy = findIntersection(x_1,x_2,y_1,y_2)
                
                if len(xx)==0:
                    xx=0;
                    yy=0;

                elif len(xx)>1:

                    xx_diff = np.abs(xx-400)
                    xx_min = np.min(xx_diff)
                    xx_idx = np.where(xx_diff==xx_min)[0]
                    xx,yy = xx[xx_idx][0],yy[xx_idx][0]

                else:

                    xx,yy = xx[0],yy[0]

                gates[band][species] += [xx]; 

            gates[band][species] = np.median(gates[band][species])


    return gates

def getSignalPDF(events,params=[0,1000,1e-1]):
    '''
    getSignalPDF estimates the probability distribution function for data based on user-specified PARAMETERS

    Keyword arguments:
    events -- numpy.arary (of flow cytometry voltage data)
    params -- three INT that are input values for np.arange to generate x-values array [start,stop,interval]

    Returns:
    ind -- numpy.array of x-values for which PDF is evaluated. 
    pdf -- probability distribution function evaluated at ind values
    '''
        
    ind = np.arange(params[0],params[1],params[2])

    pdf = gkde(events).evaluate(ind)
    
    return ind, pdf

def initializeGrid():
    '''
    initializeGrid for a two-dimensional flow cytometry plot with marginal histograms (or pdf) for each variable.

    Returns
    fig -- matplotlib.figure.figure
    ax1 -- matplot.axes._subplots.AxesSubplot for main sub-plot (countour plot)
    ax2 -- matplot.axes._subplots.AxesSubplot for marginal top-plot (RFP-A)
    ax3 -- matplot.axes._subplots.AxesSubplot for marginal right-plot (GFP-A)

    '''
    fig = plt.figure(figsize=[6,6])
    gs = gridspec.GridSpec(4,4,hspace=0,wspace=0)
    ax1 = fig.add_subplot(gs[1:, 0:3])
    ax2 = fig.add_subplot(gs[0, :-1])
    ax3 = fig.add_subplot(gs[1:, -1])
    
    return fig,ax1,ax2,ax3

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

def jointVoltageFilter(df,opr_dict):
    '''
    jointVoltageFilter removes events based on joint operation on desired channels

    Kewyord arguments:
    df -- pandas.dataframe where rows are events and columns are flow cytometry variables (e.g. channels)
    opr_dict -- Dictionary of channels as keys and operation as np.array with first element as one of {>,<,=} and second as int or float.

    Returns pandas.DataFrame.
 
    Questions:
    would one or two conditions be fine?
    '''

    conds = []

    for channel,rule in opr_dict.iteritems():
            
        if rule[0]=='>':
            
            conds.append(df[channel]>rule[1])
            
        elif rule[0]=='>=':
            
            conds.append(df[channel]>=rule[1])
        
        elif rule[0]=='<':
            
            conds.append(df[channel]<rule[1])
            
        elif rule[0]=='<=':
            
            conds.append(df[channel]<=rule[1])
        
            
        elif rule[0]=='=':
            
            conds.append(df[channel]==rule[1])
        
    return df[conjunction(*conds)]   

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

def plotGates(ax,ax_rfp,ax_gfp,rfp=525,gfp=525):
    '''
    plotGates adds vertical and horizonatl lines on plot to indicate voltage gates

    Keyword arguments
    ax -- matplot.axes._subplots.AxesSubplot for main sub-plot (countour plot)
    ax_rfp -- matplot.axes._subplots.AxesSubplot for marginal top-plot (RFP-A)
    ax_gfp -- matplot.axes._subplots.AxesSubplot for marginal right-plot (GFP-A)
    rfp -- INT voltage gate for RFP-A signal
    gfp -- INT volutage gate for GFP-A signal

    Returns None
    '''
    ax.axvline(x=rfp,ymin=0,ymax=1,lw=3,linestyle='--',color='brown',alpha=0.40)
    ax.axhline(y=gfp,xmin=0,xmax=1,lw=3,linestyle='--',color='brown',alpha=0.40)

    ax_rfp.axvline(x=rfp,ymin=0,ymax=1,lw=3,linestyle='--',color='brown',alpha=0.40)
    ax_gfp.axhline(y=gfp,xmin=0,xmax=1,lw=3,linestyle='--',color='brown',alpha=0.40)

    return None

def plotDynamicGates(ax,ax_rfp,ax_gfp,gates):
    '''
    plotDynamicGates adds vertical and horizonatl lines on plot to indicate voltage gates

    Keyword arguments
    ax -- matplot.axes._subplots.AxesSubplot for main sub-plot (countour plot)
    ax_rfp -- matplot.axes._subplots.AxesSubplot for marginal top-plot (RFP-A)
    ax_gfp -- matplot.axes._subplots.AxesSubplot for marginal right-plot (GFP-A)
    rfp -- INT voltage gate for RFP-A signal
    gfp -- INT volutage gate for GFP-A signal

    Returns None
    '''

    h_1 = gates['GFP'][('BO','BT')];
    h_2 = gates['GFP'][('BF','BV')];
    v_1 = gates['RFP'][('BO','BF')];
    v_2 = gates['RFP'][('BT','BV')];

    ax.axvline(x=v_1,ymin=0,ymax=1,lw=1,linestyle='--',color='brown',alpha=0.40)
    ax.axvline(x=v_2,ymin=0,ymax=1,lw=1,linestyle='--',color='brown',alpha=0.40)

    ax.axhline(y=h_1,xmin=0,xmax=1,lw=1,linestyle='--',color='brown',alpha=0.40)
    ax.axhline(y=h_2,xmin=0,xmax=1,lw=1,linestyle='--',color='brown',alpha=0.40)

    ax_rfp.axvline(x=v_1,ymin=0,ymax=1,lw=1,linestyle='--',color='brown',alpha=0.40)
    ax_rfp.axvline(x=v_2,ymin=0,ymax=1,lw=1,linestyle='--',color='brown',alpha=0.40)

    ax_gfp.axhline(y=h_1,xmin=0,xmax=1,lw=1,linestyle='--',color='brown',alpha=0.40)
    ax_gfp.axhline(y=h_2,xmin=0,xmax=1,lw=1,linestyle='--',color='brown',alpha=0.40)

    return None


def plotMixedTimeSeries(df,Sugar):
    '''
    plotMiedTimeSeries returns a figure with two sub-plots. Left sub-plot is a stacked bar plot 
    of the relative abundance of four Bacteroides species over three or four time points. The 
    right sub-plot is the absolute abundance of the whole community.

    Keyword arguments
    df -- pandas.DataFrame whre each row is a flow sample and variables include 
          'Sugar','Species','BO','BF','BT', and 'BV'
    Sugar -- STR

    Returns
    fig -- matplotlib.figure.Figure
    ax -- list of matplotlib.axes._subplots.AxesSubplot
    '''

    color_dict = {'BO':'navy','BF':'red','BT':'green','BV':'gold'};
    label_dict = {'BO':'Bo','BF':'Bf','BT':'Bt','BV':'Bv'};

    # get relative abundances
    df_sub = df[df.isin({'Sugar':[Sugar],'Species':['MIX']}).sum(1)==2];
    df_sub = df_sub.sort_values(['TimePoint']);
    df_sub

    fig,axes = plt.subplots(1,2,figsize=[10,5]);

    ax_l,ax_r = axes[0],axes[1]

    # some data have only three time points (x2 reps) while others have four
    if df_sub.shape[0]==6:
        bottom = [0]*6;
        x_ticks = [0,1,3,4,9,10];
    elif df_sub.shape[0]==8:
        bottom = [0]*8;
        x_ticks=[0,1,3,4,6,7,9,10];

    # relative abundance in a stacked bar plot
    for idx,row in df_sub.loc[:,['Bo','Bf','Bt','Bv']].T.iterrows():

        ax_l.bar(x_ticks,row.values,bottom=bottom,
               edgecolor='black',color=color_dict[idx.upper()],
               alpha=0.7,width=0.9);

        bottom += row.values;
        
    # absolute abundance plot in a scatter plot
    x = df_sub.loc[:,['TimePoint','Counts']].iloc[:,0];
    y = df_sub.loc[:,['TimePoint','Counts']].iloc[:,1];

    ax_r.scatter(x,y,s=100,color=(0,0,0,0.75));

    # adjust xtick labels
    if df_sub.shape[0]==6:
        plt.setp(ax_l,xticks=[0.5,3.5,9.5],xticklabels=[24,48,96]);
    elif df_sub.shape[0]==8:
        plt.setp(ax_l,xticks=[0.5,3.5,6.5,9.5],xticklabels=[24,48,72,96]);

    # labels and title
    ax_l.set_title(Sugar,fontsize=20);

    ax_l.set_xlabel('Time (hours)',fontsize=20);
    ax_l.set_ylabel('Relative Abundance',fontsize=20);

    ax_r.set_xlabel('Time (hours)',fontsize=20);
    ax_r.set_ylabel('Total Abundance',fontsize=20);

    # adjust y-axis for scatter plot
    ax_r.yaxis.tick_right();
    ax_r.yaxis.set_label_position('right');

    # adjust labels sizes
    [ii.set(fontsize=20) for ii in ax_l.get_xticklabels()+ax_l.get_yticklabels()];
    [ii.set(fontsize=20) for ii in ax_r.get_xticklabels()+ax_r.get_yticklabels()];

    # adjust x-axis and y-axis tick labels 
    plt.setp(ax_r,
             yticks=[0,0.5e5,1e5,1.5e5,2.0e5],
             yticklabels=['0K','50K','100K','150K','200K']);
    plt.setp(ax_r,xticks=[0,24,48,72,96]);

    # adjust y-axis limites
    ax_l.set_ylim([0,1.05]);
    ax_r.set_ylim([0,250000]);
    
    return fig,axes

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

def prettyKDEPlot(df,ax,color):
    '''
    prettyJointPlot draws a plot of two variables with bivariate core and adjoining univariate histograms.

    Keyword arguments:
    df -- pandas.dataframe where rows are events and columns are *two* flow cytometry variables (e.g. channels)

    Returns seaborn plot.
    '''

    x = df.iloc[:,0]; 
    y = df.iloc[:,1];

    pal = sns.light_palette(color, as_cmap=True)
    kd = sns.kdeplot(x,y,shade=True,shade_lowest=False,cmap=pal,ax=ax)

    if (np.max(np.max(df))<1000) and (np.min(np.min(df))>0): 
        
        kd.set_xlim([0,1000]);
        kd.set_ylim([0,1000]);

    # jp.ax_joint.set_xlabel(x.name,fontsize=30);
    # jp.ax_joint.set_ylabel(y.name,fontsize=30);
    # jp.ax_joint.tick_params(labelsize=20);

    # jp.ax_joint.collections[0].set_alpha(0); # what is this?

    #plt.close(jp.fig) # do not display figure

    return kd

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

def relativeAbundance(df,rfp=525,gfp=525):
    '''
    relativeAbundance computes the relative of events in each quadrant of a 2-dimensional 
    flow cytometry plot based on two user-defined gates

    Keyword arguments:
    df -- pandas.DataFrame where variables include RFP-A and GFP-A
    rfp -- INT voltage gate for RFP-A signal
    gfp -- INT volutage gate for GFP-A signal

    Returns
    count -- list of relative abudnance for BO, BF, BT, and BV, respectively
    '''    
    
    counts = countEvents(df,rfp,gfp)
    
    return counts/np.sum(counts);

def sampleData(df_all,N=1000,SYTO=400):
    '''
    sampleData extracts N flow cytometry events that have non-negative GFP and RFP signals 
    and SYTO signal above user-defined threshold.

    Keyword arguments:
    df_all -- pandas.DataFrame where index is flow event identifier, 
              and variables include RFP-A, GFP-A, and SYTO-A
    N -- INT number of samples
    SYTO -- INT voltage threshold for SYTO-A signal

    Returns
    df_sub -- sampled pandas.DataFrame.
    numEvents -- number of flow cytometry events that passe baseline gates.

    '''
    
    df_all = jointVoltageFilter(df_all,{'SYTO-A':['>',SYTO],'GFP-A':['>',0],'RFP-A':['>',0]});
    
    df_all = addPseudoCount(df_all.loc[:,['RFP-A','GFP-A','SYTO-A']]);
    
    df_sub = df_all.sample(min(N,df_all.shape[0]));
    
    numEvents = df_all.shape[0]

    return df_sub,numEvents

def sampleNumber(filename):
    '''
    sampleNumber extracts flow cytometry sample number from filenmae

    filename -- generic file name by MACSQuant VYB

    Returns integer
    '''

    return int(filename.split('.')[-1])
