#!/usr/bin/env python

# Library of functions for analysis of flow cytometry data

import pandas as pd

def minVoltageFilter(df,min_dict):
    '''
    minVoltageFinder emoves events based on minimum bounds for desired channels. 
    
    Kewyord arguments:
    df -- pandas.dataframe where 
    min_dict -- dictionary of channels as keys and minima as values.

    Returns pandas.DataFrame.
    '''

    for channel,minimum in min_dict.iteritems():
        
        df = df[df[channel]>minimum];
        
    return df