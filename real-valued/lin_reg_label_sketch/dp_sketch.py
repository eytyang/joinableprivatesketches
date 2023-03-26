'''
Given a two DataFrames 'df_known' and 'df_private', this class simulates a join (on index) 
between these two DataFrames that preserves the privacy of df_private:

    (1) The index of the join comes from a Member_Sketch object with privacy 'eps_memb'

    (2) The columns from 'df_private' in the join come from a RealValued_Sketch object 
        with privacy 'eps_val'
'''

import pandas as pd

from membership_sketch import Member_Sketch
from realvalued_sketch import RealValued_Sketch

def decode_sketch(df, col, sketch_obj):
    '''
    Use 'sketch_obj' to populate a new column 'col' in 'df'.
    Modifies the DataFrame 'df' directly.
    '''
    decoded_col = {}
    for i in df.index:
        bucket_index = sketch_obj.hashes.buckets[i]
        decoded_col[i] = sketch_obj.hashes.signs[i] * sketch_obj.sketch[bucket_index]
    df[col] = pd.Series(decoded_col)
    return df

class DP_Join:
    def __init__(self, eps_memb, eps_val, sens = 1.0, num_buckets = None):
        '''
        The DP_Join object stores the following:

            - eps_memb:     'eps' (privacy parameter) for the Member_Sketch
            - eps_val:      'eps' (privacy parameter) for the RealValued_Sketch over all columns
                                of 'df_private'
            - sens:         a positive real number, representing an upper bound on the 
                                magnitude all values in 'df_private'.
                                For next version: pupport different 'sens' for different columns
            - num_buckets:  the value of 'num_buckets' passed into the Member_Sketch and
                                the RealValued_Sketch
            - df_joined:    the simulated private join between 'df_known' and 'df_private' 
        '''
        self.eps_memb = eps_memb
        self.eps_val = eps_val
        self.sens = sens
        self.num_buckets = num_buckets
        self.df_joined = None

    def join(self, df_known, df_private):
        '''
        Simulates a private join between df_known and df_private. 
        Adds a column 'membership' to df_known, populated using a Member_Sketch.
        Adds a column 'col' for every column in 'df_private', populated using a Value_Sketch.
        '''
        index_universe = df_private.index.union(df_known.index)
        df_joined = df_known.copy()

        # Create the membership sketch and append the membership columns to df_joined.
        memb = Member_Sketch(self.eps_memb, index_universe, self.num_buckets)
        memb.populate(df_private.index)
        df_joined = decode_sketch(df_joined, 'membership', memb)
        
        # Create each column's value sketch and append them to df_joined
        num_cols = len(df_private.columns)
        for col in df_private.columns:
            new_col = {}
            val = RealValued_Sketch(self.eps_val / num_cols, self.sens, index_universe, self.num_buckets)
            val.populate(df_private[col])
            df_joined = decode_sketch(df_joined, col, val)              
        self.df_joined = df_joined

    def drop_entries(self):
        '''
        Filter rows of 'self.df_joined' where the 'membership' value is <= 0. 
        '''
        self.df_joined = self.df_joined[self.df_joined['membership'] >= 1]
