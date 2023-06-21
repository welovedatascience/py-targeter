#import optbinning
import pandas as pd
from optbinning import BinningProcess
import inspect
import numpy as np


def autoguess(data, var, remove_missing=True, num_as_categorical_nval=5,  autoguess_nrows = 1000):
        column = data[var] #<TODO> add filter on rows
        if (remove_missing):
            column = column[column.notnull()]
        column = column.values
        type_col = type(column[0])
        if type_col == bool:
            return "binary_bool"
        vals = list(set(column))
        if type_col == str:
            if len(vals) == 1:
                return "unimode"
            elif len(vals) == 2:
                return "binary_str"
            else:
                return "categorical_str"
        if (type_col == float) | (type_col == int):
            if len(vals) == 1:
                return "unimode"
            elif len(vals) == 2:
                return "binary_num"
            elif len(vals) <= num_as_categorical_nval:
                return "categorical_num"
            else:
                return "continuous"
        return "unknown"
#autoguess(df, " ABOVE50K")



class Targeter():
    def __init__(self,data:pd.DataFrame = None, target:str = None, select_vars:list = None, exclude_vars:list = None, target_type:str = "auto", categorical_variables = "auto", description_data = None, target_reference_level = None, description_target = None,num_as_categorical_nval=5,  autoguess_nrows = 1000, **optbinning_kwargs):
        # retrieve dataframe name from call and store it in ouput 'data' slot
        frame = inspect.currentframe()
        dfname=''
        try:
            for var, val in frame.f_back.f_locals.items():
                if val is data:
                    dfname = var
        finally:
            del frame
        self.data = dfname
        self.target = target
        # handle target type
        if target_type == "auto":
            target_type = autoguess(data, var = target, remove_missing=True,num_as_categorical_nval=5,  autoguess_nrows = 1000)
            if (target_type in ['categorical_num','categorical_str']):
                target_type = 'categorical'
            if (target_type in ['binary_bool']):
                target_type = 'binary'
                if target_reference_level is None:
                    target_reference_level = 1
                # <TODO> recode in numeric
                data['target'] = (data['target'] == True).astype(int)


            if (target_type in ['binary_num']):
                target_type = 'binary'
                if target_reference_level is None:
                    target_reference_level = 1

            if (target_type in ['binary_str']):
                target_type = 'binary'
        
        if (target_type == 'binary'):
            if target_reference_level is None:
                # try to guess what wiuld be a good reference value
                if type(data[target].values[0]) == str:
                    #recodage
                    target_reference_level = data[target].value_counts(sort = True, ascending = True).index[0] #Added ascending to take modality  with lowest counts
                else: 
                    # type int/num/bolean, expecting 1/True , any other type (dates?)-> max value
                    target_reference_level = max(data[target].dropna().values)
            self.target_reference_level = target_reference_level
            # recode # !<WARN> problem : will remove missing values
            data[target] = (data[target] == target_reference_level).astype(int)
            print("the reference level has been defined as:{}".format(target_reference_level))

#            if type(format_target) == str:,
        # prepare list of variables
        # prepare  data: X and y
        if select_vars == None:
            select_vars = data.columns.values
        #     select_vars[~(select_vars == target)]
        # if (exclude_vars != None):
        #     select_vars = select_vars[(~np.isin(select_vars,exclude_vars))]        
        
        select_vars = select_vars[(~np.isin(select_vars,[target, exclude_vars]))]        
        
        self.variable_names = select_vars

        # prepare data for optbinni
        X= data.filter(items =select_vars, axis = 1)
        y = data[target].values

        #X = df.drop(columns=target).values
        all_optb = BinningProcess(variable_names=  select_vars,**optbinning_kwargs)  # ...definition of what we want to do as computation
        all_optb = all_optb.fit(X, y) # effectively perform computations



        # post operations

        # build all binning tables
        #for ivar in all_optb._binned_variables:
        #    all_optb._binned_variables[ivar].binning_table.build(add_totals=False)

        self.profiles = all_optb    


    def get_binning_table(self, name):
        self.profiles.get_binned_variable(name).binning_table
    def summary(self):
        out = self.profiles.summary()
        #<TODO> additional steps to be added to add some more information
        return(out)

#    def transform(self, x, y):
#        self.profiles.fit_transform(data, data.[target].values)
    def plot(self, name, metric = 'event_rate', add_special = False, add_missing = True, style = 'actual', show_bin_labels = False):
        #<TODO> define style as defualt 'auto' for dtype=numeric use 'actual' if not use 'bin'
        self.profiles.get_binned_variable(name).binning_table.plot(metric = metric,add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels )
        
    def get_binned_variable(self,name:str):
        return(self.profiles.get_binned_variable(name))
#
    
    






    

    
            



        