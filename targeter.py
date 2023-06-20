#import optbinning
import pandas as pd
from optbinning import BinningProcess



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



class Targeter(data, target, target_type, variable_names, categorical_variables):
    def __init__(data:pd.DataFrame, target:str, select_vars = None, exclude_vars = None, target_type:str = "auto", categorical_variables = "auto", description_data = None, target_reference_level = None, description_target = None,num_as_categorical_nval=5,  autoguess_nrows = 1000, **optbinning_kwargs):
        self.data = data
        self.target = target
        self.variable_names = variable_names
        if target_type == "auto":
            target_type = autoguess(data, var = target, remove_missing=True,num_as_categorical_nval=5,  autoguess_nrows = 1000)
            if (target_type in ['categorical_num','categorical_str']):
                target_type = 'categorical'
            if (target_type in ['binary_bool']):
                target_type = 'binary'
                if target_reference_level is None:
                    target_reference_level = 1
                # <TODO> recode in numeric
                self.data['target'] = (self.data['target'] == True).astype(int)


            if (target_type in ['binary_num']):
                target_type = 'binary'
                if target_reference_level is None:
                    target_reference_level = 1

            if (target_type in ['binary_str']):
                target_type = 'binary'
        
        if (target_type == 'binary'):
            if target_reference_level is None:

                #target_reference_level = data[target].values[0] # <TODO>: smarter guess
                
                if type(data[target].values[0]) == str:
                    #recodage
                    vals = list(set(data[target].values))
                    a0 = data.loc[data[target] == vals[0]][target].count()
                    a1 = data.loc[data[target] == vals[1]][target].count()
                    if a0 < a1:
                        target_reference_level = vals[0]
                    else:
                        target_reference_level = vals[1]
                else: # type(data[target].values[0]) == float or type(data[target].values[0]) == int or type(data[target].values[0]) == bool
                    target_reference_level = max(data[target].dropna().values)
            self.target_reference_level = target_reference_level
            self.data[target] = (self.data[target] == target_reference_level).astype(int)
            print("the reference level has been defined as:{}".format(target_reference_level))
       
                



#            if type(format_target) == str:,
        # prepare list of variables
        # defines target type and reference level
        
        # prepare  data: X and y

   
        
        all_optb = BinningProcess(variable_names = ,**optbinning_kwargs)  # ...definition of what we want to do as computation
        all_optb = all_optb.fit(X, y) # effectively perform computations
       
       
       # post operations

       # build all binning tables
       for ivar. in all_optb._binned_variables:
        all_optb._binned_variables[ivar].binning_table.build(add_totals=False)

        self.profiles = all_optb    


    def get_binning_table(self, name):
        self.profiles.get_binned_variable(name).binning_table
    def summary(self):
        out = self.profiles.summary()
        #<TODO> additional steps to be one
        return(out)

    def transform(self, x, y):
        self.profiles.fit_transform(self.data, self.data.[target].values)
    
    def get_binned_variable(self,name:str):
        self.profiles.get_binned_variable(name)
    
    
    






    

    
            



        