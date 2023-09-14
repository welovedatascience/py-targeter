import pandas as pd

def autoguess(data:pd.DataFrame=None, var:str=None, 
              remove_missing:bool=True, num_as_categorical_nval:int=5,
               autoguess_nrows:int=1000):

        # perform tests on parameters
        if not isinstance(data, pd.DataFrame): 
            raise TypeError("data must be a panda dataframe")
        #TODO: add more tests/assertions

        column = data[var] #TODO add filter on rows
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
        if (type(column[0].item()) == float) | (type(column[0].item()) == int):
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