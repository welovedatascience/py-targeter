#import optbinning
import pandas as pd
from optbinning import BinningProcess
import inspect
import numpy as np
from tempfile import mkdtemp
import os
import shutil
from pickle import dump
import subprocess
from matplotlib import pyplot
from adjustText import adjust_text


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


    # def get_binning_table(self, name):
    #    self.profiles.get_binned_variable(name).binning_table


    def get_table(self, name, show_digits = 2, add_totals = False):
        return(self.get_optbinning_object(name).binning_table.build(show_digits = show_digits, add_totals = add_totals))

    def summary(self):

        out = self.profiles.summary()
        
        tmp_df = pd.DataFrame() 
        for ivar in out['name'].values:
            # print(ivar)
            tab = self.get_table(ivar)
            max_index = tab['Event rate'].values.argmax()
            max_label = tab.iloc[max_index, tab.columns.get_loc('Bin')]
            max_event_rate = tab.iloc[max_index, tab.columns.get_loc('Event rate')]
            max_count = tab.iloc[max_index, tab.columns.get_loc('Count')]
            max_df = pd.DataFrame({'Max ER - Bin': [max_label],
                                   'Max Event Rate':[max_event_rate],
                                   'Max ER - Count':[max_count]})
            tmp_df = pd.concat([tmp_df, max_df], ignore_index=True)

        out = pd.concat([out, tmp_df], axis = 1, join = 'inner')
        out['Max ER - Bin'] = out['Max ER - Bin'].map(lambda cell: np.array2string(cell) if isinstance(cell,np.ndarray)  else cell)
        return(out)

#    def transform(self, x, y):
#        self.profiles.fit_transform(data, data.[target].values)
    def plot(self, name, metric = 'event_rate', add_special = False, add_missing = True, style = 'bin', show_bin_labels = False):
        #<TODO> define style as defualt 'auto' for dtype=numeric use 'actual' if not use 'bin'
        self.get_optbinning_object(name).binning_table.plot(metric = metric,add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels )
        
    def get_optbinning_object(self,name:str):
        return(self.profiles.get_binned_variable(name))
#
    def report(self, out_directory='.', out_file=None, template = None, out_format='html', source_code_dir =  'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter'):
        
        # create temporary folder
        tmpdir = mkdtemp(prefix = 'targeter_')


        # copy template in it
        if (template is None):
            # default template:
            template = 'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter/template-targeter-report.qmd'
        to_template = os.path.join(tmpdir, 'targeter-report.qmd')
        shutil.copy(template, to_template)    
        
        tar_pickle_path = os.path.join(tmpdir, 'targeter.pickle')
        file = open(tar_pickle_path,'wb')
        dump(self, file)
        file.close()        
        
        ## <!> temporary: need package and installed package to work...
        
        tofile = os.path.join(tmpdir, 'targeter.py')
        shutil.copy(os.path.join(source_code_dir,'py-targeter', 'targeter.py'), tofile )    
        

        #ff
        cmd =  'quarto render targeter-report.qmd --output generated_report'  + ' -P tar_pickle_path:"'+ tar_pickle_path + '"' + ' --to ' + out_format
        p = subprocess.Popen(cmd, cwd=tmpdir, shell=True, stdout=subprocess.PIPE)
        p.wait()    
        

        if out_file is None:
            out_file = 'report'
        out_file = os.path.join(out_directory, out_file+'.'+out_format)
        
        report_file = os.path.join(tmpdir, 'generated_report')
        shutil.copy(report_file, out_file)    
        


        return(out_file)

    def quadrant_plot(self,name,title=None,xlab="Count",ylab="Event rate", color = 'red'):
        x = self.get_table(name)["Count"].values
        y = self.get_table(name)["Event rate"].values
        pyplot.scatter(x, y)
        pyplot.xlabel(xlab)
        pyplot.ylabel(ylab)
        labels = self.get_table(name)[["Bin"]].values

        texts = []
        for i in range(len(x)):
            text_label = ' '.join(str(label) for label in labels[i])
            texts.append(pyplot.text(x[i], y[i], text_label))

        adjust_text(texts)
        z = [self.target_stats.values[1] for i in range(len(x))]
        if title is None:
            title = name
        pyplot.plot(x, z, color=color, title = title)
        pyplot.show()

