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
        counts = data[target].value_counts()
        proportions = counts / len(data)
        self.index = list(data[target].unique())
        index = list(data[target].unique())
        self._metadata = None
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
        self.target_type = target_type
        if self.target_type == "binary":
            self.target_stats = pd.DataFrame({'Count': counts, 'Proportion': proportions, 'target_reference_level':''})
            for a in range(len(self.index)):
                    if self.index[a]==target_reference_level:
                        self.target_stats.loc[self.index[a],"target_reference_level"] = "x"
            self.mean = data[target].mean()
        
            

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
        
        
        if (target_type == 'continuous'):
            y = data[target].map(lambda x: float(x)).values # ensure int are identified as conitnious by sklearn target nature detection
            y[0] = y[0]+0.000000000001
        else:
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
        if self.target_type == "binary":
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
        if self.target_type == "continuous":
            for ivar in out['name'].values:
                tab = self.get_table(ivar)
                max_index = tab['Mean'].values.argmax()
                max_label = tab.iloc[max_index, tab.columns.get_loc('Bin')]
                max_mean = tab.iloc[max_index, tab.columns.get_loc('Mean')]
                max_count = tab.iloc[max_index, tab.columns.get_loc('Count')]
                max_df = pd.DataFrame({'Max ER - Bin': [max_label],
                                   'Max Mean':[max_mean],
                                   'Max ER - Count':[max_count]})
                tmp_df = pd.concat([tmp_df, max_df], ignore_index=True)


        

        out = pd.concat([out, tmp_df], axis = 1, join = 'inner')
        out['Max ER - Bin'] = out['Max ER - Bin'].map(lambda cell: np.array2string(cell) if isinstance(cell,np.ndarray)  else cell)
        if self._metadata is not None:
                out = pd.merge(out, self._metadata)
                out = out[['name', 'label', 'dtype', 'status', 'selected', 'n_bins', 'iv', 'js', 'gini', 'quality_score', 'Max ER - Bin', 'Max Event Rate', 'Max ER - Count']]
        return(out)

#    def transform(self, x, y):
#        self.profiles.fit_transform(data, data.[target].values)
    def plot(self, name, metric = 'event_rate', add_special = False, add_missing = True, style = 'bin', show_bin_labels = False):
        #<TODO> define style as defualt 'auto' for dtype=numeric use 'actual' if not use 'bin'
        if self.target_type == "binary":
            self.get_optbinning_object(name).binning_table.plot(metric = metric,add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels )
        if self.target_type == "continuous":
            self.get_optbinning_object(name).binning_table.plot(add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels )

        
    def get_optbinning_object(self,name:str):
        return(self.profiles.get_binned_variable(name))
#

    def save(self, path):
        """Save binning process to pickle file.

        Parameters
        ----------
        path : str
            Pickle file path.
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "wb") as f:
            dump(self, f)

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
        
        self.save(tar_pickle_path)
        

        ## <!> temporary: need package and installed package to work...
        
        tofile = os.path.join(tmpdir, 'targeter.py')
        shutil.copy(os.path.join(source_code_dir,'py-targeter', 'targeter.py'), tofile )    
        

        #ff
        cmd =  'quarto render targeter-report.qmd --output generated_report'  + ' -P tmpdir:"'+ tmpdir + '"' + ' --to ' + out_format
        p = subprocess.Popen(cmd, cwd=tmpdir, shell=True, stdout=subprocess.PIPE)
        p.wait()    
        

        if out_file is None:
            out_file = 'report'
        out_file = os.path.join(out_directory, out_file+'.'+out_format)
        
        report_file = os.path.join(tmpdir, 'generated_report')
        shutil.copy(report_file, out_file)    
        


        return(out_file)

    def quadrant_plot(self,name,title=None,xlab="Count",ylab=None, color = 'red', add_missing=True, add_specials=False, show=False):
        #Choose whether to show missing values or not 
        x = self.get_table(name)
        if add_missing == False:
            x = x.drop(x[x["Bin"] == "Missing"].index)
        if add_specials == False:
            x = x.drop(x[x["Bin"] == "Special"].index)
        x = x["Count"].values
        labels = self.get_table(name)[["Bin"]].values
        if self.target_type == "binary":
            y = self.get_table(name)
            if add_specials == False:
                y = y.drop(y[y["Bin"] == "Special"].index)
            if add_missing == False:
                y = y.drop(y[y["Bin"] == "Missing"].index)
            if add_specials == False and add_missing == False:
                y = y.drop(y[y["Bin"] == "Special"].index)
                y = y.drop(y[y["Bin"] == "Missing"].index)
            y = y["Event rate"].values
            pyplot.scatter(x, y)
            pyplot.xlabel(xlab)
           
        

            texts = []
            for i in range(len(x)):
                text_label = ' '.join(str(label) for label in labels[i])
                texts.append(pyplot.text(x[i], y[i], text_label))

            adjust_text(texts)
            z = [self.mean for i in range(len(x))]
            if title is None:
                title = name
            if ylab == None:
                ylab ="Event rate"
            pyplot.ylabel(ylab)
            pyplot.plot(x, z, color=color)
            pyplot.title(title)
        
        if self.target_type == "continuous":
            y = self.get_table(name)
            if add_missing == False:
                y = y.drop(y[y["Bin"] == "Missing"].index)
            if add_specials == False:
                y = y.drop(y[y["Bin"] == "Special"].index)
            if add_specials == False and add_missing == False:
                y = y.drop(y[y["Bin"] == "Special"].index)
                y = y.drop(y[y["Bin"] == "Missing"].index)
            y = y["Mean"].values
            pyplot.scatter(x, y)
            pyplot.xlabel(xlab)
            
        

            texts = []
            for i in range(len(x)):
                text_label = ' '.join(str(label) for label in labels[i])
                texts.append(pyplot.text(x[i], y[i], text_label))

            adjust_text(texts)
            if title is None and self._metadata is not None:
                title = labels(name)
            else:
                title = name 
            if ylab == None:
                ylab = "Mean"
            pyplot.ylabel(ylab)
            pyplot.title(title)
        if show == True:
                pyplot.show()


    def set_metadata(self,meta:pd.DataFrame,var_col:str,label_col:str):
        self._metadata = meta[[var_col,label_col]]
        self._metadata = self._metadata.rename(columns={var_col : "name", label_col : "label"})
        if not(set(self.variable_names) <= set(self._metadata["name"].values)): 
            print("Some var from meta are not in the dataset")

    def get_metadata(self):
        return self._metadata

    def label(self,names):
        if type(names) == str:
            names_list = [names]
            if names not in self.variable_names:
                return(names)
                print("{} does not exist in data".format(names))
        else:
            names_list = list(names)
            if not(set(names_list) <= set(self.variable_names)):
                raise Exception("Names does not exist in data")
        a = pd.DataFrame(names_list, columns=["name"])
        final = pd.merge(self._metadata, a)
        labels_descriptions = []
        for i in range(len(final["name"].values)):
            if str(final["label"].values[i]) == 'nan':
                labels_descriptions.append(str(final["name"].values[i]))
            else:
                labels_descriptions.append(str(final["label"].values[i]))
        return(labels_descriptions)
    
        


        
        



