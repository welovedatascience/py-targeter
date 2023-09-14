"""
Target automated EDA with targeter
"""

# WeLoveDataScience Eric Lecoutre <eric.lecoutre@welovedatascience.com>
# Copyright (C) 2023


import pandas as pd
import inspect
import numpy as np
import os 
import shutil
import subprocess

from optbinning import BinningProcess
from tempfile import mkdtemp
from pickle import dump
from matplotlib import pyplot
from adjustText import adjust_text
from shutil import rmtree

# from pkg_resources import resource_filename
from importlib_resources import files

from .autoguess import autoguess

def _check_inf(data:pd.DataFrame):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if np.inf in data[column].values:
            return True
    return False
def _apply_nan(value):
    if str(value) == "nan":
        return None
    else:
        return value



class Targeter(): 
    """ Target automated EDA main function.
    
    Parameters
    ----------

    data : DataFrame
        Panda dataframe containing target and explanatory candidates variables.

    target: str
        Name of the column of data to be used as target.
    
    select_vars: list or None, optiopnal (default=None)
        List of variables to be crossed with target. If None (default), all
        variables will be used (except target).
    
    exclude_vars: list or None, optional (default=None)
        List of variables to be excluded for the analysis. Tyically id or text
        variables. 
    
    target_type: str. One of "binary","continuous", "auto", optional (default="auto")
        If "auto", one will try to identify the nature of the target by using
        ``autoguess``.

    categorical_variables : array-like or None, optional (default=None)
        List of variables numerical variables to be considered categorical.
        These are nominal variables. 
    
    description_data: str or None, optional (default=None)
        Will be stored in output object and used in report.
    
    target_reference_level: float/int/bool/str or None, optional (default=none)
        For binary and categorical targets (eventually detected as that via 
        ``autoguess`` , value of reference. Typically one of 1, "1", "YES"...
    
    description_target: str or None, optional (default=None).
        Will be stored in output object and used in report. 

    num_as_categorical_nval: int, optional (default=5)
        When (if) using ``autoguess``, all numeric variales having 
        ``num_as_categorical_nval`` or less distinct values will be considered 
        as categorical.

    autoguess_nrows: int, optional (default=1000) 
        Number of rows of ``data`` to be used for variables nature automated
        guess.
    
    metadata: DataFrame or None, optional (default=None)
        DataFrame containing metadata to use labels in reports. DataFrame must
        contain at least 2 columns, one containing variable names (default 
        column name: "var", possibly changed using ``metadata_var`` parameter)
        and a second column containing the label (default name: "label").

    metadata_var; str, optional (default="var")
        Name of the column that contains variables names in ``metadata``
        DataFrame.
    
    metadata_label; str, optional (default="label")
        Name of the column that contains variables labels  in ``metadata``
        DataFrame.

    include_missing: str, one of "any", "always", "never", optional (default="any")
        Defines if report should display missing values category. If "always", 
        Missing will always be displayed in graphics (even if data does not
        contain any missing), if "never", Missing wil be removed (even if data
        does contain missing values). If "any", Missing category will be used
        if and only if there is at least one missing value.

    include_special: str, one of "any", "always", "never", optional (default="never")
        Similar to ``include_missing`` for ``OptBinning`` Special class.
    
    **optbinning_kwargs : keyword arguments.
        Arguments to be passed to ``ProcessBinning``. allowing for instance to
        feed its parameter ``binning_fit_params``.
    
    
    Notes
    -----

    User is highly encouraged to read   ``OptBinning``  package documentation
    to potentially exploit all of its features to customize computation / 
    behaviour (Special values, pre-binning,...).

    For variables shortlist selection, we don't rely on ``OptBinning`` but
    have our own (different) implementation via ``filter`` method.

    .. warning::
        Do not pass the option ```"solver": "mit"`` in OptBinniong parameters
        as we want to be able to store the object for reporting.

    """

    def __init__(self, data:pd.DataFrame=None, target:str=None, 
                 select_vars:list=None, exclude_vars:list=None, 
                 target_type:str="auto", categorical_variables="auto", 
                 description_data:str=None, target_reference_level=None, 
                 description_target:str=None, num_as_categorical_nval:int=5,  
                 autoguess_nrows:int=1000, 
                 metadata:pd.DataFrame=None, metadata_var:str="var", 
                 metadata_label:str="label",
                 include_missing:str="any", include_special:str="never",
                 **optbinning_kwargs):
        
        # perform tests on parameters
        if not isinstance(data, pd.DataFrame): 
            raise TypeError("data must be a panda dataframe")
        if not isinstance(target, str): 
            raise TypeError("target must be a string.")

        #TODO: add more tests/assertions

        # for str list, see https://stackoverflow.com/questions/31353661/type-of-all-elements-in-python-list
        # If you only want to know how many different types are in your list you can use this:
        # set(type(x).__name__ for x in lst)


        # retrieve dataframe name from call and store it in ouput 'data' slot
        if _check_inf(data=data):
            raise Exception("Infinite values in your dataset")
        frame = inspect.currentframe()
        dfname=''
        try:
            for var, val in frame.f_back.f_locals.items():
                if val is data:
                    dfname = var
        finally:
            del frame
        self.data = dfname
        for var in data.columns:
            if autoguess(data,var) == "binary_num":
                data[var] = data[var].apply(str)
            data[var] = data[var].apply(_apply_nan)
        self.target = target
        counts = data[target].value_counts()
        proportions = counts / len(data)
        self.index = list(data[target].unique())
        index = list(data[target].unique())

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
        else:
            self.target_stats = None
            
        if self.target_type in ["binary", "continuous"]:
            self.mean = data[target].mean()
        
#            if type(format_target) == str:,
        # prepare list of variables
        # prepare  data: X and y
        if select_vars is None:
            select_vars = data.columns.values
        #     select_vars[~(select_vars == target)]
        # if (exclude_vars != None):
        #     select_vars = select_vars[(~np.isin(select_vars,exclude_vars))] 
        # Replace the line with this code
        execeed_modality_number_variables = data.select_dtypes(include=["object"]).columns[data.select_dtypes(include=["object"]).nunique() > 30].tolist()     
        if exclude_vars is not None:
            exclude_vars = list(set(exclude_vars + execeed_modality_number_variables))
            select_vars = list(set(select_vars).difference(exclude_vars))
        else:
            exclude_vars_n = {target} | set(execeed_modality_number_variables)
            select_vars = list(set(select_vars).difference(exclude_vars_n))
        

        self.variable_names = select_vars
        if metadata is not None and metadata_var is not None and metadata_label is not None:
            self.set_metadata(meta=metadata, var_col=metadata_var, label_col=metadata_label) 
        else:
            self._metadata = None  

        # prepare data for optbinni
        X= data.filter(items =select_vars, axis = 1)
        
        
        if (target_type == 'continuous'):
            y = data[target].map(lambda x: float(x)).values # ensure int are identified as conitnious by sklearn target nature detection
            y[0] = y[0]+0.000000000001
        else:
            y = data[target].values


        #X = df.drop(columns=target).values
        all_optb = BinningProcess(variable_names=select_vars)  # ...definition of what we want to do as computation
        all_optb = all_optb.fit(X, y) # effectively perform computations



        # post operations

        # build all binning tables
        #for ivar in all_optb._binned_variables:
        #    all_optb._binned_variables[ivar].binning_table.build(add_totals=False)

        self.profiles = all_optb
        if select_vars is None:
            self.selection = list(data.columns)
        else:
            self.selection = select_vars
        self.filtered = False   
        self.include_missing = include_missing
        self.include_special = include_special



    # def get_binning_table(self, name):
    #    self.profiles.get_binned_variable(name).binning_table


    def get_table(self, name, show_digits = 2, add_totals = False):
        return(self.get_optbinning_object(name).binning_table.build(show_digits = show_digits, add_totals = add_totals))

    def summary(self,include_labels=False):

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
        out['Selected'] = ''
        for i in range(len(out["name"].values)):
            if out["name"].values[i] in self.selection:
                out.loc[i,"Selected"] = "x"
        out = out.sort_values(by = "Selected", ascending = False)
        if self._metadata is not None and include_labels == True:
                out = pd.merge(out, self._metadata)
                if self.target_type == "binary":
                    out = out[['name', 'label', 'dtype', 'status', 'selected', 'n_bins', 'iv', 'js', 'gini', 'quality_score', 'Max ER - Bin', 'Max Event Rate', 'Max ER - Count','Selected']]
                if self.target_type == "continuous":
                    out = out[['name', 'label', 'dtype', 'status', 'selected', 'n_bins', 'quality_score', 'Max ER - Bin', 'Max Mean', 'Max ER - Count','Selected']]
        out_selected = out[out["Selected"] == "x"].sort_values(by=["quality_score"], ascending = False)
        out_not_selected = out[out["Selected"] != "x"].sort_values(by=["quality_score"], ascending = False)
        out = pd.concat([out_selected, out_not_selected])

        
        return(out)

#    def transform(self, x, y):
#        self.profiles.fit_transform(data, data.[target].values)
    def plot(self, name, metric = 'event_rate', add_special = False, add_missing = True, style = 'bin', show_bin_labels = True):
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

    def report(self, out_directory='.', out_file=None, template=None, 
               out_format='html',filter="auto", filter_count_min=500, 
               filter_n=20, force_var=None, delete_tmp=True):

        if filter == "auto":
            if self.filtered == False:
                a1 = self.filter(n=filter_n, metric="quality_score").selection
            
                if self.target_type == "binary":
                    a2 = self.filter(n=filter_n, metric="js").selection
                    a3 = self.filter(n=filter_n, metric="iv").selection
                    a4 = self.filter(n=filter_n, metric="Max Event Rate", count_min=filter_count_min).selection
                    a5 = self.filter(n=20, metric="gini").selection
                    self.selection = list(set(a1 + a2 + a3 + a4 + a5))
            
                if self.target_type == "continuous":
                    a2 = self.filter(n=filter_n, metric="Max Mean", count_min=filter_count_min).selection
                    self.selection = list(set(a1 + a2))
            
            self.filtered = True
    
        if force_var is not None:
            self.selection = list(set(list(self.selection) + list(force_var)))
        else:
            self.selection = list(set(list(self.selection)))
    
    # create temporary folder
        tmpdir = mkdtemp(prefix='targeter_')
    
    # copy template to the temporary folder
        if template is None:
            ## see https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
            # template = pkgutil.get_data(__name__, "assets/template-targeter-report.qmd")
            template_path = 'template-targeter-report.qmd'  # always use slash
            template =  files("targeter.assets").joinpath(template_path)
            
            # template = 'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter/template-targeter-report.qmd'
        to_template = os.path.join(tmpdir, 'targeter-report.qmd')
        shutil.copy(template, to_template)
    
        tar_pickle_path = os.path.join(tmpdir, 'targeter.pickle')
        self.save(tar_pickle_path)
        os.environ['TARGETER_TMPDIR'] = tmpdir
        #TODO check if quarto is installed 
        cmd = 'quarto render targeter-report.qmd --output generated_report --to ' + out_format
    
        p = subprocess.Popen(cmd, cwd=tmpdir, shell=True, stdout=subprocess.PIPE)
        p.wait()
    
        if out_file is None:
            out_file = 'report'
        out_file = os.path.join(out_directory, out_file+'.'+out_format)
    
        report_file = os.path.join(tmpdir, 'generated_report').replace(os.sep, "/")
        #TODO try and catch error
        shutil.copy(report_file, out_file)
    
        if delete_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
        return out_file


    def quadrant_plot(self,name,title=None,xlab="Count",ylab=None, color = 'red', add_missing=True, add_special=False, show=False):
        #Choose whether to show missing values or not 
        df = self.get_table(name)
        if add_special == False:
            df = df[~df["Bin"].isin(['Special'])]
        if add_missing == False:
            df = df[~df["Bin"].isin(['Index'])]
        labels = df["Bin"].values
        
        x = df["Count"].values

        if self.target_type == "binary":
            y = df["Event rate"].values
            if ylab == None:
                ylab ="Event rate"
        
        if self.target_type == "continuous":
            y = df["Mean"].values
            if ylab == None:
                ylab ="Mean"
    
        pyplot.scatter(x, y)
        pyplot.xlabel(xlab)

        texts = []
        for i in range(len(x)):
            text_label = ' '.join(str(label) for label in labels[i])
            texts.append(pyplot.text(x[i], y[i], text_label))

        adjust_text(texts)

        z = [self.mean for i in range(len(x))]
        pyplot.plot(x, z, color=color)

        if title is None and self._metadata is None:
            title = name
        if title is None and self._metadata is not None:
            title = self.label(name)[0]
        pyplot.title(title)

        pyplot.ylabel(ylab)
        
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
        if self._metadata is None:
            label_descriptions = []
            for i in range(names):
                label_descriptions.append(names[i])
                return label_descriptions
        else:
            final = pd.merge(a, self._metadata, on = 'name', how='left')
            labels_descriptions = []
            for i in range(len(final["name"].values)):
                if str(final["label"].values[i]) == str(np.nan):
                    labels_descriptions.append(str(final["name"].values[i]))
                else:
                    labels_descriptions.append(str(final["label"].values[i]))
        
        return(labels_descriptions)
    
    def filter(self,
               metric:str="auto",
               n:int=25,
               min_criteria:float=0.1,
               count_min:int = None,
               force_var:list = None,
               max_criteria:float = None,
               sort_method:bool = False):
        final = self.summary()
        target_type = self.target_type
        continuous_metrics = ["quality_score", "Max Mean"]
        binary_metrics = ["iv", "js", "gini", "quality_score", "Max Event Rate"]
        if (metric == "auto"):
            if (target_type == "binary"):
                metric = binary_metrics[0]
            elif (target_type =="continuus"):
                metric = continuous_metrics[0]
        else:
            if target_type == "binary" and metric not in binary_metrics:
                    raise Exception("{} does not match available metrics".format(metric))
            if target_type == "continuous" and metric not in continuous_metrics:
                    raise Exception("{} does not match available metrics".format(metric))
        # determine set
        final = final.drop(final[final[metric] < min_criteria].index)
        if max_criteria is not None:
            final = final.drop(final[final[metric] > max_criteria].index)
        if count_min is not None:
            final = final.drop(final[final["Max ER - Count"] < count_min].index)    
        if sort_method == True:
            final = final.sort_values(by = metric, ascending =  True)
        else:
            final = final.sort_values(by = metric, ascending = False)
        if n is not None:
            final = final.iloc[0:n,:]
        variables_selected = list(final["name"].values)
        
        if force_var is not None:
            variables_selected = variables_selected + force_var
        variables_selected = list(set(variables_selected))

        self.selection = variables_selected
        self.filtered = True

        return(self)


