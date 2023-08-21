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
        '''
    Automatically guesses the data type of a variable based on its values.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing the variable.
    var : str
        The name of the variable for which the data type needs to be guessed.
    remove_missing : bool, optional
        Whether to remove missing values before guessing. Default is True.
    num_as_categorical_nval : int, optional
        Number of unique values to consider a numeric variable as categorical. Default is 5.
    autoguess_nrows : int, optional
        Number of rows to use for autoguessing the variable types. Default is 1000.

    Returns
    -------
    data_type : str
        The guessed data type of the variable. Possible values include:
        - "binary_bool": Binary boolean variable.
        - "binary_str": Binary categorical variable with string values.
        - "categorical_str": Categorical variable with string values.
        - "binary_num": Binary numeric variable.
        - "categorical_num": Categorical numeric variable.
        - "continuous": Continuous numeric variable.
        - "unknown": Unknown data type.

    '''
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

def check_inf(data:pd.DataFrame):
    '''
    Checks if there are any infinite values (np.inf) in the numeric columns of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to be checked for infinite values.

    Returns
    -------
    has_infinite : bool
        True if any infinite values are found, False otherwise.

    '''
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if np.inf in data[column].values:
            return True
    return False
    
def apply_nan(value):
    if str(value) == "nan":
        return None
    else:
        return value




class Targeter():
    
    def __init__(self,data:pd.DataFrame = None, target:str = None, select_vars:list = None, exclude_vars:list = None, target_type:str = "auto", categorical_variables = "auto", description_data = None, target_reference_level = None, description_target = None,num_as_categorical_nval=5,  autoguess_nrows = 1000, metadata=None,var_col="Nom colonne", label_col="newname", **optbinning_kwargs):
        # retrieve dataframe name from call and store it in ouput 'data' slot
        '''
    Initializes a Targeter object for variable binning and analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing the variables.
    target : str
        The name of the target variable for analysis.
    select_vars : list of str, optional
        A list of variable names to be considered for analysis. If None, all columns will be used.
    exclude_vars : list of str, optional
        A list of variable names to be excluded from analysis.
    target_type : str, optional
        The type of the target variable: "binary" or "continuous". If "auto", the type will be guessed.
    categorical_variables : str or list of str, optional
        The variable(s) that should be treated as categorical. If "auto", the variables will be guessed.
    description_data : dict, optional
        A dictionary containing descriptions for the data.
    target_reference_level : str or int, optional
        The reference level of the target variable for binary classification.
    description_target : str, optional
        A description for the target variable.
    num_as_categorical_nval : int, optional
        Number of unique values to consider a numeric variable as categorical.
    autoguess_nrows : int, optional
        Number of rows to use for autoguessing the variable types.
    metadata : pandas.DataFrame, optional
        A DataFrame containing metadata for variable names and labels.
    var_col : str, optional
        The name of the column in the metadata DataFrame that contains the variable names.
    label_col : str, optional
        The name of the column in the metadata DataFrame that contains the labels for the variables.
    **optbinning_kwargs : dict, optional
        Additional arguments to be passed to the optbinning.BinningProcess constructor.

    '''
        if check_inf(data=data):
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
            data[var] = data[var].apply(apply_nan)
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
        if metadata is not None and var_col is not None and label_col is not None:
            self.set_metadata(meta=metadata,var_col=var_col, label_col=label_col) 
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
        



    # def get_binning_table(self, name):
    #    self.profiles.get_binned_variable(name).binning_table
    


    def get_table(self, name:str, show_digits:int = 2, add_totals:bool = False):
        '''
        Parameters
        ----------
            name : str
                Name of the variable choosen
            show_digits : int

            add_totals : bool 
        
        Returns
        -------

        '''
        return(self.get_optbinning_object(name).binning_table.build(show_digits = show_digits, add_totals = add_totals))

    def summary(self,include_labels=False):
        """
    Generate a summary of the binning process and statistics for the variables.

    Parameters
    ----------
    include_labels : bool, optional
        Whether to include labels from metadata in the summary. Default is False.

    Returns
    -------
    summary_df : pandas.DataFrame
        A summary DataFrame containing information and statistics for each variable.

    """
    

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
        if self.target_type == "continuous":
            for i in range(len(out["name"].values)):
                out.loc[i,"iv"] = self.get_optbinning_object(out.loc[i,"name"]).binning_table.iv
        
        return(out)

#    def transform(self, x, y):
#        self.profiles.fit_transform(data, data.[target].values)
    def plot(self, name, metric = 'event_rate', add_special = False, add_missing = True, style = 'bin', show_bin_labels = True):
        '''
    Plots the binning table statistics or the actual bin boundaries for a specified variable.

    Parameters
    ----------
    name : str
        The name of the variable for which the plot will be generated.
    metric : str, optional
        The metric to use for plotting bin statistics. Applicable only for binary target type.
    add_special : bool, optional
        Whether to include special bins in the plot (if present).
    add_missing : bool, optional
        Whether to include the bin for missing values in the plot (created by optbinning).
    style : str, optional
        The style of the plot. Choose between 'bin' (plot bin statistics) or 'actual' (plot actual bin boundaries).
    show_bin_labels : bool, optional
        Whether to show bin labels on the plot.

    Raises
    ------
    ValueError
        If the specified style is not valid.

    '''
        #<TODO> define style as defualt 'auto' for dtype=numeric use 'actual' if not use 'bin'
        if self.target_type == "binary":
            self.get_optbinning_object(name).binning_table.plot(metric = metric,add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels )
        if self.target_type == "continuous":
            self.get_optbinning_object(name).binning_table.plot(add_special = add_special, add_missing = add_missing, style = style, show_bin_labels = show_bin_labels)

        
    def get_optbinning_object(self,name:str):
        return(self.profiles.get_binned_variable(name))
#

    def save(self, path):
        """
    Save the binning process to a pickle file.

    Parameters
    ----------
    path : str
        The path to the pickle file where the binning process will be saved.

    Raises
    ------
    TypeError
        If the provided `path` is not a string.

    """
        if not isinstance(path, str):
            raise TypeError("path must be a string.")

        with open(path, "wb") as f:
            dump(self, f)

    def report(self, out_directory='.', out_file=None, template=None, out_format='html',source_code_dir='C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter',filter="auto", filter_count_min=500, filter_n=20, force_var=None, delete_tmp=False,include_missing:str = "any", include_special:str = "never"):
        '''
    Generates a report containing statistics on selected variables, plots and quadrant plots.

    Parameters
    ----------
    out_directory : str, optional
        The directory where the generated report will be saved. Default is the current working directory.
    out_file : str, optional
        The base name of the generated report file (without the extension). Default is "report".
    template : str, optional
        The path to the Quarto Markdown template file for the report. If not provided, a default template is used.
    out_format : str, optional
        The output format of the report (e.g., 'html', 'pdf'). Default is 'html'.
    source_code_dir : str, optional
        The directory containing the source code files for the `py-targeter` package.
    filter : str, optional
        The filter mode for variable selection. If "auto", applies automatic variable selection based on metrics.
    filter_count_min : int, optional
        The minimum count of occurrences for a bin to be considered during automatic filtering.
    filter_n : int, optional
        The maximum number of variables to be selected during automatic filtering.
    force_var : list of str, optional
        A list of variable names that must be included in the report regardless of other criteria.
    delete_tmp : bool, optional
        Whether to delete the temporary directory used for report generation. Default is False.
    include_missing : str, optional
        How to include bins for missing values in quadrant plots: "any" (at least one occurrence), "never", or "always".
    include_special : str, optional
        How to include special bins in quadrant plots: "never", "always", or "auto" (based on `filter` mode).
    
    Returns
    -------
    out_file : str
        The path to the generated report file.

    '''
        self.include_missing = include_missing
        self.include_special = include_special
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
            template = 'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter/template-targeter-report.qmd'
        to_template = os.path.join(tmpdir, 'targeter-report.qmd')
        shutil.copy(template, to_template)
    
        tar_pickle_path = os.path.join(tmpdir, 'targeter.pickle')
        self.save(tar_pickle_path)
 
        tofile = os.path.join(tmpdir, 'targeter.py')
        shutil.copy(os.path.join(source_code_dir, 'py-targeter', 'targeter.py'), tofile)
    
        os.environ['TARGETER_TMPDIR'] = tmpdir
    
        cmd = 'quarto render targeter-report.qmd --output generated_report --to ' + out_format
    
        p = subprocess.Popen(cmd, cwd=tmpdir, shell=True, stdout=subprocess.PIPE)
        p.wait()
    
        if out_file is None:
            out_file = 'report'
        out_file = os.path.join(out_directory, out_file+'.'+out_format)
    
        report_file = os.path.join(tmpdir, 'generated_report').replace(os.sep, "/")
        shutil.copy(report_file, out_file)
    
        if delete_tmp:
            os.remove(tmpdir, 'targeter.py')
    
        return out_file


    def quadrant_plot(self,name,title=None,xlab="Count",ylab=None, color = 'red', add_missing=True, add_special=False, show=False):
        #Choose whether to show missing values or not 
        '''
    Generates a quadrant plot for a given variable's bin statistics.

    Parameters
    ----------
    name : str
        The name of the variable for which the quadrant plot is generated.
    title : str, optional
        The title of the quadrant plot. If not provided, the variable's label (if available) or its name is used.
    xlab : str, optional
        The label for the x-axis. Default is "Count".
    ylab : str, optional
        The label for the y-axis. If not provided, "Event rate" (for binary target) or "Mean" (for continuous target) is used.
    color : str, optional
        The color of the reference line representing the mean value. Default is 'red'.
    add_missing : bool, optional
        Whether to include the bin for missing values in the plot (created by optbinning).
    add_special : bool, optional
        Whether to include special bins in the plot (if present).
    show : bool, optional
        Whether to display the plot. Default is False.

    '''
        df = self.get_table(name)
        if add_special == False:
            df = df[~df["Bin"].isin(['Special'])]
        if add_missing == False:
            df = df[~df["Bin"].isin(['Missing'])]
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
        '''
    Sets metadata for variables in the dataset with specified variable names and labels.

    Parameters
    ----------
    meta : pd.DataFrame
        The metadata containing variable names and corresponding labels.
    var_col : str
        The name of the column in the metadata DataFrame that contains the variable names.
    label_col : str
        The name of the column in the metadata DataFrame that contains the labels for the variables.

    Raises
    ------
    Warning
        If some variable names from the metadata are not present in the dataset.

    '''
        self._metadata = meta[[var_col,label_col]]
        self._metadata = self._metadata.rename(columns={var_col : "name", label_col : "label"})
        if not(set(self.variable_names) <= set(self._metadata["name"].values)): 
            print("Some var from meta are not in the dataset")

    def get_metadata(self):
        return self._metadata

    def label(self,names):
        '''
    Assigns labels to a list of variables.

    Parameters
    ----------
    names : str or list of str
        The variable names to which labels need to be assigned. Can be a single variable name as a string,
        or a list of variable names.

    Returns
    -------
    label_descriptions : list of str
        A list of label descriptions corresponding to the input variable names.

    Raises
    ------
    Exception
        If any of the specified variable names do not exist in the data.

    '''    
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
    
    def filter(self,metric:str="iv",n:int=25,min_criteria:float=0.1, count_min:int = None,force_var:list = None,max_criteria:float = None, sort_method:bool = False):
        '''
    Filters and selects variables based on a chosen metric and other criteria.

    Parameters
    ----------
    metric : str
        {'quality_score', 'Max Mean', 'iv', 'js', 'gini', 'Max Event Rate'}, default = 'iv'.
        The metric used to filter the list of variables.
    n : int
        default = 25. The maximum number of variables to be selected.
    min_criteria : float
        default = 0.1. The minimum threshold for variable selection based on the chosen metric.
    max_criteria : float, optional
        The maximum threshold for variable selection based on the chosen metric.
    count_min : int, optional
        Minimum count of variable occurrences to be considered for selection.
    force_var : list, optional
        A list of variables to be included in the final selection regardless of other criteria.
    sort_method : bool, default = False
        If True, sort variables in ascending order of the chosen metric; otherwise, sort in descending order.

    Returns
    -------
    self
        The modified instance with the selected variables.

    Raises
    ------
    Exception
        If the specified metric does not match the available metrics for the target variable type.

    '''
        final = self.summary()
        continuous_metrics = ["quality_score", "Max Mean", 'iv']
        binary_metrics = ["iv", "js", "gini", "quality_score", "Max Event Rate"]
        if self.target_type == "binary" and metric not in binary_metrics:
                raise Exception("{} does not match available metrics".format(metric))
        if self.target_type == "continuous" and metric not in continuous_metrics:
                raise Exception("{} does not match available metrics".format(metric))
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




    
        


        
        







    
        


        
        



