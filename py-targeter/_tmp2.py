exec(open('py-targeter/targeter.py').read())



import os

# tmpdir = os.environ['TARGETER_TMPDIR']  


tmpdir = 'C:/temp2'
# tmpdir = 'C:/Users/natha/AppData/Local/Temp/targeter_a'
#
# ff

print(tmpdir)
df2 = pd.read_csv(r"C:\Users\natha\OneDrive\Documents\WeLoveDataScience\df-wlds-equete2014.csv")
target = 'O_523'
variable = 'AGE'
df2 = df2[df2[target].notnull()]

df3 = pd.read_csv("C:/Users/natha/OneDrive/Documents/WeLoveDataScience/adult.csv")
dtypes = {col1: 'float32' for col1 in df2.columns if df2[col1].dtype == 'float64'}
dtypes.update({col2: 'int32' for col2 in df2.columns if df2[col2].dtype == 'int64'})
dtypes.update({col3 : 'object' for col3 in df2.columns if df2[col3].dtype == 'object'})


df = pd.read_csv(r"C:\Users\natha\OneDrive\Documents\WeLoveDataScience\df-wlds-equete2014.csv", dtype = dtypes)

df = df[df[target].notnull()]
df = df.drop(df[df['R_Client_Taux_de_plainte'] == np.inf].index)
df = df.drop(df[df['R_Client_pct_enquete_repondu'] == np.inf].index)

tar = Targeter(target=target,data=df,categorical_variables=list(df.select_dtypes(include=["object"]).columns),select_vars=['B_NOTANSWER_O363'])
tar.set_metadata(meta=meta,var_col="Nom colonne",label_col="newname")


out_directory='.'
out_file=None
template = None

out_format='html'
source_code_dir =  'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter'


# copy template in it
if (template is None):
    # default template:
    template = 'C:/Users/natha/OneDrive/Documents/WeLoveDataScience/py-targeter/template-targeter-report.qmd'
to_template = os.path.join(tmpdir, 'targeter-report.qmd')
shutil.copy(template, to_template)    

tar_pickle_path = os.path.join(tmpdir, 'targeter.pickle')
tar.save( tar_pickle_path)

## <!> temporary: need package and installed package to work...p

tofile = os.path.join(tmpdir, 'targeter.py')
shutil.copy(os.path.join(source_code_dir,'py-targeter', 'targeter.py'), tofile )    
os.environ['TARGETER_TMPDIR'] = tmpdir
import subprocess



cmd =  'quarto render targeter-report.qmd --output generated_report  -P tmpdir:"' + tmpdir + '" --to ' + out_format

# cmd =  'quarto render targeter-report.qmd --output generated_report'  + ' -P tmpdir:"'+ tmpdir + '"' + ' --to ' + out_format
# cmd =  'quarto render targeter-report.qmd --output generated_report --to ' + out_format




import papermill
p = subprocess.Popen(cmd, cwd=tmpdir, shell=True)
p.wait()    


if out_file is None:
    out_file = 'report'
out_file = os.path.join(out_directory, out_file+'.'+out_format)

report_file = os.path.join(tmpdir, 'generated_report').replace(os.sep,"/")




shutil.copy(report_file, out_file)    









# from IPython.display import display, HTML
# from itables import init_notebook_mode, show
# import os
# import matplotlib.pyplot as pyplot
# from adjustText import adjust_text


# def label(var, metadata = None):
#   if metadata is None:
#     return(var)
# metadata = None

# def format_df(df, index = True):
#     """format the dataframe for display"""
#     return display(HTML(df.to_html(index=index)))

# list(tar.profiles.variable_names)

# # loop over variables
# for ivar in list(tar.profiles.variable_names):
#   # display title+anchor for links

#   ivar_label = label(ivar, metadata)
#   itxt = "<div id='var-" + ivar.lower() + "'>\n\n## " + ivar_label

#   if (ivar_label != ivar):
#     itxt = itxt + "(" + ivar + ")"
#   itxt = itxt + "\n</div>"

#   from IPython.display import display, Markdown
#   display(Markdown(f"""{itxt}"""))

# #https://stackoverflow.com/questions/74162212/two-columns-layout-in-quarto
# # https://quarto.org/docs/authoring/figures.html

#   display(Markdown("\n\n::: {<layout='[[45,-10,45],[100]]'}\n\n"))

#   tar.plot(ivar)

#   tar.quadrant_plot(ivar)

#   vars =['Bin','Count', 'Count (%)', 'Event rate','WoE']
#   format_df(tar.get_table(ivar)[vars])

#   display(Markdown("\n\n:::\n\n"))



tar.quadrant_plot('FNLWGT', show=True)