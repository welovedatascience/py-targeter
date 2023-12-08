import streamlit as st
import pandas as pd 

import targeter
# import os
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

import os
import numpy as np
from pickle import load as pickle_load

# tmpdir = os.environ['TARGETER_TMPDIR']
tmpdir='.'
tmpdir='/hackdata/share/code/python/py-targeter/targeter/assets/'

 
st.set_page_config(layout="wide")



print(tmpdir)
tar_pickle_path = os.path.join(tmpdir, 'targeter.pickle')

# tar_pickle_path = '/hackdata/share/code/python/py-targeter/targeter/assets/sample-targeter.pickle'

file = open(tar_pickle_path, 'rb')
tar = pickle_load(file)
file.close()


## prepare tab_sum to be diplsayed
tab_sum = tar.summary(include_labels=True)

if tar.target_type == "binary":
    tab_sum.rename(columns={ 'quality_score':'qs','Max ER - Bin': 'Max bin','Max Event Rate':'MaxBin Rate','Max ER - Count':'MaxBin N', 'Selected':'Selected'}, inplace = True)
    tab_sum.sort_values(by=['iv'], ascending = False)
    if tar._metadata is not None:
        vars = ['name','label','n_bins','iv','qs','Max bin','MaxBin Rate','MaxBin N', 'Selected']
    else:
        vars = ['name', 'n_bins','iv','qs','Max bin','MaxBin Rate','MaxBin N', 'Selected']
    tab_sum = tab_sum.astype({'iv': float, 'qs': float}) # https://dev.to/theolakusibe/solving-attributeerror-float-object-has-no-attribute-rint-4la#:~:text=Solution%20To%20convert%20the%20column%20with%20a%20mixed,by%20using%20the%20pandas.DataFrame.astype%20function%2C%20as%20illustrated%20below
    tab_sum['iv'] = tab_sum['iv'].round(3)
    tab_sum['qs'] = tab_sum['qs'].round(3)
    tab_sum['MaxBin Rate'] = tab_sum['MaxBin Rate'].round(3)

if tar.target_type == "continuous":
    tab_sum.rename(columns={ 'quality_score':'qs','Max ER - Bin': 'Max bin','Max Mean':'MaxBin Mean','Max ER - Count':'MaxBin N', 'Selected':'Selected'}, inplace = True)
    tab_sum.sort_values(by=['qs'], ascending = False)
    if tar._metadata is not None:
        vars = ['name', 'label','n_bins','qs','Max bin','MaxBin Mean','MaxBin N', 'Selected']
    else:
        vars = ['name','n_bins', 'qs', 'Max bin', 'MaxBin Mean', 'MaxBin N', 'Selected']
    tab_sum = tab_sum.astype({'qs': float}) # https://dev.to/theolakusibe/solving-attributeerror-float-object-has-no-attribute-rint-4la#:~:text=Solution%20To%20convert%20the%20column%20with%20a%20mixed,by%20using%20the%20pandas.DataFrame.astype%20function%2C%20as%20illustrated%20below
    tab_sum['qs'] = tab_sum['qs'].round(3)
    tab_sum['MaxBin mean'] = tab_sum['MaxBin Mean'].round(3)

tab_sum = tab_sum[vars]

#Infer basic colDefs from dataframe types
gb = GridOptionsBuilder.from_dataframe(tab_sum)

# gb.configure_side_bar()


gb.configure_selection('single', use_checkbox=False, pre_select_all_rows=1)
gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()
 
#Display the grid 
st.header("py-targeter explorer")

grid_response = AgGrid(
    tab_sum, 
    gridOptions=gridOptions, 
    width='100%',
    data_return_mode='FILTERED_AND_SORTED', 
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    enable_enterprise_modules=False
    )


if len(grid_response['selected_rows']) >0:

    selected_var = grid_response['selected_rows'][0]['name']

 
if tar.target_type == "binary":
        df = tar.get_table(selected_var)
        show_special = 'If any' # - parameter picklist
        show_missing = 'If any' # - parameter picklist

        df['Bin'] = df.Bin.astype(str)
        
        if (show_special=='Never'):
            df = df[df['Bin'] != 'Special']
        if (show_missing=='Never'):
            df = df[df['Bin'] != 'Missing']

        if (show_special=='If any'):
            df = df[(df['Bin'] != 'Special') | 
                        ( (df['Bin']=='Special') & ((df['Event']>0)|(df['Non-event']>0))) ]
        if (show_missing=='If any'):
            df = df[(df['Bin'] != 'Missing') | 
                        ( (df['Bin']=='Missing') & ((df['Event']>0)|(df['Non-event']>0))) ]


        # events = df[['Bin','Event']]
        # events = events.rename(columns={'Event':'Count'})
        # events['Target']='Event'

        # nonevents = df[['Bin','Non-event']]
        # nonevents = nonevents.rename(columns={'Non-event':'Count'})
        # nonevents['Target']='Non-event'

        # dfp = pd.concat( [events, nonevents])
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_bar(x=df['Bin'], y=df['Event'])
        fig.add_bar(x=df['Bin'], y=df['Non-event'])
        fig.update_layout(barmode='stack',
                  title =selected_var,
                  template = 'plotly_dark')
        
        fig.add_trace(
            go.Scatter(x=df['Bin'], y=df['Event rate'], name="%"),
            secondary_y=True,
        )
        
        st.plotly_chart(fig)
    
        with st.expander('Table'):
            st.markdown(df.to_html(), unsafe_allow_html=True)
