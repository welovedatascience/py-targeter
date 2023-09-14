"""
Targeter testing
"""

# WeLoveDataScience Eric Lecoutre <eric.lecoutre@welovedatascience.com>
# Copyright (C) 2023

import pandas as pd

from pkg_resources import resource_filename
from pytest import approx, raises
from targeter import Targeter


data_url = resource_filename("targeter", "assets/adult.csv")
adult = pd.read_csv(data_url)

def test_params():
    with raises(TypeError):
        tar = Targeter(data=1) #data not pd.DataFrame

    with raises(TypeError):
        tar = Targeter(data=adult, target=1) 

# def test_stored_values():
#     tar = Targeter(data=adult, target='ABOVE50K',select_vars = ['EDUCATION'])
#     assert tar.selection == ['EDUCATION']


#TODO: add tests