.. py-targeter documentation master file, created by
   sphinx-quickstart on Thu Sep 14 12:44:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

py-targeter: Efficient Visual Targets Exploration 
=================================================

Automated Explanatory Data Analysis (EDA) for targets exploration,for both 
binary and numeric targets. Describes a target by crossing it with 
any other candidate explanatory variables. Generates aggregated statistics
allowing to prioritize inspection, such as Information Value (including an
extension of this metric for continuous targets). We also provide plot
methods, automated reports based on Quarto.

Package is aimed at investigating big datasets, both in terms of records and 
variables in an efficient way. 

**py-targeter** relies heavily on wonderful poackageage **OptBinning**

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation
   tutorials/tutorial_targeter 

.. toctree::
   :maxdepth: 1
   :caption: Technical documentation:

   targeter

