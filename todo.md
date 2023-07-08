# <todo>

## internship scholar report
[!] write report table of content, send to Eric for validation 
[ ] Section introduction/ internship content 
[ ] Describe Environment set-up
    [ ] some information about technical problem encountered and fix (Anaconda)
    [ ] learned: working with venv
[ ] Section code implementation presentation
[ ] Section application on real data
    [ ] presentation of data
    [ ] first usage of routines: outcome
    [ ] iterative application: removal of highly correlated variables
    [ ] report interpretation, not necessary field per field but synthesis
[ ] Section modeling
    [ ]... <todo>


## statistical analysis:EDA
[!] Profiles reports on colored variables
    [ ] exclude conceptually correlated variables (satisfaction concept)
    [ ] 3 reports (3 colored variables)

## Statistical analysis: modeling
[!] select variables to be used, add target and some statistics in report
[!] prepare train/test (scikit learn)
[ ] feature engineering pipeline (WOE recoding excluded, done later) - one hot encoding, outliers...
[ ] Run targeter on targeter on train set
[ ] apply WOE recoding on training (-> new/other dataset)
[ ] fit different models
    Cross:
    - train dataset (after feature engineering pipeline)
    - WOE recoded one
    with:
    - Lasso regression model
    - Decision tree model
    - Random forest model
 [ ] Apply pipeline on test sets    
    [ ] two tests sets: raw / WOE one
 [ ] Ensure to derive features importance
 [ ] Derive model precision/accuracy metrics
 [ ] models comparisons and selection
    [ ] any benefit/improvement with of WOE?


## code

### Quarto template
[!] Fix error in temmplate (hwen running report)
    KeyError: "['label'] not in index"
    When metadata is None (not provided)
[!] play with quarto parameters
        format:
        html:
            fig-width: 3.5
            fig-height: 3
    in template headerto change values so that 2 plots show side by side having maximum width
[ ] Also use parameters include_missings and  include_specials for base plot (not quadrant_plot)


### python Code
[x][20230705] - check i fthere are some Inf in explanatory variables
    -> to be tested
[ ][20230705] - exclude catagorical variables with more than 30 values
    - to be testes
    - [ ] value of '30' to be put as parameter of main function Targeter (default: 30)

[ ] translate into English  parameeter value (var_col="Nom colonne")
[ ]  change :   
    include_missings:str = "Any", include_specials:str = "Never"
    to
    include_missings:str = "any", include_specials:str = "never"
    Impact on all codes


## code: python package py-targeter
[ ]   Document class+functions *in code* similar syntax used in optbinning (-> documentation auto-generation of classes)
[ ] Wrap as a package   
    [X] requirements: pip freeze + manage at package level (google)
    [ ] way/how to include a data  file (adults.csv)
    [ ] README.md to complete
    [ ] adapt quarto templates  default path + reference to targeter.py > we should *never* see a path 'c:/...' hardcoded anymore but reference/way to retrieve and use whereppackage is installed.
[ ] "vignette" (jupyter notebook that inntroduces package functionalities with example)
[ ] test installation & usage in windows subsystem for Linux (VScode should be able to view & connect to it when selecting python environment)
[-] Add tests to package (even very simple one) and use gihtub action test python package Eric did configure.

# Notes prises 

Jeu de données applati

Dimentsion du gîte, propriétaire du gite
Sejour et réservation c'est pareil 
Les maisons n'ont pas du tout les mêmes poids
Peut être étape de data management pour aggreger les maisons (regrouper les réservations en maison)   ---> pose soucis 

Pondérer pour que chaque maison ait le même poids 
