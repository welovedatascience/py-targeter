from matplotlib import pyplot as plt 
from targeter import Targeter
import pandas as pd 
df2 = pd.read_csv("C:/Users/natha/OneDrive/Documents/WeLoveDataScience/adult.csv")
import targeter
from sklearn.datasets import load_breast_cancer
df2['tmp' ]=df2.apply(lambda X: '>50K' if X.ABOVE50K == 1  else '<50K', axis = 1)
target2 = 'tmp'
variable2 = 'AGE'
tar2 = targeter.Targeter(df2,target=target2)
tar2.target_stats
tar2.quadrant_plot("AGE")
plt.subplot(2,1,1)
tar2.quadrant_plot("WORKCLASS")
plt.subplot(2,1,1)
plt.show()
