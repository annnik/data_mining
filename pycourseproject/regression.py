import numpy as np
import pandas as pd
import patsy as pt


# file download
df = pd.DataFrame.from_csv("/breast_canser.csv")
# x - table with data (x1, x2, x3)
x = df.iloc[:,:-1]
# y - table with data of dependent variable
y = df.iloc[:,-1]
# result matrix creation
pt_y, pt_x = pt.dmatrices("y ~ x1 + x2 + x3", df)
# MNK optimization
res = np.linalg.lstsq(pt_x, pt_y)
# get coefficients of model
b = res[0].ravel()
print b