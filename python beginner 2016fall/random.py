""""
from scipy import stats
import numpy as np
x = np.random.random(10)
y = np.random.random(10)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)


import pandas as pd
import numpy as np
df= df.dropna(axis=1, how="") #delete the column, axis=0 is delete the rows
df["types"]=df["types"].astype("category") #overwrite the original string column into category type
df.isnull.any()


 function.attribute
 function(argument)



import pandas as pd
import numpy as np
pip install -U scikit-learn
"""

import numpy as np
from scipy.integrate import quad
def integrand (t,n,x):
    return np.exp(-x*t)/t**n
def expint(n,x):
    return quad(integrand, 1, np.inf, args=(n,x))[0]
vec_expint=np.vectorize (expint)
vec_expint(3,np.arange (1,4,0.5))

dgfdsgfdsgf

