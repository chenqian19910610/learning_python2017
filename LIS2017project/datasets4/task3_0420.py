#tips: use "administrator" to install keras, tensorflow, theano in anaconda prompt and cmd simutaneously

import math
def f(x):
    return math.tanh(x)

def deriviate(x):
    h=1/10000
    rise=f(x+h)-f(x)
    run=h
    slope=rise/run
    return slope

print(deriviate(-0.2024))