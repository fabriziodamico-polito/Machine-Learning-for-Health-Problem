# -*- coding: utf-8 -*-
"""
Optimization Methods Demo — LLS, Gradient Descent, and Steepest Descent.

Demonstrates three approaches to solving the linear least-squares problem
on a synthetically generated system Aw = y.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin
import numpy as np

#LLS
Np = 100 #number of rows
Nf = 4 #number of columns
A = np.random.randn(Np,Nf) # matrix/Ndarray A
w = np.random.randn(Nf,1) # true vector w
y = A@w# column vector y
m = mymin.SolveLLS(y,A) # instantiate the object
m.run() #run LLS
m.print_result("LLS") # print the results ( inherited method)
m.plot_what("LLS")# plot what ( inherited method)

#GA
Nit = 1000 # number of steps for the gradient algorithm
gamma = 1e-5 # learning rate for the gradient algorithm
g=mymin. SolveGrad(y,A) # instantiate SolveGrad
g.run(gamma, Nit ) # run SolveGrad
g.print_result ( "Gradient algorithm") # inherited method
logx = 0 # we want a natural scale on the x−axis
logy = 1 # we want a logarithmic scale on the y−axis
g.plot_err ( "Gradient algorithm : square error " ,logy , logx) # inherited method
g.plot_what ( "Gradient algorithm ") # inherited method

#SDM
Nit = 1000
tol = 1e-8
s = mymin.SolveSteepDesc(y,A)
s.run(Nit,tol)
s.print_result("Steepest Descent")
s.plot_what("Steepest Descent")
