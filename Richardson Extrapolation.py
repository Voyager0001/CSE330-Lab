#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[28]:


NAME = "Voyager0001"
COLLABORATORS = ""


# ---

# ## CSE330 Lab: Richardson Extrapolation
# ---

# ## Instructions
# 
# 
# Today's assignment is to:
# 1. Implement Richardson Extrapolation method using Python
# 
# ## Richardson Extrapolation:
# We used central difference method to calculate derivatives of functions last task. In this task we will use Richardson extrapolation to get a more accurate result.
# Let,
# $$ D_h = \frac{f(x_1+h) -f(x_1-h)}{2h}\tag{5.1}$$ 
# 
# 
# General Taylor Series formula:
# $$ f(x) = f(x_1) + f'(x_1)(x - x_1) + \frac{f''(x_1)}{2}(x - x_1)^2+... $$
# Using Taylor's theorem to expand we get,
# 
# \begin{align}
# f(x_1+h) &= f(x_1) + f^{\prime}(x_1)h + \frac{f^{\prime \prime}(x_1)}{2}h^2 + \frac{f^{\prime \prime \prime}(x_1)}{3!}h^3 + \frac{f^{(4)}(x_1)}{4!}h^4 + \frac{f^{(5)}(x_1)}{5!}h^5 + O(h^6)\tag{5.2} \\
# f(x_1-h) &= f(x_1) - f^{\prime}(x_1)h + \frac{f^{\prime \prime}(x_1)}{2}h^2 - \frac{f^{\prime \prime \prime}(x_1)}{3!}h^3 + \frac{f^{(4)}(x_1)}{4!}h^4 - \frac{f^{(5)}(x_1)}{5!}h^5 + O(h^6)\tag{5.3}
# \end{align}
# Subtracting $5.3$ from $5.2$ we get,
# $$ f(x_1+h) - f(x_1-h) = 2f^{\prime}(x_1)h + 2\frac{f^{\prime \prime \prime}(x_1)}{3!}h^3 + 2\frac{f^{(5)}(x_1)}{5!}h^5 + O(h^7)\tag{5.4}$$
# So,
# \begin{align}
# D_h &= \frac{f(x_1+h) - f(x_1-h)}{2h} \\
# &= \frac{1}{2h} \left( 2f^{\prime}(x_1)h + 2\frac{f^{\prime \prime \prime}(x_1)}{3!}h^3 + 2\frac{f^{(5)}(x_1)}{5!}h^5 + O(h^7) \right) \\
# &= f^{\prime}(x_1) + \frac{f^{\prime \prime \prime}(x_1)}{6}h^2 + \frac{f^{(5)}(x_1)}{120}h^4 + O(h^6) \tag{5.5}
# \end{align}
# We get our derivative $f'(x)$ plus some error terms of order $>= 2$ Now, we want to bring our error order down to 4.
# 
# If we use $h, \text{and} \frac{h}{2}$ as step size in $5.5$, we get,
# \begin{align}
# D_h &= f^{\prime}(x_1) + f^{\prime \prime \prime}(x_1)\frac{h^2}{6} + f^{(5)}(x_1) \frac{h^4}{120} + O(h^6) \tag{5.6} \\
# D_{h/2} &= f^{\prime}(x_1) + f^{\prime \prime \prime}(x_1)\frac{h^2}{2^2 . 6} + f^{(5)}(x_1) \frac{h^4}{2^4 . 120} + O(h^6) \tag{5.7}
# \end{align}
# Multiplying $5.7$ by $4$ and subtracting from $5.6$ we get,
# \begin{align}
# D_h - 4D_{h/2} &= -3f^{\prime}(x) + f^{(5)}(x_1) \frac{h^4}{160} + O(h^6)\\ 
# \Longrightarrow D^{(1)}_h = \frac{4D_{h/2} - D_h}{3} &= f^{\prime}(x) - f^{(5)}(x_1) \frac{h^4}{480} + O(h^6) \tag{5.8}
# \end{align}
# Let's calculate the derivative using $5.8$
# 
# 
# ### 1. Let's import the necessary headers

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.polynomial import Polynomial


# ### 2. Let's create a function named `dh(f, h, x)`
# function `dh(f, h, x)` takes three parameters as input: a function `f`, a value `h`, and a set of values `x`. It returns the derivatives of the function at each elements of array `x` using the Central Difference method. This calculates equation $(5.1)$.

# In[30]:


def dh(f, h, x):
    '''
    Input:
        f: np.polynomial.Polynonimial type data. 
        h: floating point data.
        x: np.array type data.
    Output:
        return np.array type data of slope at each point x.
    '''
    # --------------------------------------------
    # YOUR CODE HERE
    return (f(x+h)-f(x-h)) / 2*h
    # --------------------------------------------


# ### 3. Let's create another funtion `dh1(f, h, x)`.
# `dh1(f, h, x)` takes the same type of values as `dh(f, h, x)` as input. It calculates the derivative using previously defined `dh(f, h, x)` function and using equation $5.8$ and returns the values.

# In[31]:


def dh1(f, h, x):
    '''
    Input:
        f: np.polynomial.Polynonimial type data. 
        h: floating point data.
        x: np.array type data.
    Output:
        return np.array type data of slope at each point x.
    '''
    # --------------------------------------------
    # YOUR CODE HERE
    return (4*dh(f,h/2,x)-dh(f,h,x))/3
    # --------------------------------------------


# ### 4. Now let's create the `error(f, hs, x_i)` function
# The `error(f, hs, x_i)` function takes a function `f` as input. It also takes a list of different values of h as `hs` and a specific value as `x_i` as input. It calculates the derivatives as point `x_i` using both functions described in **B** and **C**, i.e. `dh` and `dh1`

# In[32]:


def error(f, hs, x_i):  #Using the functions we wrote dh() my c_diff  and dh1() which is my first order c diff, we find the error through appending their diffrences with Y_actual ny f(x)
    '''
    Input:
        f  : np.polynomial.Polynonimial type data. 
        hs : np.array type data. list of h.
        x_i: floating point data. single value of x.
    Output:
        return two np.array type data of errors by two methods..
    '''
    f_prime = f.deriv(1)   #first order derivitive f^1(x)
    Y_actual = f_prime(x_i)  

    diff_error = []
    diff2_error = []

    for h in hs: #where h is my loop counter iterating through hs
        # for each values of hs calculate the error using both methods
        # and append those values into diff_error and diff2_error list.

        # --------------------------------------------
        # YOUR CODE HERE
        diff_error_h_i = dh(f, h, x_i) - Y_actual
        diff_error = np.append(diff_error, diff_error_h_i)
        diff2_error_h_i = dh1(f, h, x_i) - Y_actual
        diff2_error = np.append(diff2_error, diff2_error_h_i)
        # --------------------------------------------
        
    print(pd.DataFrame({"h": hs, "Diff": diff_error, "Diff2": diff2_error}))

    return diff_error, diff2_error


# ### 5. Finally let's run some tests
# function to draw the actual function

# In[33]:


def draw_graph(f, ax, domain=[-10, 10], label=None):
    data = f.linspace(domain=domain)
    ax.plot(data[0], data[1], label='Function')


# ### Draw the polynomial and it's actual derivative function

# In[34]:


fig, ax = plt.subplots()
ax.axhline(y=0, color='k')

p = Polynomial([2.0, 1.0, -6.0, -2.0, 2.5, 1.0])
p_prime = p.deriv(1)
draw_graph(p, ax, [-2.4, 1.5], 'Function')
draw_graph(p_prime, ax, [-2.4, 1.5], 'Derivative')

ax.legend()


# ### Draw the actual derivative and richardson derivative using `h=1` and `h=0.1` as step size.

# In[35]:


fig, ax = plt.subplots()
ax.axhline(y=0, color='k')

draw_graph(p_prime, ax, [-2.4, 1.5], 'actual')

h = 1
x = np.linspace(-2.4, 1.5, 50, endpoint=True)
y = dh1(p, h, x)
ax.plot(x, y, label='Richardson; h=1')

h = 0.1
x = np.linspace(-2.4, 1.5, 50, endpoint=True)
y = dh1(p, h, x)
ax.plot(x, y, label='Richardson; h=0.1')

ax.legend()


# ### Draw error-vs-h cuve

# In[36]:


fig, ax = plt.subplots()
ax.axhline(y=0, color='k')
hs = np.array([1., 0.55, 0.3, .17, 0.1, 0.055, 0.03, 0.017, 0.01])
e1, e2 = error(p, hs, 2.0)
ax.plot(hs, e1, label='e1')
ax.plot(hs, e2, label='e2')

ax.legend()


# In[ ]:




