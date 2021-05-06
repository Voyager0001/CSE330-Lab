#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Voyager0001"
COLLABORATORS = ""


# ---

# # Differentiation: Forward, Backward, And Central
# ---

# ## Task 1: Differentiation
# 
# We have already learnt about *forward differentiation*, *backward diferentiation* and *central differentiation*. In this part of the assignment we will write methods to calculate this values and check how they perform.
# 
# The equations are as follows,
# 
# \begin{align}
# \text{forward differentiation}, f^\prime(x) \simeq \frac{f(x+h)-f(x)}{h} \tag{4.6} \\
# \text{backward differentiation}, f^\prime(x) \simeq \frac{f(x)-f(x-h)}{h} \tag{4.7} \\
# \text{central differentiation}, f^\prime(x) \simeq \frac{f(x+h)-f(x-h)}{2h} \tag{4.8}
# \end{align}

# ## Importing libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


# Here, `forward_diff(f, h, x)`, `backward_diff(f, h, x)`, and `central_diff(f, h, x)` calculates the *forward differentiation*, *backward differentiation* and *central differentiation* respectively. finally the `error(f, f_prime, h, x)` method calculates the different values for various $h$ and returns the errors.
# 
# Later we will run some code to test out performance. The first one is done for you.

# In[3]:


def forward_diff(f, h, x):
    return (f(x+h) - f(x)) / h


# In[4]:


def backward_diff(f, h, x):
    # --------------------------------------------
    # YOUR CODE HERE
    return ( f(x) -f(x+h)) / h
    
    # --------------------------------------------


# In[5]:


def central_diff(f, h, x):
    # --------------------------------------------
    # YOUR CODE HERE
    return ( f(x+h)-f(x-h)) / 2*h
    # --------------------------------------------


# In[6]:


def error(f, f_prime, h, x):
    Y_correct = f_prime(x)
    f_error = np.array([])
    b_error = np.array([])
    c_error = np.array([])
    
    for h_i in h:
        # for different values of h (h_i)
        # calculate the error at the point x for  (i) forward method 
        #                                         (ii) backward method
        #                                         (ii) central method
        # the first one is done for you
        f_error_h_i = forward_diff(f, h_i, x) - Y_correct
        f_error = np.append(f_error, f_error_h_i)

        # --------------------------------------------
        # YOUR CODE HERE
        b_error_h_i = backward_diff(f, h_i, x) - Y_correct
        b_error = np.append(b_error, b_error_h_i)
        c_error_h_i = central_diff(f, h_i, x) - Y_correct
        c_error = np.append(c_error, c_error_h_i)
        # --------------------------------------------
        pass
    
    return f_error, b_error, c_error


# ## Plot1
# Polynomial and Actual Derivative Function

# In[7]:


fig, ax = plt.subplots()
ax.axhline(y=0, color='k')

p = Polynomial([2.0, 1.0, -6.0, -2.0, 2.5, 1.0])
data = p.linspace(domain=[-2.4, 1.5])
ax.plot(data[0], data[1], label='Function')

p_prime = p.deriv(1)
data2 = p_prime.linspace(domain=[-2.4, 1.5])
ax.plot(data2[0], data2[1], label='Derivative')

ax.legend()


# In[8]:


h = 1
fig, bx = plt.subplots()
bx.axhline(y=0, color='k')

x = np.linspace(-2.0, 1.3, 50, endpoint=True)
y = forward_diff(p, h, x)
bx.plot(x, y, label='Forward; h=1')
y = backward_diff(p, h, x)
bx.plot(x, y, label='Backward; h=1')
y = central_diff(p, h, x)
bx.plot(x, y, label='Central; h=1')

data2 = p_prime.linspace(domain=[-2.0, 1.3])
bx.plot(data2[0], data2[1], label='actual')

bx.legend()


# In[9]:


h = 0.1
fig, bx = plt.subplots()
bx.axhline(y=0, color='k')

x = np.linspace(-2.2, 1.3, 50, endpoint=True)
y = forward_diff(p, h, x)
bx.plot(x, y, label='Forward; h=0.1')
y = backward_diff(p, h, x)
bx.plot(x, y, label='Backward; h=0.1')
y = central_diff(p, h, x)
bx.plot(x, y, label='Central; h=0.1')

data2 = p_prime.linspace(domain=[-2.2, 1.3])
bx.plot(data2[0], data2[1], label='actual')

bx.legend()


# In[10]:


h = 0.01
fig, bx = plt.subplots()
bx.axhline(y=0, color='k')

x = np.linspace(-2.2, 1.3, 50, endpoint=True)
y = forward_diff(p, h, x)
bx.plot(x, y, label='Forward; h=0.01')
y = backward_diff(p, h, x)
bx.plot(x, y, label='Backward; h=0.01')
y = central_diff(p, h, x)
bx.plot(x, y, label='Central; h=0.01')

data2 = p_prime.linspace(domain=[-2.2, 1.3])
bx.plot(data2[0], data2[1], label='actual')

bx.legend()


# In[11]:


fig, bx = plt.subplots()
bx.axhline(y=0, color='k')

h = np.array([1., 0.55, 0.3, .17, 0.1, 0.055, 0.03, 0.017, 0.01])
err = error(p, p_prime, h, 2.0)

bx.plot(h, err[0], label='Forward')
bx.plot(h, err[1], label='Backward')
bx.plot(h, err[2], label='Central')
bx.legend()


# In[ ]:




