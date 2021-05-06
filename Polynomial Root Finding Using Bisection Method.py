#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[179]:


NAME = "Voyager0001"
COLLABORATORS = ""


# ---

# Polynomial Root Finding Using Bisection Method
# ---
# 
# ### `pandas` Dataframe:
# Before we start we will know a little about `pandas` dataframe. `pandas` is a python library. Dataframes are classes that are used to store complex data. You can initialize one as a python dictionary. Let's take a look. 

# In[180]:


import pandas as pd

x = [0, 1, 2, 3, 4, 5, 6, 7]
y = [1.0, 5.2, 3.9, 8.4, 14.6, 12.3, 8.9, 10.2]

dictionary = {
    "x": x,
    "y": y,
}
df = pd.DataFrame(dictionary)
df.head(8)


# We can use it to print data in a tabular format. We can even use more numpy arrays with it.

# In[181]:


import numpy as np

x = np.linspace(0, np.pi/2, 15)
y = np.sin(x)

dictionary = {
    "x": x,
    "sin(x)": y,
}
df = pd.DataFrame(dictionary)
df.head(15)


# ### Root Finding:
# Let $f(x)$ be a function of $x$. if for some $x=\alpha, f(x) = f(\alpha) = 0$, we say $\alpha$ is a root of function $x$.
# 
# Let, 
# $$f(x) = x^5 + 2.5x^4 - 2x^3 -6x^2 + x + 2\tag{6.1}$$
# 
# The graph of $f(x)$ looks like this.

# In[182]:


from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

f = Polynomial([2.0, 1.0, -6.0, -2.0, 2.5, 1.0])
x = np.linspace(-2.5, 1.5, 100)
y = f(x)
dictionary = {
    'x': x,
    'y': y
}
plt.axhline(y=0, color='k')
plt.plot(x, y)
plt.plot(f.roots(), [0.0]*5, 'ro')
print(f.roots())


# Roots are the points where the graph intersects with the $X$-axis.
# 
# ### Bisection
# 
# One way to find out root's are to use bisection method. Here is the strategy, if $\alpha$ is a root between and interval $[a, b]$ then graph will cross the $X$-axis at $\alpha$. So, $sign( f(\alpha-h) ) = - sign( f(\alpha+h) )$, for small value of $h$. 
# 
# So, we can work our way up towards the root by taking average of $a$ and $b$, as long as the signs are different.
# 
# we will start with $a_0$ and $b_0$, such that, $f(a_0) f(b_0) < 0$.
# Then we iterate as this,
# \begin{align}
# m_k &= \frac{a_k + b_k}{2} \\
# \text{if, } f(a_k) f(m_k) < 0, \text{ then, } a_{k+1} &= a_k \text{ and } b_{k+1} = m_k\\
# \text{else, } a_{k+1} &= m_k \text{ and, } b_{k+1} = b_k
# \end{align}
# 
# We keep iterating until we find the root with sufficient precision. We usually use a formula like this,
# $$
# \frac{|m_{k+1} - m_k|}{|m_{k+1}|} \leq \epsilon \\  
# $$
# Where, $\epsilon$ is a very small value, like $\epsilon < 10^{-6}$
# 
# ### Complete the code below
# Complete the code below to iterate and solve for a root of the following equation, between the interval, $[-0.5, 1.3]$:
# \begin{aligned}
#     f(x) = 2 + x - 6x^2 - 2x^3 + 2.5x^4 + x^5.
# \end{aligned}

# In[183]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# The polynomial and the range is defined for you
f = Polynomial([2.0, 1.0, -6.0, -2.0, 2.5, 1.0])
a = -0.5
b = 1.3
m = (a + b) / 2
e = 1e-6

root = 0.0    # You need to update this value

# Populate the following lists in each iteration
list_a = []
list_b = []
list_m = []
list_f = []

# YOUR CODE HERE
def m_k(a_k, b_k):
    return (a_k+b_k)/2.00
list_a.append(a)
list_b.append(b)
list_m.append(m)
list_f.append(f(a)*f(m))
k=0
while True:
    if list_f[k]<0:
        list_a.append(list_a[k])
        list_b.append(list_m[k])
        list_m.append(m_k(list_a[k+1], list_b[k+1]))
        list_f.append(f(list_a[k+1])*f(list_m[k+1]))
        
    else:
        list_a.append(list_m[k])
        list_b.append(list_b[k])
        list_m.append(m_k(list_a[k+1], list_b[k+1]))
        list_f.append(f(list_a[k+1])*f(list_m[k+1]))
        
    if abs(list_m[k+1]-list_m[k])/abs(list_m[k+1])<=e:
        root=list_m[k]
        break
        
    k+=1


# In[184]:


xs = np.linspace(-2.5, 1.5, 100)
ys = f(xs)

plt.axhline(y=0, color='k')
plt.plot(xs, ys)
plt.plot(root, f(root), 'ro')

print(pd.DataFrame({'a':list_a, 'b':list_b, 'm':list_m, 'f(m)':list_f}))

assert "{:.3f}".format(root) == "0.672"


# In[ ]:




