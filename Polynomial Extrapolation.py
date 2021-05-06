Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).

Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

NAME = "Voyager0001"
COLLABORATORS = "None"

---

# Part 1: Representing a Polynomial

Polynomials are function of the following format

$$p(x) = a_0 + a_1 x ^ 1 + a_2 x ^ 2 + ... + a_n x ^ n$$

 $[a_0, a_1, \cdots a_n]$ are called coefficients and $n$ (called the degree or order) is a non-negative integer


This can also be written as

$$y = f(x) = a_0 x^0 + a_1 x ^ 1 + a_2 x ^ 2 + ... + a_n x ^ n$$

**Example**

For example, $$ y = 1 + 2x^2 + 5x^4 $$ is a polynomial of order 4 ($ = n$) with $n+1$ coeffecients $a_0 = 1, a_1 = 0, a_2 = 2, a_3 = 0, a_4 = 5$

## Method 1: Using List

---

import numpy as np
import matplotlib.pyplot as plt

# numpy is used for efficient array (vector or matrix) operations
# pyplot is used for plotting 
# Must read: [https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm]

a = [1, 0, 2, 0, 5] # coeffecients of the polynomial
n = len(a) - 1 # degree. Remember: number of coeff = degree + 1

'''
For a single value of x, finding p(x)

Note that this is an example of block comment in python. A block comment 
starts with three ' and ends with three '.
'''

x = 5.0
p_x = 0.0

for i in range(n + 1):
    '''
    p_x = a[i] * x # WRONG, because no power
    p_x = a[i] * (x ** i) # WRONG, have to add the terms
    '''
    p_x += a[i] * (x ** i) # a ** b means pow(a, b) or a^b

'''
For an array of x, finding p(x) for each element
'''

x_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
p_x_arr = []

'''
# naive way:
for i in range(len(x_arr)):
    print(x_arr[i])
'''

# better way: array traversing
for x in x_arr:
    temp = 0.0
    for i in range(n + 1):
        temp += a[i] * (x ** i)
    
    p_x_arr.append(temp) # array er last e insert kore dao
    

print("p({}) =".format(x_arr), p_x_arr)
# note how we formatted the string. A formatted string starts with 'f'.

# Using numpy array for vectorization
import numpy as np 
# numpy is used for efficient array (vector or matrix) operations
# Must read: [https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm]


a = np.array([1, 0, 2, 0, 5])
x_arr = np.array([1, 2, 3, 4, 5])
p_x_arr = 0.0

# vectorized version. requires only one loop
for i in range(n + 1):
    p_x_arr += a[i] * (x_arr ** i) # a ** b means pow(a, b) or a^b
    
print("p({}) =".format(x_arr), p_x_arr)



## Method 2 (Better): Using a Class

---
Complete the implementation of the polynomial class as showed in the lecture

'''
Lab task 1
Here we implement a Polynomial class with three methods: the constructor
__init__(), the toString method __repr__(), and a method to make the objects
of the class callable, __call__() method
'''

# Polynomial Class

class Polynomial:
    # Constructor, note that it starts and ends with two underscores
    def __init__(self, coeff):
        '''
        Every internal variable of the object must be saved and initialized
        in this method: self.variable = value
        '''
        self.coeff = coeff
        self.degree = len(coeff) - 1

    # Method to make the object callable 
    def __call__(self, x_arr):
        '''
        Here we assumed x_arr is a numpy array. Remember that a numpy array acts 
        like a vector (1D matrix). So an operation x + 1 would add 1 to each element
        of the matrix (unlike python's defaule list). Simlarly, x ** 2 would return
        element wise square of the array. 

        Hence, this method would return an array, where the i'th element is the 
        (polynomial) interpolated value of x[i], given the coeffecients a[i].
        '''
        p_x_arr = 0
        # --------------------------------------------
        # HINT: Should look like
        # for i in range(self.degree + 1):
        #     ????
        # --------------------------------------------

        # remember 1: length = degree + 1 for a polynomial
        # remember 2: range(0, a) is same as range(a)
        # remember 3: range(a, b) means a is inclusive, b is exclusive

        # --------------------------------------------
        
        # YOUR CODE HERE
        
        for i in range(self.degree + 1):
            p_x_arr += self.coeff[i] * (x_arr ** i) # a ** b means pow(a, b) or a^b
        return p_x_arr
        # --------------------------------------------

    # String representation method of the object (similar to toString() of java)
    def __repr__(self):
        str_ret = 'Polynomial of degree {}\np(x) = '.format(self.degree)
        for i in range(self.degree + 1):
            a = self.coeff[i]
            if i != 0:
                if a >= 0:
                    str_ret += '+ {}x^{} '.format(a, i)
                else:
                    str_ret += '- {}x^{} '.format(-a, i)
            else:
                str_ret += '{}x^{} '.format(a, i)

        return str_ret

    # custom method 1: to get the degree of the polynomial
    def get_degree(self):
        # --------------------------------------------
        return len(p_x_arr)-1
        # --------------------------------------------

    # custom method 2: to get the coefficients of the polynomial
    def get_coeffs(self):
        # --------------------------------------------
        return self.coeff
        # --------------------------------------------

# This block is used for testing, it should run without any error.

p = Polynomial(np.array([1.0, 0.0, 2.0, 0.0, 5.0]))
assert p(1) == 8
assert p(6) == 6553
test_x = np.array([1, 6])
test_p_x = np.array([8, 6553])
np.testing.assert_array_equal(p(test_x), test_p_x)

assert p.get_degree() == 4
np.testing.assert_array_equal(p.get_coeffs(),  [1.0, 0.0, 2.0, 0.0, 5.0])


# an example to see if our implementation works
coeff = np.array([1.0, 0.0, 2.0, 0.0, 5.0])
p = Polynomial(coeff)
print(p)  # check if printable
x_arr = np.array([1, 2, 3, 4, 5, 6])
print()
print("p({}) =".format(x_arr), p(x_arr)) # check if the object is callable
# should print p([1 2 3 4 5]) =  [   8.   89.  424. 1313. 3176.]

# Part 2: Polynomial Interpolation (Matrix Method)

If we have  $n+1$ nodes, that is,  $\{(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_{n}, y_{n})\}$ that satisfies a polynomial of order $n$, it can be written as:

\begin{align}
&a_0 + a_1  x_0 + a_2  x_0^2 + \cdots a_n  + x_0^n = y_0\\
&a_0 + a_1  x_1 + a_2  x_1^2 + \cdots a_n  + x_1^n = y_1\\
&a_0 + a_1  x_2 + a_2  x_2^2 + \cdots a_n  + x_2^n = y_2\\
&\cdots\\
&a_0 + a_1  x_{n-1} + a_2  x_{n}^2 + \cdots + a_n  x_{n}^n = y_{n}\\
\end{align}

Here, $p(x) = a_0 + a_1x^1 + a_2x^2 + \cdots a_nx^n$ is called the fitted polynomial of the given data points (nodes). Using this polynomial to find the $y_k$ corresponding to an $x_k$ with the range of the given nodes is called polynomial interpolation.

In matrix form, the equations can be written as  $$\mathbf{Xa = y}$$

where $\mathbf{X} = $

\begin{bmatrix}
x_0^0 & x_0^1 & x_0^2 & \cdots & x_0^n\\
x_1^0 & x_1^1 & x_1^2 & \cdots & x_1^n\\
\vdots\\
x_n^0 & x_{n}^1 & x_n^2 & \cdots & x_n^n\\
\end{bmatrix}

$\mathbf{a} = $
\begin{bmatrix}
a_0\\
a_1\\
\vdots\\
a_n
\end{bmatrix}

and $\mathbf{y} = $
\begin{bmatrix}
y_0\\
y_1\\
\vdots\\
y_n
\end{bmatrix}

From this, we can solve for $\mathbf{a}$ using
$$\mathbf{a = X^{-1}y}$$



'''
Lab task 2
Here we implement a function which takes a discrete x and y array, and returns
a Polynomial object (the one we just implemented). This polynomial object can 
be used to calculate y for any other value of x (not in that list) within the
range
'''
def get_poly(data_x, data_y):
    n_nodes = len(data_x)
    # np.zeors( (a, b) ) returns a (a x b) matrix, i.e., a rows and b columns 
    X = np.zeros( (n_nodes, n_nodes) )

    # See the lecture video how the matrix is formed
    # --------------------------------------------
    # Hint: The code will like like this:
    # for i in range(n_nodes):
    #   for j in range(n_nodes):
    #     X[i, j] = ????
    # --------------------------------------------
    for i in range(n_nodes):
        for j in range(n_nodes):
            X[i, j] = data_x[i]**j
    # --------------------------------------------
    # We could have also used np.linalg.inv to find the inverse
    # but pinv is more efficient
    X_inv = np.linalg.pinv(X) #pseudo inverse
    a = np.dot(X_inv, data_y)
    p = Polynomial(a)

    return p

# This block is used for testing. It should run without any error

data_x = np.array([-3, -2, -1, 0, 1, 3])
data_y = np.array([-80., -13., 6., 1., 5., 16.])
p = get_poly(data_x, data_y)

assert p.degree == 5
np.testing.assert_array_almost_equal(p.coeff, [ 1., -5.075, 5.52083333, 4.85416667, -1.02083333, -0.27916667])
np.testing.assert_almost_equal(p(-1.5), 1.603515625)
np.testing.assert_almost_equal(p(2), 26.5)



data_x = np.array([-3., -2., -1., 0., 1., 3.])
data_y = np.array([-80., -13., 6., 1., 5., 16.])
p = get_poly(data_x, data_y)
'''
np.linspace(a, b, n) returns a numpy array of n points equally 
spaced from a to b
'''
x_arr = np.linspace(-3, 3, 100)
# interpolated values
y_interp = p(x_arr)

# pyplot is used for plotting 
# Must read: [https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm]

print(y_interp)

plt.plot(x_arr, y_interp, 'r')
plt.plot(data_x, data_y, 'go')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
    

plt.show()
# You should get a smooth graph that fits the given data points

import numpy as np
import matplotlib.pyplot as plt
data_x=np.array([.014,.016,.018,.020,.022, .024,.026,.028,.030,.032])
data_y=np.array([275.129, 210.646,166.436,134.813,111.416, 93.620,79.771,68.782,59.917,52.661])
data_x2=np.array([.014,.016,.018,.020,.022, .024,.026,.028,.030,.032])
data_y2=np.array([330.155,252.775,199.723,161.776,133.669,112.344,95.725,82.539,71.900,63.194])
plt.plot(data_x, data_y, 'r')
plt.plot(data_x, data_y, 'go')
plt.plot(data_x2, data_y2, "b")
plt.plot(data_x2, data_y2, "go")
plt.xlabel('Distance, r, meter \n (Graph made by 19201081) \n Red is Curve 1 and Blue is Curve 2')
plt.ylabel('Electrostatic Force, Fe, Newton')
plt.grid(b=True, which='major', axis='both')
'''plt.figure(figsize=(192,108), dpi=10)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)
'''
plt.show()


330.1548,252.7748,199.7233,161.7759,133.6991,112.3443,95.7254,82.5387,71.9004,63.1937

