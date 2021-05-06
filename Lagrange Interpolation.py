Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).

Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

NAME = "Voyager0001"
COLLABORATORS = ""

---

# CSE330 Lab: Polynomial Interpolation using Lagrange Form
---

### Importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt

### The Lagrange_Polynomial class
General form of an $n$ degree Lagrange polynomial:

\begin{equation}
p_n(x) = \sum_{k=0}^{n} f(x_k)l_k(x) = \sum_{k=0}^{n} y_kl_k(x),\tag{1}
\end{equation}

where
\begin{equation}
l_k(x) = \prod_{j=0, j\neq k}^{n} \frac{x-x_j}{x_k-x_j}. \tag{2}
\end{equation}

Note that the Lagrange method is more efficient than the matrix method because *we do not need to calculate any inverse matrices*.

1. **The constructor `__init__(self, data_x, data_y)` is written for you.**
    
     * Here, we check whether the input vectors (numpy arrays) are equal or not. 
     * We store `data_x` and `data_y`
     * We calculate and store the degree of the polynomial.
$$\$$

2. **The `_repr__(self)` function has been written for you.**

    * This is similar to the `toString()` method in Java. This returns a formatted string of the object whenever the object is printed.
$$\$$

3. **You have to implement the `l(self, k, x)` function.**
    * This function would take `k` and `x` as inputs and calculate the Lagrange basis using the Equation $(2)$.
$$\$$

4. **You have to implement the `__call__(self, x_arr)` function.** 
    * This function makes an object of a class callable.
    * The function calculates the lagrange polynomial from a set of given nodes. `self.data_x` and `self.data_y` contains the coordinates of the given nodes of the original function. Using Equation $(1)$, you have to use `self.data_x`, `self.data_y`, and the `l(k, x_k, x)` function to find the interpolated output of the polynomial for all elements of `x_arr`.
`x_arr` is a numpy array containing points through which we want to plot our polynomial.

class Lagrange_Polynomial:
    def __init__(self, data_x, data_y):
        '''
        First we need to check whether the input vectors (numpy arrays) are equal 
        or not. 
        assert (condition), "msg"
        this command checks if the condition is true or false. If true, the code 
        runs normally. But if false, then the code returns an error message "msg"
        and stops execution
        '''
        assert len(data_x) == len(data_y), "length of data_x and data_y must be equal"

        '''
        Lagrange polynomials do not use coefficeints a_i, rather the nodes 
        (x_i, y_i). Hence, we just need to store these inside the object
        '''

        self.data_x = data_x
        self.data_y = data_y

        self.degree = len(data_x) - 1
        # we assume that the inputs are numpy array, so we can perform 
        # element wise operations

    def __repr__(self):
        # method for string representation
        # you don't need to worry about the following code if you don't understand
        strL = f"LagrangePolynomial of order {self.degree}\n"
        strL += "p(x) = "
        for i in range(len(self.data_y)):
            if self.data_y[i] == 0:
                continue
            elif self.data_y[i] >= 0:
                strL += f"+ {self.data_y[i]}*l_{i}(x) "
            else:
                strL += f"- {-self.data_y[i]}*l_{i}(x) "

        return strL
  
    def l(self, k, x):
        '''
        This metod implements the Lagrange Basis to be used for interpolation
        using Lagrange Polynomials. You can implement this either using loop
        or without using any loops (thanks to numpy and vectorization)
        '''

        l_k = 1.0 # Initialization

        # --------------------------------------------
        # YOUR CODE HERE
        x_k = self.data_x[k]   
        for j in range(self.degree + 1):   
          x_j = self.data_x[j]
          if x_j!=x_k:
            l_k *= (x-x_j)/(x_k-x_j)
        
        # 
        # HINT FOR LOOP METHOD: Should look like
        # x_k = self.data_x[k]
        # for j in range(self.degree + 1):
        #    l_k *= ?????
        #
        # HINT FOR VECTORIZED METHOD (no loops): 
        #   Google how to use np.prod and np.delete 
        # l_k = np.prod(?? np.delete(??) ??) /  np.prod(?? np.delete(??) ??)
        # --------------------------------------------
        return l_k


    def __call__(self, x_arr):
        """
        The method to make the object callable (see the code of the matrix method).
        'x_arr' is a set of given points (a numpy array). You have to use 
        self.data_x and self.data_y to find the interpolated output of the 
        polynomial for all elements of 'x_arr'.

        Implement as you wish but your 'total' numpy array where the i'th element
        p_x_arr[i] represents the interpolated value of p(x_arr[i]).
        """

        # initialize with zero
        p_x_arr  = np.zeros(len(x_arr))

        # --------------------------------------------
        # YOUR CODE HERE
        for i, x in enumerate(x_arr):
            for k in range(self.degree + 1):
               p_x_arr[i] += self.data_y[k]* self.l(k, x)
        
        # 
        # HINT: Should look like
        # for i, x in enumerate(x_arr):
        #   for k in range(self.degree + 1):
        #       ??????
        #       p_x_arr[i] = ??? self.data_y[k] ??? self.l(k, x)
        # --------------------------------------------

        return p_x_arr

### Calling the LagrangePolynomial object and plotting the polynomial.
First we create a lagrange polynomial object `p` by calling `Lagrange_Polynomial(data_x, data_y)`. Then, we call the object as a function, which is possible because we had implemented the `__call__` function in the Lagrange_Polynomial class, and pass in `x_arr`. `x_arr` is 1D numpy array (a vector), which we created using linspace function and contains $x_i$, i.e., the points through which we want to plot our polynomial. Calling the object as a function and inputting `x_arr` returns the corresponding $y_i$ values and stores them in the `p_x_arr` numpy array.

Finally, the polynomial is plotted by passing in `x_arr` and `p_x_arr` in plt.plot(), i.e., the $x_i$ and $y_i$ pairs.

*Note that in the plot the given nodes will be marked in red.*

data_x = np.array([-3., -2., -1., 0., 1., 3., 4.])
data_y = np.array([-60., -80., 6., 1., 45., 30., 16.])

p = Lagrange_Polynomial(data_x, data_y)
print(p)

#generating 100 points from -3 to 4 in order to create a smooth line
x_arr = np.linspace(-3, 4, 50)
p_x_arr = p(x_arr)


np.testing.assert_array_almost_equal(p([1]), [45.])
np.testing.assert_array_almost_equal(p([2]), [112.38095238])
np.testing.assert_array_almost_equal(p([1, 2, 3]), [ 45., 112.38095238, 30.])


# plot to see if your implementation is correct
#google the functions to understand what each parameters mean, if not apparent
plt.plot(x_arr, p_x_arr)
plt.plot(data_x, data_y, 'ro')
plt.legend(['interpolated', 'node points'], loc = 'lower right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Polynomial')

plt.show()


