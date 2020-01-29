NumPy has an excellent `basic tutorial
<http://www.scipy.org/Tentative_NumPy_Tutorial>`_ available.  Here I just copy
the Quick Tour from that tutorial but you should read the rest as well.  In
these examples the python prompt is shown as ">>>" in order to distinguish the
input from the outputs.

Arrays can be created in different ways::

  >>> a = array( [ 10, 20, 30, 40 ] )   # create an array out of a list
  >>> a
  array([10, 20, 30, 40])
  >>> b = arange( 4 )                   # create an array of 4 integers, from 0 to 3
  >>> b
  array([0, 1, 2, 3])
  >>> c = linspace(-pi,pi,3)            # create an array of 3 evenly spaced samples from -pi to pi
  >>> c
  array([-3.14159265,  0.        ,  3.14159265])

New arrays can be obtained by operating with existing arrays::

  >>> d = a+b**2                        # elementwise operations
  >>> d
  array([10, 21, 34, 49])

Arrays may have more than one dimension::

  >>> x = ones( (3,4) )
  >>> x
  array([[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]])
  >>> x.shape                            # a tuple with the dimensions
  (3, 4)

and you can change the dimensions of existing arrays::

  >>> y = arange(12)
  >>> y
  array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
  >>> y.shape = 3,4              # does not modify the total number of elements
  >>> y
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

It is possible to operate with arrays of different dimensions as long as they fit well (broadcasting)::

  >>> 3*a                                # multiply each element of a by 3
  array([ 30,  60,  90, 120])
  >>> a+y                                # sum a to each row of y
  array([[10, 21, 32, 43],
         [14, 25, 36, 47],
         [18, 29, 40, 51]])

Similar to Python lists, arrays can be indexed, sliced and iterated over::

  >>> a[2:4] = -7,-3                     # modify last two elements of a
  >>> for i in a:                        # iterate over a
  ...     print i
  ...
  10
  20
  -7
  -3

When indexing more than one dimension, indices are separated by commas::

  >>> x[1,2] = 20
  >>> x[1,:]                             # x's second row
  array([ 1,  1, 20,  1])
  >>> x[0] = a                           # change first row of x
  >>> x
  array([[10, 20, -7, -3],
         [ 1,  1, 20,  1],
         [ 1,  1,  1,  1]])
