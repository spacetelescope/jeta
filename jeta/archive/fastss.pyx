import numpy as np
import cython
cimport numpy as np

DTYPE = np.int

ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE_double_t

@cython.wraparound(False)
@cython.boundscheck(False)
def _search_both_sorted(np.ndarray[dtype=DTYPE_double_t, ndim=1] a not None,
                       np.ndarray[dtype=DTYPE_double_t, ndim=1] v not None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would
    be preserved.

    Parameters
    ----------
    a : 1-D array_like
        Input array, sorted in ascending order.
    v : array_like
        Values to insert into `a`.
    """
    cdef int nv = v.shape[0]
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] idx = np.empty(nv, dtype=DTYPE)
    cdef int na = a.shape[0]
    cdef unsigned int ia = 0
    cdef unsigned int iv
    cdef double vi

    for iv in range(nv):
        vi = v[iv]
        while True:
            if ia < na:
                if vi <= a[ia]:
                    idx[iv] = ia
                    break
                else:
                    ia += 1
            else:
                idx[iv] = na
                break

    return idx

@cython.wraparound(False)
@cython.boundscheck(False)
def _nearest_index(np.ndarray[dtype=DTYPE_double_t, ndim=1] a not None,
                   np.ndarray[dtype=DTYPE_double_t, ndim=1] v not None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would
    be preserved.

    Parameters
    ----------
    a : 1-D array_like
        Input array, sorted in ascending order.
    v : array_like
        Values to insert into `a`.
    """
    cdef int nv = v.shape[0]
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] idx = np.empty(nv, dtype=DTYPE)
    cdef int na = a.shape[0]
    cdef unsigned int ia = 0
    cdef unsigned int iv
    cdef double vi

    for iv in range(nv):
        vi = v[iv]
        while True:
            if ia < na:
                if vi <= a[ia]:
                    if ia == 0:
                        idx[iv] = ia
                    elif abs(vi - a[ia - 1]) < abs(vi - a[ia]):
                        idx[iv] = ia - 1
                    else:
                        idx[iv] = ia
                    break
                else:
                    ia += 1
            else:  # ia == na without vi ever being less than a[ia]
                   # Thus vi > all values of a
                idx[iv] = na - 1
                break

    return idx

@cython.wraparound(False)
@cython.boundscheck(False)
def _interp_linear(np.ndarray[dtype=DTYPE_double_t, ndim=1] yin not None,
                   np.ndarray[dtype=DTYPE_double_t, ndim=1] a not None,
                   np.ndarray[dtype=DTYPE_double_t, ndim=1] v not None):
    """
    yout = (xout - x0) / (x1 - x0) * (y1 - y0) + y0

    Parameters
    ----------
    yin : 1-D array_like
         Input y values
    a : 1-D array_like
         Input x values corresponding to yin
    v : 1-D array_like
         Output x values corresponding to yout
    """
    cdef double vi
    cdef int nv = v.shape[0]
    cdef np.ndarray[dtype=DTYPE_double_t, ndim=1] yout = np.empty(nv, dtype=np.float64)
    cdef unsigned int na = a.shape[0]
    cdef unsigned int ia = 0
    cdef unsigned int iv
    cdef unsigned int na1 = na - 1
    cdef unsigned int na2 = na - 2
    cdef unsigned int ia1

    for iv in range(nv):
        vi = v[iv]
        while True:
            if ia < na:
                if vi <= a[ia]:
                    if ia == 0:
                        yout[iv] = (vi - a[0]) / (a[1] - a[0]) * (yin[1] - yin[0]) + yin[0]
                    else:
                        ia1 = ia - 1
                        yout[iv] = (vi - a[ia1]) / (a[ia] - a[ia1]) * (yin[ia] - yin[ia1]) + yin[ia1]
                    break
                else:
                    ia += 1
            else:  # ia == na without vi ever being less than a[ia]
                   # Thus vi > all values of a
                yout[iv] = (vi - a[na1]) / (a[na1] - a[na2]) * (yin[na1] - yin[na2]) + yin[na1]
                break

    return yout
