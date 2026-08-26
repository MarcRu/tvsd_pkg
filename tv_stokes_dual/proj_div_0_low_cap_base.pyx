
import cython
from cython.parallel import prange
from libc.math cimport M_PI_2, cos, sin, sqrt

import numpy as np
cimport numpy as np
#import time

"""
===================================================================
This Python-file provides classes to determine the partial
projection on the subspace with divB(...)=0 (and the partial
Moore-Penrose-Inverse of the discrete Laplace-operator with 
Neumann-boundary-conditions), which is needed to run the projection
parallely on low-RAM-processors.
To see a definition for divB, see proj_div_0.pyx.
The actual functionality is contained in this Cython-file.

!!!
It is strongly recommended though to use the classes from
proj_div_0_low_cap.py to access the functions in this file. 
They were developed as interface for easier access.
!!!
===================================================================
"""



"""
------------------------------------------------------------------
PARTIAL DISCRETE COSINE TRANSFORMS
------------------------------------------------------------------


These functions evaluate the partial DCT of 2D-arrays in 
   X- and Y-direction. More precisely, the DCT of arrays
           (0 0 ...    0   ... 0)
           (: :        :       :)
   MAT :=  (0 0 ... mat_in ... 0)
           (: :        :       :)
           (0 0 ...    0   ... 0)
   get evaluated only in the block mat_in.


NOTICE! THIS IS A NAIVE IMPLEMENTATION OF DCT!
   THIS CAN PROBABLY BE OPTIMIZED BY ADAPTING THE 
   FAST-DCT-ALGORITHM TO PARTIAL DCTs. 
   FOR FURTHER IMPROVEMENT IT MIGHT MAKE SENSE TO PREDETERMINE
   SOME COSINE VALUES.   
"""


@cython.boundscheck(False)
cpdef void pdct_x(double [:, :] mat_in, double [:, :] mat_out, \
            int N1, int blocksize_y, \
            int lbound_in, int rbound_in, int lbound_out, int rbound_out) \
            nogil:
    """
    Discrete Partial Cosine Transform in x direction
    Args:
        mat_in: 2D-array with shape (blocksize_y, rbound_in - lbound_in)
        mat_out: 2D-array with shape (blocksize_y, rbound_out - lbound_out)
        N1: Size of big array MAT in x-direction
        blocksize_y: Size of smaller arrays mat_in, mat_out in y-direction
        lbound_in, rbound_in, lbound_out, rbound_out:
                    Bounds of mat_in and mat_out in x-direction
    """

    # set the variable extension types
    cdef int i, j1, j2
    cdef double prefactor = sqrt(2. / <double>N1)
    cdef double prefactor0 = sqrt(1. / <double>N1)
    cdef double PI_2N1 = M_PI_2 / <double>N1
    cdef double cos_j1_j2

    #Initialize mat_out
    for j1 in prange(lbound_out, rbound_out):
        for i in range(blocksize_y):
            mat_out[i, j1 - lbound_out] = 0.0

    #Compute dct
    for j1 in range(lbound_out, rbound_out):
        if (j1 == 0):
            for i in range(blocksize_y):
                mat_out[i, 0] = 0.0
                for j2 in range(lbound_in, rbound_in):
                    mat_out[i, j1-lbound_out] += prefactor0 * mat_in[i, j2 - lbound_in]
        else:
            for i in range(blocksize_y):
                mat_out[i, j1 - lbound_out] = 0.0
            for j2 in range(lbound_in, rbound_in):
                cos_j1_j2 = cos(<double>(j1 * (2 * j2 + 1)) * PI_2N1)
                for i in range(blocksize_y):
                    mat_out[i, j1 - lbound_out] += prefactor * \
                            cos_j1_j2 * mat_in[i, j2 - lbound_in]

    return


@cython.boundscheck(False)
cpdef void pdct_y(double [:, :] mat_in, double [:, :] mat_out, \
            int N2, int blocksize_x, \
            int ubound_in, int bbound_in, int ubound_out, int bbound_out) \
            nogil:
    """
    Discrete Partial Cosine Transform in y direction
    Args:
        mat_in: 2D-array with shape (bbound_in - ubound_in, blocksize_x)
        mat_out: 2D-array with shape (bbound_out - ubound_out, blocksize_x)
        N2: Size of big array MAT in y-direction
        blocksize_x: Size of smaller arrays mat_in, mat_out in x-direction
        ubound_in, bbound_in, ubound_out, bbound_out:
                    Bounds of mat_in and mat_out in y-direction
    """


    # set the variable extension types
    cdef int i1, i2, j
    cdef double prefactor = sqrt(2. / <double>N2)
    cdef double prefactor0 = sqrt(1. / <double>N2)
    cdef double PI_2N2 = M_PI_2 / <double>N2
    cdef double cos_i1_i2

    #Initialize mat_out
    for i1 in prange(ubound_out, bbound_out):
        for j in range(blocksize_x):
            mat_out[i1 - ubound_out, j] = 0.0

    #Compute dct
    for i1 in prange(ubound_out, bbound_out):
        if (i1 == 0):
            for j in range(blocksize_x):
                mat_out[0, j] = 0.0
                for i2 in range(ubound_in, bbound_in):
                    mat_out[i1-ubound_out, j] += prefactor0 * mat_in[i2 - ubound_in, j]
        else:
            for j in range(blocksize_x):
                mat_out[i1 - ubound_out, j] = 0.0
            for i2 in range(ubound_in, bbound_in):
                cos_i1_i2 = cos(<double>(i1 * (2 * i2 + 1)) * PI_2N2)
                for j in range(blocksize_x):
                    mat_out[i1 - ubound_out, j] += prefactor * \
                            cos_i1_i2 * mat_in[i2 - ubound_in, j]

    return


@cython.boundscheck(False)
cpdef void pidct_x(double [:, :] mat_in, double [:, :] mat_out, \
            int N1, int blocksize_y, \
            int lbound_in, int rbound_in, int lbound_out, int rbound_out) \
            nogil:
    """
    Inverse Discrete Partial Cosine Transform in x direction
    Args:
        mat_in: 2D-array with shape (blocksize_y, rbound_in - lbound_in)
        mat_out: 2D-array with shape (blocksize_y, rbound_out - lbound_out)
        N1: Size of big array MAT in x-direction
        blocksize_y: Size of smaller arrays mat_in, mat_out in y-direction
        lbound_in, rbound_in, lbound_out, rbound_out:
                    Bounds of mat_in and mat_out in x-direction
    """

    # set the variable extension types
    cdef int i, j1, j2
    cdef double prefactor = sqrt(2. / <double>N1)
    cdef double prefactor0 = sqrt(1. / <double>N1)
    cdef double PI_2N1 = M_PI_2 / <double>N1
    cdef double cos_j1_j2

    #Initialize mat_out
    for j1 in prange(lbound_out, rbound_out):
        for i in range(blocksize_y):
            mat_out[i, j1 - lbound_out] = 0.0

    #Compute dct
    for j1 in prange(lbound_out, rbound_out):
        for i in range(blocksize_y):
            mat_out[i, j1 - lbound_out] = 0.0
        for j2 in range(lbound_in, rbound_in):
            if (j2 == 0):
                for i in range(blocksize_y):
                    mat_out[i, j1 - lbound_out] += prefactor0 * mat_in[i, j2 - lbound_in]
            else:
                cos_j1_j2 = cos(<double>(j2 * (2 * j1 + 1)) * PI_2N1)
                for i in range(blocksize_y):
                    mat_out[i, j1 - lbound_out] += prefactor * \
                            cos_j1_j2 * mat_in[i, j2 - lbound_in]

    return


@cython.boundscheck(False)
cpdef void pidct_y(double [:, :] mat_in, double [:, :] mat_out, \
            int N2, int blocksize_x, \
            int ubound_in, int bbound_in, int ubound_out, int bbound_out) \
            nogil:
    """
    Inverse Discrete Partial Cosine Transform in y direction
    Args:
        mat_in: 2D-array with shape (bbound_in - ubound_in, blocksize_x)
        mat_out: 2D-array with shape (bbound_out - ubound_out, blocksize_x)
        N2: Size of big array MAT in y-direction
        blocksize_x: Size of smaller arrays mat_in, mat_out in x-direction
        ubound_in, bbound_in, ubound_out, bbound_out:
                    Bounds of mat_in and mat_out in y-direction
    """

    # set the variable extension types
    cdef int i1, i2, j
    cdef double prefactor = sqrt(2. / <double>N2)
    cdef double prefactor0 = sqrt(1. / <double>N2)
    cdef double PI_2N1 = M_PI_2 / <double>N2
    cdef double cos_i1_i2

    for i1 in prange(ubound_out, bbound_out):
        for j in range(blocksize_x):
            mat_out[i1 - bbound_out, j] = 0.0

    #Compute dct
    for i1 in prange(ubound_out, bbound_out):
        for j in range(blocksize_x):
            mat_out[i1 - ubound_out, j] = 0.0
        for i2 in range(ubound_in, bbound_in):
            if (i2 == 0):
                for j in range(blocksize_x):
                    mat_out[i1 - ubound_out, j] += prefactor0 * mat_in[i2 - ubound_in, j]
            else:
                cos_i1_i2 = cos(<double>(i2 * (2 * i1 + 1)) * PI_2N1)
                for j in range(blocksize_x):
                    mat_out[i1 - ubound_out, j] += prefactor * \
                            cos_i1_i2 * mat_in[i2 - ubound_in, j]

    return




"""
------------------------------------------------------------------
Prepare Sigma squared
------------------------------------------------------------------
"""



@cython.boundscheck(False)
cpdef void prepare_Sigma_sq(double [:] mat_out, double h):
    """
    Returns the diagonal entries of Sigma^2-matrix, which is needed
    to compute the Moore-Penrose-Inverse of the Laplacian with
    Neumann-boundaries in the image space of the
    Discrete Cosine Transform.
    The size N of Sigma^2 is determined from the allocated space of
    the given mat_out Array. The first entry (mat_out[0]) is zero.
    All other entries mat_out[1],...,mat_out[N-1] are the diagonal
    entries of Sigma^2.
    """

    cdef int i, N
    cdef double prefactor, sinfactor

    N = mat_out.shape[0]
    prefactor = 4.0 / (h * h)
    sinfactor = M_PI_2 / <double>(N)

    mat_out[0] = 0.0
    for i in prange(1, N, nogil=True):
        mat_out[i] = prefactor * sin(sinfactor * <double>(i)) ** 2

    return

"""
------------------------------------------------------------------
INVERSE LAPLACIAN WITH NEUMANN BOUNDARIES
------------------------------------------------------------------
"""

#Needs predetermined diagonal entries of the matrices
#  Sigma1^2 and Sigma2^2 in the paper.
@cython.boundscheck(False)
cpdef void inv_laplacian_dct_img_sparse(
            double [:, :] mat_in, double [:, :] mat_out,
            double [:] Sigma1_sq, double [:] Sigma2_sq,
            int lbound, int ubound,
            int N1_loc, int N2_loc) \
            nogil:
    """
    Computes a block of the Moore-Penrose-Inverse of the Laplacian with
    Neumann-boundaries in the image space of the
    Discrete Cosine Transform.
    Args:
        mat_in: block of the input-matrix
        mat_out: block of the output-matrix
        Sigma1_sq: Predetermined factors x-direction (see prepare_Sigma_sq)
        Sigma2_sq: Predetermined factors y-direction (see prepare_Sigma_sq)
        lbound: left bound of block
        ubound: upper bound of block
        N1_loc: number of pixel columns of block
        N2_loc: number of pixel rows of block
    """
    # set the variable extension types
    cdef int j, i

    for i in prange(0, N2_loc):
        if (i + ubound == 0):
            for j in range(0, N1_loc):
                if (j + lbound == 0):
                    mat_out[0,0] = 0.0
                else:
                    mat_out[0,j] = (-1.0) * mat_in[0,j] / Sigma1_sq[j]
        else:
            for j in range(0, N1_loc):
                if (j + lbound == 0):
                    mat_out[i,0] = (-1.0) * mat_in[i,0] / Sigma2_sq[i]
                else:
                    mat_out[i,j] = (-1.0) * mat_in[i,j] / \
                            (Sigma1_sq[j] + Sigma2_sq[i])
    
    return


@cython.boundscheck(False)
cpdef double [:, :] inv_laplacian_neum_sparse_general( \
                double [:, :] mat_in, double [:, :] mat_out,
                double [:, :] tmp_mat1, double [:, :] tmp_mat2,
                double [:] Sigma1_sq, double [:] Sigma2_sq,
                unsigned int [:] lbounds_in, unsigned int [:] rbounds_in,
                unsigned int [:] ubounds_in, unsigned int [:] bbounds_in,
                unsigned int [:] sizes1_in, unsigned int [:] sizes2_in,
                unsigned int [:] lbounds_out, unsigned int [:] rbounds_out,
                unsigned int [:] ubounds_out, unsigned int [:] bbounds_out,
                unsigned int [:] sizes1_out, unsigned int [:] sizes2_out,
                int N1, int N2, int M1, int M2, 
                int k1, int k2, int m1, int m2):

    """
    Computes a block of the Moore-Penrose-Inverse of the Laplacian with
    Neumann-boundaries.
    It is strongly recommended to use the methods part_inv_sparse(...) or
    part_inv_sparse_general(...) of the InvLapl-class in proj_div_0_low_cap.py
    to call this function. 
    Args:
        mat_in:     matrix at block (k2, k1), only non-zero-part
        mat_out:    inverse lapl matrix at block (m2, m1)
        Sigma_sq:   Prepared matrix Sigma^2 (see notes or further up) to evaluate
                    inverse laplacian in the image of the DCT
        tmp_mat:    Allocated space. Must (each) at least have the number of
                    rows and columns of the biggest blocks.
        N2, N1:     number of rows and columns of big matrix
        M2, M1:     number of rows and columns of blocks
        (k2, k1):   indices of input block (k2 row, k1 column)
        (m2, m1):   indices of output block (m2 row, m1 column)
        bounds_in:  left, right, upper and bottom bounds of input blocks
        bounds_out: left, right, upper and bottom bounds of output blocks
        sizes:      sizes in x- (sizes1) and y- (sizes2) direction
                    The parameters are so general, that the whole matrix can
                    be divided into M2xM1 different blocks for the input-matrix
                    (lbounds_in, sizes1_in, ...) and for the output-matrix
                    (lbounds_out, sizes1_out, ...).
    """

    cdef int i, j, l1, l2
    cdef int size2m2 = sizes2_out[m2]
    cdef int size1m1 = sizes1_out[m1]

    #Initialize mat_out
    for i in prange(size2m2, nogil=True):
        for j in range(size1m1):
            mat_out[i,j] = 0.0

    for l2 in prange(M2, nogil=True):
        for l1 in prange(M1):

            #Apply Cosine transform in x-direction on mat_in
            pdct_x(mat_in, tmp_mat1, N1, sizes2_in[k2], \
                lbounds_in[k1], rbounds_in[k1], lbounds_in[l1], rbounds_in[l1])
            
            #Apply Cosine transform in y-direction
            pdct_y(tmp_mat1, tmp_mat2, N2, sizes1_in[l1], \
                ubounds_in[k2], bbounds_in[k2], ubounds_in[l2], bbounds_in[l2])    
            
            #Determine inv laplace in Image space of Cosine transform
            inv_laplacian_dct_img_sparse(tmp_mat2, tmp_mat1,
                Sigma1_sq[lbounds_in[l1]:rbounds_in[l1]],
                Sigma2_sq[ubounds_in[l2]:bbounds_in[l2]], 
                lbounds_in[l1], ubounds_in[l2], 
                sizes1_in[l1], sizes2_in[l2])
            
            #Apply Cosine backtransform in x-direction
            pidct_x(tmp_mat1, tmp_mat2, N1, sizes2_in[l2], \
                lbounds_in[l1], rbounds_in[l1], lbounds_out[m1], rbounds_out[m1])
            
            #Apply Cosine backtransform in y-direction
            pidct_y(tmp_mat2, tmp_mat1, N2, sizes1_out[m1], \
                ubounds_in[l2], bbounds_in[l2], ubounds_out[m2], bbounds_out[m2])

            #At the end just add the result on mat_out
            for i in prange(size2m2):
                for j in range(size1m1):
                    mat_out[i,j] += tmp_mat1[i,j]
    
    return mat_out


"""
------------------------------------------------------------------
DIFFERENTIALOPERATORS gradF AND divB FOR DIFFERENT DOMAINS
------------------------------------------------------------------

CAREFUL! The functions diffF_x, diffF_y, diffB_x, diffB_y
   as well as the functions gradF and divB add
   the derivatives of mat_in on the (predefined) values of mat_out.
   The matrix mat_out must be initialized (not just allocated)
   before calling these functions.
"""



@cython.boundscheck(False)
cpdef void diffF_x(double [:, :] mat_in, double [:, :] mat_out,
                    int cols_out, int rows_out,
                    bint domain_right_boundary, double h) \
                    nogil:
    """
    Forward differences in x-direction uncutted or cutted.
    For domain_right_boundary=True,
    the output dimension is the same as the input dimension.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0  0)
    ( 0 -1  1  ...  0  0)
    ( |  |      \   |  |)
    ( 0  0  0  ... -1  1)
    ( 0  0  0  ...  0  0)
    For domain_right_boundary=False,
    the dimension reduces by one.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0 | 0)
    ( 0 -1  1  ...  0 | 0)
    ( |  |  |   \   | | |)
    ( 0  0  0  ...  1 | 0)
    ( 0  0  0  ... -1 | 1)
    Args:
        mat_in: input block
        mat_out: output block
        cols_out: number of columns of output matrix
        rows_out: number of rows of output matrix
        domain_right_boundary: see description above
        h: grid size
    """
    # set the variable extension types
    cdef int i, j
    cdef double prefactor = 1.0 / h

    # loop over the matrix
    for i in prange(rows_out):
        for j in range(0, cols_out - 1):
            mat_out[i, j] += prefactor * (mat_in[i, j + 1] - mat_in[i, j])

        #right boundary
        if (domain_right_boundary):
            #mat_out[i, cols_out - 1] += 0.0
            continue
        else: 
            #Domain is not at the right boundary:
            #   We use values right of the borders of mat_out
            mat_out[i, cols_out - 1] +=  prefactor * (mat_in[i, cols_out] - mat_in[i, cols_out - 1])
    
    return


@cython.boundscheck(False)
cpdef void diffF_y(double [:, :] mat_in, double [:, :] mat_out,
                    int cols_out, int rows_out,
                    bint domain_bottom_boundary, double h) \
                    nogil:
    """
    Forward differences in y-direction uncutted or cutted.
    For domain_bottom_boundary=True,
    the output dimension is the same as the input dimension.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0  0)
    ( 0 -1  1  ...  0  0)
    ( |  |      \   |  |)
    ( 0  0  0  ... -1  1)
    ( 0  0  0  ...  0  0)
    For domain_bottom_boundary=False,
    the dimension reduces by one.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0 | 0)
    ( 0 -1  1  ...  0 | 0)
    ( |  |  |   \   | | |)
    ( 0  0  0  ...  1 | 0)
    ( 0  0  0  ... -1 | 1)
    Args:
        mat_in: input block
        mat_out: output block
        cols_out: number of columns of output matrix
        rows_out: number of rows of output matrix
        domain_right_boundary: see description above
        h: grid size
    """
    # set the variable extension types
    cdef int i, j
    cdef double prefactor = 1.0 / h

    # loop over the matrix
    for i in prange(rows_out):
        if (i < rows_out - 1):
            for j in range(0, cols_out):
                mat_out[i, j] += prefactor * (mat_in[i + 1, j] - mat_in[i, j])

        #bottom boundary: i == rows - 1
        elif (domain_bottom_boundary):
            #for j in range(0, cols_out):
                #mat_out[rows - 1, j] += 0.0
            continue
        else: 
            #Domain is not at the bottom boundary:
            #We use values below the borders of mat_out
            for j in range(0, cols_out):
                mat_out[rows_out - 1, j] += prefactor * (mat_in[rows_out, j] - mat_in[rows_out - 1, j])
    
    return
        

@cython.boundscheck(False)
cpdef void diffB_x(double [:, :] mat_in, double [:, :] mat_out,
                    int cols_in, int rows_in,
                    bint domain_right_boundary, double h) \
                    nogil:
    """
    Backward differences in x-direction uncutted or cutted.
    Defined in a way that it is adjoint to the forward differences.
    For domain_right_boundary=True,
    the output dimension is the same as the input dimension.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0  0)
    (-1  1  ...  0  0  0)
    ( |  |   \   |  |  |)
    ( 0  0  ... -1  1  0)
    ( 0  0  ...  0 -1  0)
    For domain_right_boundary=False,
    the output dimension extends by 1.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0 )
    (-1  1  ...  0  0 )
    ( |  |   \   |  | )
    ( 0  0  ... -1  1 )
    (-----------------)
    ( 0  0  ...  0 -1 )
    Args:
        mat_in: input block
        mat_out: output block
        cols_in: number of columns of input matrix
        rows_in: number of rows of input matrix
        domain_right_boundary: see description above
        h: grid size
    """

    # set the variable extension types
    cdef int i, j
    cdef double prefactor = 1.0 / h

    # loop over the matrix
    for i in prange(rows_in):
        #left boundary
        mat_out[i, 0] += prefactor * mat_in[i, 0]

        #main part
        for j in range(1, cols_in - 1):
            mat_out[i, j] += prefactor * (mat_in[i, j] - mat_in[i, j - 1])
        
        #right boundary
        if (domain_right_boundary):
            mat_out[i, cols_in - 1] += (-1) * prefactor * mat_in[i, cols_in - 2]
        else:
            mat_out[i, cols_in - 1] += prefactor * (mat_in[i, cols_in - 1] - mat_in[i, cols_in - 2])
            #extend size
            mat_out[i, cols_in] += (-1) * prefactor * mat_in[i, cols_in - 1]

    return

@cython.boundscheck(False)
cpdef void diffB_y(double [:, :] mat_in, double [:, :] mat_out, 
                    int cols_in, int rows_in,
                    bint domain_bottom_boundary, double h) \
                    nogil:
    """
    Backward differences in y-direction uncutted or cutted.
    Defined in a way that it is adjoint to the forward differences.
    For domain_bottom_boundary=True,
    the output dimension is the same as the input dimension.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0  0)
    (-1  1  ...  0  0  0)
    ( |  |   \   |  |  |)
    ( 0  0  ... -1  1  0)
    ( 0  0  ...  0 -1  0)
    For domain_bottom_boundary=False,
    the output dimension extends by 1.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0 )
    (-1  1  ...  0  0 )
    ( |  |   \   |  | )
    ( 0  0  ... -1  1 )
    (-----------------)
    ( 0  0  ...  0 -1 )
    Args:
        mat_in: input block
        mat_out: output block
        cols_in: number of columns of input matrix
        rows_in: number of rows of input matrix
        domain_right_boundary: see description above
        h: grid size
    """

    # set the variable extension types
    cdef int i, j
    cdef double prefactor = 1.0 / h

    # loop over the matrix
    for i in prange(rows_in):
        #main part
        if (i > 0 and i < rows_in - 1): 
            for j in range(cols_in):
                mat_out[i, j] += prefactor * (mat_in[i, j] - mat_in[i - 1, j])
        
        #upper boundary
        elif (i == 0):
            for j in range(cols_in):
                mat_out[0, j] += prefactor * mat_in[0, j]

        #right boundary
        elif (domain_bottom_boundary):
            for j in range(cols_in):
                mat_out[rows_in - 1, j] += (-1) * prefactor * mat_in[rows_in - 2, j]
        else:
            for j in range(cols_in):
                mat_out[rows_in - 1, j] += prefactor * (mat_in[rows_in - 1, j] - mat_in[rows_in - 2, j])
                #extend size
                mat_out[rows_in, j] += (-1) * prefactor * mat_in[rows_in - 1, j]
    return





@cython.boundscheck(False)
cpdef void gradF(double [:, :] mat_in, double [:, :, :] mat_out,
                int cols_out, int rows_out,
                bint domain_right_boundary, bint domain_bottom_boundary,
                double h) nogil:
    """
    Performs gradient of image as forward differences.
    Uses diffF_x and diffF_y.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Depending on the domain this function call refers to, 
    the forward differences can be uncutted or cutted.
    For domain_right_boundary=True or domain_bottom_boundary=True, the 
    output dimension in x-/y-direction is the same as the input dimension.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0  0)
    ( 0 -1  1  ...  0  0)
    ( |  |      \   |  |)
    ( 0  0  0  ... -1  1)
    ( 0  0  0  ...  0  0)
    For domain_right_boundary=False or domain_bottom_boundary=False,
    the dimension reduces by one.
    It corresponds to the following linear operator:
    (-1  1  0  ...  0 | 0)
    ( 0 -1  1  ...  0 | 0)
    ( |  |  |   \   | | |)
    ( 0  0  0  ...  1 | 0)
    ( 0  0  0  ... -1 | 1)
    Args:
        mat_in: input block
        mat_out: output block
        cols_out: number of columns of output matrix
        rows_out: number of rows of output matrix
        domain_right_boundary: see description above
        domain_bottom_boundary: see description above
        h: grid size
    """

    cdef double [:, :] matx_out = mat_out[0]
    cdef double [:, :] maty_out = mat_out[1]

    diffF_x(mat_in, matx_out, cols_out, rows_out, domain_right_boundary, h)
    diffF_y(mat_in, maty_out, cols_out, rows_out, domain_bottom_boundary, h)

    return



@cython.boundscheck(False)
cpdef void divB(double [:, :, :] mat_in, double [:, :] mat_out,
                int cols_in, int rows_in,
                bint domain_right_boundary, bint domain_bottom_boundary,
                double h) nogil:
    """
    Performs divergence of image pair as backward differences
    as suggested from Chambolle 
    ("An Algorithm for Total Variation Minimization and Applications").
    Combined with a forward diffence gradient this corresponds
    to a discrete Laplacian with Neumann boundary conditions.
    Uses diffB_x and diffB_y.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Depending on the domain this function call refers to, 
    the backward differences can be uncutted or cutted.
    For domain_right_boundary=True or domain_bottom_boundary=True,
    the output dimension is the same as the input dimension.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0  0)
    (-1  1  ...  0  0  0)
    ( |  |   \   |  |  |)
    ( 0  0  ... -1  1  0)
    ( 0  0  ...  0 -1  0)
    For domain_right_boundary=False or domain_bottom_boundary=False,
    the output dimension extends by 1.
    It corresponds to the following linear operator:
    ( 1  0  ...  0  0 )
    (-1  1  ...  0  0 )
    ( |  |   \   |  | )
    ( 0  0  ... -1  1 )
    (-----------------)
    ( 0  0  ...  0 -1 )
    """


    cdef double [:, :] matx_in = mat_in[0]
    cdef double [:, :] maty_in = mat_in[1]

    diffB_x(matx_in, mat_out, cols_in, rows_in, domain_right_boundary, h)
    diffB_y(maty_in, mat_out, cols_in, rows_in, domain_bottom_boundary, h)

    return



"""
------------------------------------------------------------------
PROJECTION ON SPACE DIV . = 0
------------------------------------------------------------------
"""

@cython.boundscheck(False)
cpdef double [:, :, :] proj_div0_sparse_general( \
                double [:, :, :] tau_in, double [:, :, :] tau_out,
                double [:, :] tmp_mat1, double [:, :] tmp_mat2,
                double [:, :] tmp_mat3, double [:, :] tmp_mat4,
                double [:] Sigma1_sq, double [:] Sigma2_sq,
                unsigned int [:] labounds_in, unsigned int [:] rabounds_in,
                unsigned int [:] uabounds_in, unsigned int [:] babounds_in,
                unsigned int [:] asizes1_in, unsigned int [:] asizes2_in, 
                unsigned int [:] labounds_out, unsigned int [:] rabounds_out,
                unsigned int [:] uabounds_out, unsigned int [:] babounds_out,
                unsigned int [:] asizes1_out, unsigned int [:] asizes2_out,
                int size1k1, int size2k2, int size1m1, int size2m2,
                int N1, int N2, int M1, int M2, 
                int k1, int k2, int m1, int m2, double h):

    """
    Computes a block of the projection on the subspace div^-(.) = 0,
    using the Moore-Penrose-Inverse of the Laplacian with Neumann boundaries.
    It is strongly recommended to use the methods proj_sparse(...) or
    proj_sparse_general(...) of the ProjectorDiv0-class in proj_div_0_low_cap.py
    to call this function. 
    Args:
        tau_in      2-channel-matrix at block (k2, k1), only non-zero-part
                    shape: (2, sizes2[k2], sizes1[k1])
        tau_out     projected 2-channel-matrix at block (m2, m1)
                    shape: (2, sizes2[m2], sizes1[m1])
        Sigma_sq    Prepared matrix Sigma^2 to evaluate
                    inverse laplacian in the image of the DCT.
                    Can be prepared by calling prepare_Sigma_sq(...).
        tmp_mat     Allocated space. Must (each) at least have the 
                    rows and columns of the biggest blocks.
        N2, N1      number of rows and columns of big matrix
        M2, M1      number of rows and columns of blocks
        (k2, k1)    indices of input block (row, column)
        (m2, m1)    indices of output block (row, column)
        abounds     adapted left, right, upper and bottom bounds of blocks
                    adapted means, that block (k2, k1) has been increased
                    to the size of its halo (in case of abounds_in)
                    or block (m2, m1) has been increased to the size
                    of its halo (in case of abounds_out)
        asizes      adapted sizes in x- (sizes1) and y- (sizes2) direction
        size1k1,... actual sizes of (k2,k1)- and (m2,m1)-blocks (without halo)
    """

    cdef int i, j, i_in, j_in               #loop variables
    cdef bint drb_k, dbb_k, drb_m, dbb_m    #blocks at the edge?
    
    cdef int i_diff_out_in = uabounds_out[m2] - uabounds_in[k2] 
    cdef int j_diff_out_in = labounds_out[m1] - labounds_in[k1]

    #initialize drb_k, dbb_k, drb_m, dbb_m
    drb_k = False
    dbb_k = False
    drb_m = False
    dbb_m = False

    #is domain (k2, k1) at the right boundary? at the bottom boundary?
    if (k1 == M1 - 1):
        drb_k = True
    if (k2 == M2 - 1):
        dbb_k = True
    #what about domain (m2, m1)?
    if (m1 == M1 - 1):
        drb_m = True
    if (m2 == M2 - 1):
        dbb_m = True

    #initialize tau_out
    for i in prange(size2m2, nogil=True):
        for j in range(size1m1):
            tau_out[0,i,j] = 0.0
            tau_out[1,i,j] = 0.0

    #initialize tmp_mat3 (everything else gets initialized somewhere else)
    #make space for divB(tau_in)
    cdef int sx_tmp_mat3 = size1k1
    cdef int sy_tmp_mat3 = size2k2
    if (drb_k == False):
        sx_tmp_mat3 += 1
    if (dbb_k == False):
        sy_tmp_mat3 += 1
    for i in prange(sy_tmp_mat3, nogil=True):
        for j in range(sx_tmp_mat3):
            tmp_mat3[i,j] = 0.0

    #calculate divergence of tau_in and save it in tmp_mat3
    divB(tau_in, tmp_mat3, size1k1, size2k2, drb_k, dbb_k, h)

    #evaluate inverse laplacian of divergence of tau_in and save it in tmp_mat4
    invlapl_div_tau = inv_laplacian_neum_sparse_general( \
            tmp_mat3, tmp_mat4, tmp_mat1, tmp_mat2, Sigma1_sq, Sigma2_sq, \
            labounds_in, rabounds_in, uabounds_in, babounds_in, asizes1_in, asizes2_in, \
            labounds_out, rabounds_out, uabounds_out, babounds_out, asizes1_out, asizes2_out, \
            N1, N2, M1, M2, k1, k2, m1, m2)

    #determine grad inv_lapl div tau and save it in tau_out
    gradF(tmp_mat4, tau_out, size1m1, size2m2, drb_m, dbb_m, h)

    #Now evaluate the projection Pi tau = tau - grad inv_lapl div tau
    #  parallely on tau_outl. Since it is assumed that tau_in == 0.0
    #  outside of block (k2, k1), only the intersecting pixels are relevant.
    for i in prange(size2m2, nogil = True):
        for j in range(size1m1):
            tau_out[0,i,j] = - tau_out[0,i,j]
            tau_out[1,i,j] = - tau_out[1,i,j]

            #Is (i,j) in (m2,m1)-coordinates within block (k2,k1)?
            i_in = i + i_diff_out_in
            j_in = j + j_diff_out_in
            if ((i_in >= 0 and i_in < size2k2) and \
                    (j_in >= 0 and j_in < size1k1)):
                
                tau_out[0,i,j] += tau_in[0,i_in,j_in]
                tau_out[1,i,j] += tau_in[1,i_in,j_in]

    return tau_out
