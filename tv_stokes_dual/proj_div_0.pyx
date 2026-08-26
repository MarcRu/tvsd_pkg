"""
===================================================================
This Cython-file provides necessary function for the simple dual
TV-Stokes-Implementation tvsd.py (without Domain Decomposition).
===================================================================
"""


import cython
from cython.parallel import prange
from libc.math cimport sqrt, sin, M_PI_2
from scipy.fft import dct, idct, dst, idst
import numpy as np




"""
DISCRETE COSINE TRANSFORMS
NOTICE! THESE FUNCTIONS USE THE SCIPY IMPLEMENTATION OF DCT!
   THIS CAN MAYBE BE OPTIMIZED BY IMPLEMENTING FAST DCT
   PARALLELY IN CYTHON WITH PREDETERMINED COSINE VALUES
"""


def dct_x(mat_in, mat_out):
    """
    Discrete Cosine Transform in x-direction
    """

    mat_out = dct(mat_in, type=2, axis=1, norm='ortho')
    
    return mat_out

def idct_x(mat_in, mat_out):
    """
    Inverse Discrete Cosine Transform in x-direction
    """

    mat_out = idct(mat_in, type=2, axis=1, norm='ortho')
    
    return mat_out


def dct_y(mat_in, mat_out):
    """
    Discrete Cosine Transform in y-direction
    """

    mat_out = dct(mat_in, type=2, axis=0, norm='ortho')
    
    return mat_out

def idct_y(mat_in, mat_out):
    """
    Inverse Discrete Cosine Transform in y-direction
    """

    mat_out = idct(mat_in, type=2, axis=0, norm='ortho')
    
    return mat_out



"""
MOORE-PENROSE-INVERSES OF LAPLACIANS.
"""


@cython.boundscheck(False)
cpdef void prepare_Sigma_sq_dct(double [:] mat_out, double h):
    """
    Returns the diagonal entries of Sigma^2-matrix for computing the inverse
    laplacian with Neumann boundary conditions in the image space of the 
    Discrete Cosine Transform.
    The grid size is denoted by h.
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



@cython.boundscheck(False)
cpdef void prepare_Sigma_sq_dst(double [:] mat_out, double h):
    """
    Returns the diagonal entries of Sigma^2-matrix for computing the inverse
    laplacian with Dirichlet boundary conditions in the image space of the 
    Discrete Sine Transform.
    The grid size is denoted by h.
    Similar as Sigma^2 for DCT, but uses Sigma_{N+1} and skips 0 entry.
    """
    cdef int i, N
    cdef double prefactor, sinfactor

    N = mat_out.shape[0]
    prefactor = 4.0 / (h * h)
    sinfactor = M_PI_2 / <double>(N + 1)

    mat_out[0] = 0.0
    for i in prange(0, N, nogil=True):
        mat_out[i] = prefactor * sin(sinfactor * <double>(i + 1)) ** 2

    return



@cython.boundscheck(False)
cpdef void inv_laplacian_dct_img(double [:, :] mat_in, double [:, :] mat_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq):
    """
    Inverse operator of Laplacian with Neumann boundary conditions
    in the image of the discrete cosine transform.
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2. They can be determined with prepare_Sigma_sq_dct(...).
    """


    # set the variable extension types
    cdef int j, i, N1, N2

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]

    mat_out[0,0] = 0.0
    for j in range(1, N1):
        mat_out[0,j] = (-1.0) * mat_in[0,j] / Sigma1_sq[j]
    for i in prange(1, N2, nogil=True):
        mat_out[i,0] = (-1.0) * mat_in[i,0] / Sigma2_sq[i]
        for j in range(1, N1):
            mat_out[i,j] = (-1.0) * mat_in[i,j] / \
                    (Sigma1_sq[j] + Sigma2_sq[i])

    return




@cython.boundscheck(False)
cpdef void inv_laplacian_dst_img(double [:, :] mat_in, double [:, :] mat_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq):
    """
    Inverse operator of Laplacian with Dirichlet boundary conditions
    in the image of the discrete sine transform.
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2. They can be determined with prepare_Sigma_sq_dst(...).
    """
    
    # set the variable extension types
    cdef int j, i, N1, N2

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]

    for i in prange(0, N2, nogil=True):
        for j in range(0, N1):
            mat_out[i,j] = -mat_in[i,j] / \
                    (Sigma1_sq[j] + Sigma2_sq[i])
    
    return





@cython.boundscheck(False)
cpdef double [:, :] inv_laplacian_neum(double [:, :] mat_in, double [:, :] mat_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq, 
                double [:, :] tmp_mat1, double [:, :] tmp_mat2):
    """
    Moore-Penrose-Inverse of Laplacian with Neumann boundary conditions,
    calculated with help of Discrete Cosine Transform.
    Needs allocated space tmp_mat1 and tmp_mat2 in the size of mat_in and mat_out.
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2. They can be determined with prepare_Sigma_sq_dct(...).
    """
    
    cdef int N1, N2

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]

    #apply dct in both dimensions
    tmp_mat1 = dct_x(mat_in, tmp_mat1)
    tmp_mat2 = dct_y(tmp_mat1, tmp_mat2)

    #perform inverse laplacian in the dct image
    inv_laplacian_dct_img(tmp_mat2, tmp_mat1, Sigma1_sq, Sigma2_sq)

    #apply inverse dct in both dimensions
    tmp_mat2 = idct_y(tmp_mat1, tmp_mat2)
    mat_out = idct_x(tmp_mat2, mat_out)

    return mat_out




@cython.boundscheck(False)
cpdef double [:, :] inv_laplacian_diri(double [:, :] mat_in, double [:, :] mat_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq, 
                double [:, :] tmp_mat1, double [:, :] tmp_mat2):
    """
    Moore-Penrose-Inverse of Laplacian with Dirichlet boundary conditions,
    calculated with help of Discrete Sine Transform.
    Needs allocated space tmp_mat1 and tmp_mat2 in the size of mat_in and mat_out.
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2. They can be determined with prepare_Sigma_sq_dst(...).
    """
    
    cdef int N1, N2

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]

    tmp_mat1 = dst(mat_in, type=1, axis=0)
    tmp_mat2 = dst(tmp_mat1, type=1, axis=1)

    #perform inverse laplacian in the dct image
    inv_laplacian_dst_img(tmp_mat2, tmp_mat1, Sigma1_sq, Sigma2_sq)

    
    tmp_mat2 = idst(tmp_mat1, type=1, axis=0)
    mat_out = idst(tmp_mat2, type=1, axis=1)

    return mat_out



"""
DIFFERENTIAL OPERATORS
All forward differences will be performed in a way that the last entry
will be 0:
(-1  1  0  ...  0  0)
( 0 -1  1  ...  0  0)
( |  |      \   |  |)
( 0  0  0  ... -1  1)
( 0  0  0  ...  0  0)
For domain_right_boundary=False or domain_bottom_boundary=False,
the dimension reduces by one.
All backward differences will be performed in a way that the last entry
will be ignored:
( 1  0  ...  0 | 0)
(-1  1  ...  0 | 0)
( 0 -1  ...  0 | 0)
( |  |   \   | | |)
( 0  0  ...  1 | 0)
( 0  0  ... -1 | 0)
The purpose of this is to make sure that gradF is conjugate to -divB
and gradB is conjugate to -divF.
Note that divB(gradF(...)) is the laplacian with Neumann boundary conditions
and divF(gradB(...)) is the laplacian with Dirichlet boundary conditions.
"""



@cython.boundscheck(False)
cpdef void gradF(double [:, :] mat_in, 
                double [:, :] gradx_out, double [:, :] grady_out,
                double h):
    """
    Performs gradient of image as forward differences.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Can be used in the context of Laplace-operators with Neumann boundaries.
    """

    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]
    prefactor = 1.0 / h

    for i in prange(0, N2-1, nogil=True):
        #center
        for j in range(0, N1-1):
            gradx_out[i,j] = prefactor * (mat_in[i,j+1] - mat_in[i,j])
            grady_out[i,j] = prefactor * (mat_in[i+1,j] - mat_in[i,j])
        #right border
        gradx_out[i,N1-1] = 0.0
        grady_out[i,N1-1] = prefactor * (mat_in[i+1,N1-1] - mat_in[i,N1-1])
    for j in range(0, N1-1):
        #lower border
        gradx_out[N2-1,j] = prefactor * (mat_in[N2-1,j+1] - mat_in[N2-1,j])
        grady_out[N2-1,j] = 0.0

    #bottom right corner
    gradx_out[N2-1,N1-1] = 0.0
    grady_out[N2-1,N1-1] = 0.0

    return




@cython.boundscheck(False)
cpdef void gradB(double [:, :] mat_in, 
                double [:, :] gradx_out, double [:, :] grady_out,
                double h):
    """
    Performs gradient of image as backward differences.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Can be used in the context of Laplace-operators with Dirichlet boundaries.
    """

    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = gradx_out.shape[0]
    N1 = gradx_out.shape[1]
    prefactor = 1.0 / h

    #gradx
    for i in prange(0, N2 - 1, nogil=True):
        #center
        for j in range(1, N1 - 1):
            gradx_out[i,j] = prefactor * ( \
                        mat_in[i,j] - mat_in[i,j-1])
        #left border
        gradx_out[i,0] = prefactor * mat_in[i,0]
        #right border
        gradx_out[i,N1-1] = prefactor * (-1.0) * mat_in[i,N1-2]
    for j in range(0, N1 - 1):
        #lower border
        gradx_out[N2-1,j] = 0.0

    #grady
    for i in prange(1, N2 - 1, nogil=True):
        #center
        for j in range(0, N1 - 1):
            grady_out[i,j] = prefactor * ( \
                        mat_in[i,j] - mat_in[i-1,j])
        #right border
        grady_out[i,N1-1] = 0.0
    for j in range(0, N1 - 1):
        #upper border
        grady_out[0,j] = prefactor * mat_in[0,j]
        #lower border
        grady_out[N2-1,j] = prefactor * (-1.0) * mat_in[N2-2,j]
    #top right corner
    grady_out[0,N1-1] = 0.0
    #bottom right corner
    grady_out[N2-1,N1-1] = 0.0

    return



@cython.boundscheck(False)
cpdef void divB(double [:, :] matx_in, double [:, :] maty_in,
                double [:, :] mat_out, double h):
    """
    Performs divergence of image pair as backward differences.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Can be used in the context of Laplace-operators with Neumann boundaries.
    """

    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = matx_in.shape[0]
    N1 = matx_in.shape[1]
    prefactor = 1.0 / h

    for i in prange(1, N2 - 1, nogil=True):
        #center
        for j in range(1, N1 - 1):
            mat_out[i,j] = prefactor * ( \
                        matx_in[i,j] - matx_in[i,j-1]
                        + maty_in[i,j] - maty_in[i-1,j])
        #left border
        mat_out[i,0] = prefactor * ( \
                        matx_in[i,0]
                        + maty_in[i,0] - maty_in[i-1,0])
        #right border
        mat_out[i,N1-1] = prefactor * ( \
                        - matx_in[i,N1-2]
                        + maty_in[i,N1-1] - maty_in[i-1,N1-1])
    for j in range(1, N1 - 1):
        #upper border
        mat_out[0,j] = prefactor * ( \
                        matx_in[0,j] - matx_in[0,j-1] \
                        + maty_in[0,j])
        #lower border
        mat_out[N2-1,j] = prefactor * ( \
                        matx_in[N2-1,j] - matx_in[N2-1,j-1] \
                        - maty_in[N2-2,j])
    #top left corner
    mat_out[0,0] = prefactor * (
            matx_in[0,0] + maty_in[0,0])
    #top right corner
    mat_out[0,N1-1] = prefactor * (
            - matx_in[0,N1-2] + maty_in[0,N1-1])
    #bottom left corner
    mat_out[N2-1,0] = prefactor * (
            matx_in[N2-1,0] - maty_in[N2-2,0])
    #bottom right corner
    mat_out[N2-1,N1-1] = prefactor * (
            - matx_in[N2-1,N1-2] - maty_in[N2-2,N1-1])

    return



@cython.boundscheck(False)
cpdef void divF(double [:, :] matx_in, double [:, :] maty_in,
                double [:, :] mat_out, double h):
    """
    Performs divergence of image pair as forward differences.
    Size of mat_in and mat_out must be equal and at least 2x2.
    Can be used in the context of Laplace-operators with Dirichlet boundaries.
    """

    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = matx_in.shape[0]
    N1 = matx_in.shape[1]
    prefactor = 1.0 / h

    for i in prange(0, N2-1, nogil=True):
        #center
        for j in range(0, N1-1):
            mat_out[i,j] = prefactor * ( \
                        matx_in[i,j+1] - matx_in[i,j]
                        + maty_in[i+1,j] - maty_in[i,j])
        #right border
        mat_out[i,N1-1] = 0.0
    for j in range(0, N1-1):
        #lower border
        mat_out[N2-1,j] = 0.0

    #bottom right corner
    mat_out[N2-1,N1-1] = 0.0
    
    return




@cython.boundscheck(False)
cpdef void laplacian_neum(double [:, :] mat_in, double [:, :] mat_out, double h):
    """
    Performs laplacian with Neumann boundaries (no flow in and out):
    lapl_neum = divB gradF
    Size of mat_in and mat_out must be equal and at least 2x2
    """

    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]
    prefactor = 1.0 / (h * h)

    for i in prange(1, N2 - 1, nogil=True):
        #center
        for j in range(1, N1 - 1):
            mat_out[i,j] = prefactor * (
                        mat_in[i+1,j] + mat_in[i-1,j] \
                        + mat_in[i,j+1] + mat_in[i,j-1] \
                        - 4 * mat_in[i,j])
        #left border
        mat_out[i,0] = prefactor * (
                        mat_in[i+1,0] + mat_in[i-1,0] \
                        + mat_in[i,1] \
                        - 3 * mat_in[i,0])
        #right border
        mat_out[i,N1-1] = prefactor * (
                        mat_in[i+1,N1-1] + mat_in[i-1,N1-1] \
                        + mat_in[i,N1-2] \
                        - 3 * mat_in[i,N1-1])
    for j in range(1, N1 - 1):
        #upper border
        mat_out[0,j] = prefactor * (
                        mat_in[0,j+1] + mat_in[0,j-1] \
                        + mat_in[1,j] \
                        - 3 * mat_in[0,j])
        #lower border
        mat_out[N2-1,j] = prefactor * (
                        mat_in[N2-1,j+1] + mat_in[N2-1,j-1] \
                        + mat_in[N2-2,j] \
                        - 3 * mat_in[N2-1,j])
    #top left corner
    mat_out[0,0] = prefactor * (
            mat_in[0,1] + mat_in[1,0] - 2 * mat_in[0,0])
    #top right corner
    mat_out[0,N1-1] = prefactor * (
            mat_in[0,N1-2] + mat_in[1,N1-1] - 2 * mat_in[0,N1-1])
    #bottom left corner
    mat_out[N2-1,0] = prefactor * (
            mat_in[N2-1,1] + mat_in[N2-2,0] - 2 * mat_in[N2-1,0])
    #bottom right corner
    mat_out[N2-1,N1-1] = prefactor * (
            mat_in[N2-1,N1-2] + mat_in[N2-2,N1-1] - 2 * mat_in[N2-1,N1-1])

    return



@cython.boundscheck(False)
cpdef void laplacian_diri(double [:, :] mat_in, double [:, :] mat_out, double h):
    """
    Performs laplacian with Dirichlet boundaries (boundary 0):
    lapl_diri = divF gradB
    Size of mat_in and mat_out must be equal and at least 2x2
    """
    
    # set the variable extension types
    cdef int j, i, N1, N2
    cdef double prefactor

    N2 = mat_in.shape[0]
    N1 = mat_in.shape[1]
    prefactor = 1.0 / (h * h)

    for i in prange(1, N2 - 1, nogil=True):
        #center
        for j in range(1, N1 - 1):
            mat_out[i,j] = prefactor * (
                        mat_in[i+1,j] + mat_in[i-1,j] \
                        + mat_in[i,j+1] + mat_in[i,j-1] \
                        - 4 * mat_in[i,j])
        #left border
        mat_out[i,0] = prefactor * (
                        mat_in[i+1,0] + mat_in[i-1,0] \
                        + mat_in[i,1] \
                        - 4 * mat_in[i,0])
        #right border
        mat_out[i,N1-1] = prefactor * (
                        mat_in[i+1,N1-1] + mat_in[i-1,N1-1] \
                        + mat_in[i,N1-2] \
                        - 4 * mat_in[i,N1-1])
    for j in range(1, N1 - 1):
        #upper border
        mat_out[0,j] = prefactor * (
                        mat_in[0,j+1] + mat_in[0,j-1] \
                        + mat_in[1,j] \
                        - 4 * mat_in[0,j])
        #lower border
        mat_out[N2-1,j] = prefactor * (
                        mat_in[N2-1,j+1] + mat_in[N2-1,j-1] \
                        + mat_in[N2-2,j] \
                        - 4 * mat_in[N2-1,j])
    #top left corner
    mat_out[0,0] = prefactor * (
            mat_in[0,1] + mat_in[1,0] - 4 * mat_in[0,0])
    #top right corner
    mat_out[0,N1-1] = prefactor * (
            mat_in[0,N1-2] + mat_in[1,N1-1] - 4 * mat_in[0,N1-1])
    #bottom left corner
    mat_out[N2-1,0] = prefactor * (
            mat_in[N2-1,1] + mat_in[N2-2,0] - 4 * mat_in[N2-1,0])
    #bottom right corner
    mat_out[N2-1,N1-1] = prefactor * (
            mat_in[N2-1,N1-2] + mat_in[N2-2,N1-1] - 4 * mat_in[N2-1,N1-1])

    return


"""
ORTHOGONAL PROJECTIONS
on discrete subspace K with
div(...) = 0
"""


@cython.boundscheck(False)
cpdef double [:, :, :] proj_neumann_dct(
                double [:, :, :] tau_in,
                double [:, :, :] tau_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq, 
                double [:, :] tmp_mat1, double [:, :] tmp_mat2,
                double h):
    """
    Performs projection on subspace K with divB(...) = 0 by
    computing the expression P_K = I - gradF inv_lapl_neum divB
    with inverse of laplacian with Neumann boundary conditions.
    Needs allocated space tmp_mat1 and tmp_mat2 in the size of 
    tau1_in, tau2_in, tau1_out and tau2_out
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2 .
    They can be determined with prepare_Sigma_sq_dct(...).
    """

    cdef int j, i, N1, N2
    cdef double [:, :] tau1_in = tau_in[0]
    cdef double [:, :] tau2_in = tau_in[1]
    cdef double [:, :] tau1_out = tau_out[0]
    cdef double [:, :] tau2_out = tau_out[1]
    N2 = tau1_in.shape[0]
    N1 = tau2_in.shape[1]

    ##write divergence into tmp_mat1
    divB(tau1_in, tau2_in, tmp_mat1, h)

    #write inverse laplacian into tmp_mat2:
    #  offer disk space of tmp_mat1 and tmp_mat2 in a way that
    #  they are never used at the same time
    tmp_mat2 = inv_laplacian_neum(tmp_mat1, tmp_mat2,
                Sigma1_sq, Sigma2_sq, 
                tmp_mat2, tmp_mat1)

    #determine gradient of inverse laplacian and write it into
    #  tau1_out and tau2_out
    gradF(tmp_mat2, tau1_out, tau2_out, h)

    #now evaluate the projection Pi tau = tau - grad inv_lapl div tau
    #  parallelly on tau1_out and tau2_out
    for i in prange(N2, nogil=True):
        for j in range(N1):
            tau1_out[i,j] = tau1_in[i,j] - tau1_out[i,j]
            tau2_out[i,j] = tau2_in[i,j] - tau2_out[i,j]

    return tau_out






@cython.boundscheck(False)
cpdef double [:, :, :] proj_dirichlet_dst(
                double [:, :, :] tau_in,
                double [:, :, :] tau_out,
                double [:] Sigma1_sq, double [:] Sigma2_sq, 
                double [:, :] tmp_mat1, double [:, :] tmp_mat2,
                double h):
    """
    Performs projection on subspace K with divF(...) = 0 by
    computing the expression P_K = I - gradB inv_lapl_diri divF
    with inverse of laplacian with Dirichlet boundary conditions.
    Needs allocated space tmp_mat1 and tmp_mat2 in the size of 
    tau1_in, tau2_in, tau1_out and tau2_out.
    Needs predetermined diagonal entries of the matrices
    Sigma1^2 and Sigma2^2. 
    They can be determined with prepare_Sigma_sq_dst(...).
    """
    cdef int j, i, N1p, N2p, N1, N2
    cdef double [:, :] tau1_in = tau_in[0]
    cdef double [:, :] tau2_in = tau_in[1]
    cdef double [:, :] tau1_out = tau_out[0]
    cdef double [:, :] tau2_out = tau_out[1]
    N2p = tau1_in.shape[0]
    N1p = tau1_in.shape[1]
    N2 = N2p - 1
    N1 = N1p - 1

    #write divergence into tmp_mat1
    divF(tau1_in, tau2_in, tmp_mat1, h)

    #write inverse laplacian into tmp_mat2:
    #  offer disk space of tmp_mat1 and tmp_mat2 in a way that
    #  they are never used at the same time
    tmp_mat2[0:N2, 0:N1] = inv_laplacian_diri(tmp_mat1[0:N2, 0:N1], 
                tmp_mat2[0:N2, 0:N1], Sigma1_sq, Sigma2_sq, 
                tmp_mat2[0:N2, 0:N1], tmp_mat1[0:N2, 0:N1])

    #determine gradient of inverse laplacian and write it into
    #  tau1_out and tau2_out
    gradB(tmp_mat2, tau1_out, tau2_out, h)

    #now evaluate the projection Pi tau = tau - grad inv_lapl div tau
    #  parallelly on tau1_out and tau2_out
    for i in prange(N2p, nogil=True):
        for j in range(N1p):
            tau1_out[i,j] = tau1_in[i,j] - tau1_out[i,j]
            tau2_out[i,j] = tau2_in[i,j] - tau2_out[i,j]

    return tau_out



"""
INTEGRATION OF PROJECTED TANGENT FIELD
"""


@cython.boundscheck(False)
cpdef double [:, :] integrate_tangentfield(
                double [:, :, :] tau, double [:, :] g_out,
                double h, int int_mode):
    """

    The number of pixels N2xN1 get determined from the allocated space from
    output g_out.
    int_mode == -1:        Backward differences (assumes divB(tau)=0)
    int_mode == +1:        Forward differences (assumes divF(tau)=0)
    """

    cdef int j, i, N1, N2
    cdef double sum, mean
    
    N2 = g_out.shape[0]
    N1 = g_out.shape[1]
    assert tau.shape[1] >= N2 and tau.shape[2] >= N1



    if (int_mode == -1):
        #tau has N2 rows and N1 columns;
        #only the rows 1,...,N2-1 and columns 1,...,N1-1 are interesting

        #Assume upper left corner ... value does not matter
        g_out[0,0] = 0.0

        #Integrate down the left column
        for i in range(1, N2):
            g_out[i,0] = g_out[i-1,0] - h * tau[0,i,0]

        #Integrate from left to right
        for i in prange(0, N2, nogil=True):
            #Integrate from right to left
            for j in range(1, N1):
                g_out[i,j] = g_out[i,j-1] + h * tau[1,i,j]


    elif (int_mode == 1):
        #Assume upper left corner ... value does not matter
        g_out[0,0] = 0.0

        #Integrate down the left column
        for i in range(1, N2):
            g_out[i,0] = g_out[i-1,0] - h * tau[0,i-1,0]

        #Now parallely integrate every line
        for i in prange(N2, nogil=True):
            #Integrate from left to right
            for j in range(1, N1):
                g_out[i,j] = g_out[i,j-1] + h * tau[1,i,j-1]


    else:
        raise Exception("invalid int_mode")
    


    return g_out
