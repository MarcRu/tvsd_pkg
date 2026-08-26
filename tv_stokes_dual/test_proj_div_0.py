import math
import numpy as np
import time

import proj_div_0 as pd0

from scipy.fft import dct, idct

def dct_matrix(N):
    C = np.zeros([N,N], dtype=np.float64)
    C[0,:] = math.sqrt(1.0 / float(N)) * np.ones((1,N), dtype=float)
    for i in range(1, N):
        for j in range(N):
            C[i,j] = math.sqrt(2.0 / float(N)) * math.cos(math.pi / float(N) * float(i) * (float(j) + 0.5) )
    return C


def dct_tests():
    print("Dct x Test 1")
    D = 2 * np.eye(5, dtype=np.float64)
    D[0,1] = 1
    CD = np.zeros((5, 5), dtype=np.float64)
    CDgt = np.matmul(D, np.transpose(dct_matrix(5)))
    t1 = time.time()
    CD = pd0.dct_x(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Dct x Test 1 passed with duration: ", (t2 - t1))

    print("Dct x Test 2")
    D = np.eye(1000, dtype=np.float64)
    CD = np.zeros((1000, 1000), dtype=np.float64)
    CDgt = np.matmul(D, np.transpose(dct_matrix(1000)))
    t1 = time.time()
    CD = pd0.dct_x(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Dct x Test 2 passed with duration: ", (t2 - t1))

    print("Dct x Test 3")
    D = np.zeros((5,7))
    D[0,1] = 1
    D[3,2] = 3
    CD = np.zeros((7, 7), dtype=np.float64)
    CDgt = np.matmul(D, np.transpose(dct_matrix(7)))
    t1 = time.time()
    CD = pd0.dct_x(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Dct x Test 1 passed with duration: ", (t2 - t1))

    print("Idct x Test 1")
    D = 2 * np.eye(5, dtype=np.float64)
    D[0,1] = 1
    CD = np.zeros((5, 5), dtype=np.float64)
    CDgt = np.matmul(D, dct_matrix(5))
    t1 = time.time()
    CD = pd0.idct_x(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Idct x Test 1 passed with duration: ", (t2 - t1))

    print("Idct x Test 2")
    D = np.eye(1000, dtype=np.float64)
    CD = np.zeros((1000, 1000), dtype=np.float64)
    CDgt = np.matmul(D, dct_matrix(1000))
    t1 = time.time()
    CD = pd0.idct_x(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Idct x Test 2 passed with duration: ", (t2 - t1))

    print("Dct y Test")
    D = 2 * np.eye(5, dtype=np.float64)
    D[0,1] = 1
    CD = np.zeros((5, 5), dtype=np.float64)
    CDgt = np.matmul(dct_matrix(5), D)
    t1 = time.time()
    CD = pd0.dct_y(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Dct y Test passed with duration: ", (t2 - t1))

    print("Idct y Test")
    D = 2 * np.eye(5, dtype=np.float64)
    D[0,1] = 1
    CD = np.zeros((5, 5), dtype=np.float64)
    CDgt = np.matmul(np.transpose(dct_matrix(5)), D)
    t1 = time.time()
    CD = pd0.idct_y(D, CD)
    t2 = time.time()
    np.testing.assert_array_almost_equal(CD, CDgt, 8)
    print("Idct y Test passed with duration: ", (t2 - t1))


def inv_laplace_neum_test():

    N2 = 1000
    N1 = 1002
    h = 1.0

    print("Pseudo Inverse Laplace Neumann Test")
    M = np.random.random((N2,N1))
    lapl_M = np.zeros((N2,N1), dtype = np.float64)
    invlapl_lapl_M = np.zeros((N2,N1), dtype = np.float64)
    lapl_invlapl_lapl_M = np.zeros((N2,N1), dtype = np.float64)
    invlapl_M = np.zeros((N2,N1), dtype = np.float64)
    lapl_invlapl_M = np.zeros((N2,N1), dtype = np.float64)
    inv_lapl_lapl_invlapl_M = np.zeros((N2,N1), dtype = np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
    tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)

    t1 = time.time()
    pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
    t2 = time.time()
    pd0.laplacian_neum(M, lapl_M, h)
    t3 = time.time()
    invlapl_M = pd0.inv_laplacian_neum(M, invlapl_M, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    t4 = time.time()
    invlapl_lapl_M = pd0.inv_laplacian_neum(lapl_M, invlapl_lapl_M, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    pd0.laplacian_neum(invlapl_lapl_M, lapl_invlapl_lapl_M, h)
    np.testing.assert_array_almost_equal(lapl_M, lapl_invlapl_lapl_M, 7)
    print("Pseudo Inverse Condition 1 passed.")
    pd0.laplacian_neum(invlapl_M, lapl_invlapl_M, h)
    inv_lapl_lapl_invlapl_M = pd0.inv_laplacian_neum(lapl_invlapl_M, inv_lapl_lapl_invlapl_M, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    np.testing.assert_array_almost_equal(invlapl_M, inv_lapl_lapl_invlapl_M, 7)
    print("Pseudo Inverse Condition 2 passed.")
    
    #Test if laplace laplace+ and laplace+ laplace are diagonal and therefore hermitian
    N1 = 6
    N2 = 5
    for i1 in range(N2):
        for j1 in range(N1):
            for i2 in range(N2):
                for j2 in range(N1):
                    M1 = np.zeros((N2, N1), dtype=np.float64)
                    M2 = np.zeros((N2, N1), dtype=np.float64)
                    lapl_M1 = np.zeros((N2, N1), dtype=np.float64)
                    lapl_M2 = np.zeros((N2, N1), dtype=np.float64)
                    invlapl_lapl_M1 = np.zeros((N2, N1), dtype=np.float64)
                    invlapl_lapl_M2 = np.zeros((N2, N1), dtype=np.float64)
                    invlapl_M1 = np.zeros((N2, N1), dtype=np.float64)
                    invlapl_M2 = np.zeros((N2, N1), dtype=np.float64)
                    lapl_invlapl_M1 = np.zeros((N2, N1), dtype=np.float64)
                    lapl_invlapl_M2 = np.zeros((N2, N1), dtype=np.float64)
                    M1[i1, j1] = 1.0
                    M2[i2, j2] = 1.0
                    pd0.laplacian_neum(M1, lapl_M1, h)
                    pd0.laplacian_neum(M2, lapl_M2, h)
                    invlapl_lapl_M1 = pd0.inv_laplacian_neum(lapl_M1, invlapl_lapl_M1, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
                    invlapl_lapl_M2 = pd0.inv_laplacian_neum(lapl_M2, invlapl_lapl_M2, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
                    np.testing.assert_almost_equal(\
                            invlapl_lapl_M1[i2, j2], \
                            invlapl_lapl_M2[i1, j1], 6)
                    #print("Pseudo Inverse Condition 3 passed.")
                    invlapl_M1 = pd0.inv_laplacian_neum(M1, invlapl_M1, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
                    invlapl_M2 = pd0.inv_laplacian_neum(M2, invlapl_M2, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
                    pd0.laplacian_neum(invlapl_M1, lapl_invlapl_M1, h)
                    pd0.laplacian_neum(invlapl_M2, lapl_invlapl_M2, h)
                    np.testing.assert_almost_equal( \
                            lapl_invlapl_M1[i2, j2], \
                            lapl_invlapl_M2[i1, j1], 6)
                    #print("Pseudo Inverse Condition 4 passed.")
    print("Pseudo Inverse Condition 3 and 4 passed.")
    print("Duration of one Laplace Neumann operation: ", (t3 - t2))
    print("Duration of one Inverse Laplace Neumann operation: ", (t4 - t3))
    print("Duration of preparing Sigma1_sq and Sigma2_sq:", (t2 - t1))
    print("Pseudo Inverse Laplace Neumann Test passed.")



def inv_laplace_diri_test():
    N2 = 1000
    N1 = 1002
    h = 1.0

    print("Inverse Laplace Dirichlet Test")
    M = np.random.random((N2,N1))
    lapl_M = np.zeros((N2,N1), dtype = np.float64)
    invlapl_lapl_M = np.zeros((N2,N1), dtype = np.float64)
    invlapl_M = np.zeros((N2,N1), dtype = np.float64)
    lapl_invlapl_M = np.zeros((N2,N1), dtype = np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
    tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)

    t1 = time.time()
    pd0.prepare_Sigma_sq_dst(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dst(Sigma2_sq, h)
    t2 = time.time()
    pd0.laplacian_diri(M, lapl_M, h)
    t3 = time.time()
    invlapl_M = pd0.inv_laplacian_diri(M, invlapl_M, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    t4 = time.time()
    invlapl_lapl_M = pd0.inv_laplacian_diri(lapl_M, invlapl_lapl_M, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    np.testing.assert_array_almost_equal(M, invlapl_lapl_M, 7)
    print("Inverse Dirichlet Condition 1 passed.")
    pd0.laplacian_diri(invlapl_M, lapl_invlapl_M, h)
    np.testing.assert_array_almost_equal(M, lapl_invlapl_M, 7)
    print("Inverse Dirichlet Condition 2 passed.")

    print("Duration of one Laplace Dirichlet operation: ", (t3 - t2))
    print("Duration of one Inverse Laplace Dirichlet operation: ", (t4 - t3))
    print("Duration of preparing Sigma1_sq and Sigma2_sq:", (t2 - t1))
    print("Inverse Laplace Dirichlet Test passed.")


def diff_operator_test():
    #test div, grad and laplacian by testing laplace = div*grad

    print("Differential Operator Test 1")

    N2 = 1000
    N1 = 1002
    h = 1.0

    M = np.random.random((N2,N1))
    lapl_M = np.zeros((N2,N1), dtype = np.float64)
    gradx_M = np.zeros((N2,N1), dtype = np.float64)
    grady_M = np.zeros((N2,N1), dtype = np.float64)
    divgrad_M = np.zeros((N2,N1), dtype = np.float64)

    t1 = time.time()
    pd0.gradF(M, gradx_M, grady_M, h)
    t2 = time.time()
    pd0.divB(gradx_M, grady_M, divgrad_M, h)
    t3 = time.time()
    pd0.laplacian_neum(M, lapl_M, h)
    t4 = time.time()
    np.testing.assert_array_almost_equal(divgrad_M, lapl_M, 8)
    #print("Duration of Gradient Front operation: ", (t2 - t1))
    #print("Duration of Divergence Back operation: ", (t3 - t2))
    #print("Duration of Laplace Neumann operation:", (t4 - t3))
    print("Differential Operator Test 1 passed.")

    print("Differential Operator Test 2")

    N2 = 1000
    N1 = 1002
    h = 1.0
    M = np.random.random((N2+1,N1+1))
    lapl_M = np.zeros((N2,N1), dtype = np.float64)
    gradx_M = np.zeros((N2+1,N1+1), dtype = np.float64)
    grady_M = np.zeros((N2+1,N1+1), dtype = np.float64)
    divgrad_M = np.zeros((N2+1,N1+1), dtype = np.float64)

    t1 = time.time()
    pd0.gradB(M, gradx_M, grady_M, h)
    t2 = time.time()
    pd0.divF(gradx_M, grady_M, divgrad_M, h)
    t3 = time.time()
    pd0.laplacian_diri(M[0:N2, 0:N1], lapl_M, h)
    t4 = time.time()
    np.testing.assert_array_almost_equal(divgrad_M[0:N2, 0:N1], lapl_M, 8)
    #print("Duration of Gradient Back operation: ", (t2 - t1))
    #print("Duration of Divergence Front operation: ", (t3 - t2))
    #print("Duration of Laplace Dirichlet operation:", (t4 - t3))
    print("Differential Operator Test 2 passed.")


def proj_with_dct_test():
    #Test, whether it is a projection (Pi tau = Pi Pi tau) and
    # whether divergence is 0 (div Pi tau = 0)

    print("Project Neumann Test")

    N2 = 10
    N1 = 10
    h = 1.0

    tau = np.random.random((2,N2,N1))
    Pi_tau = np.zeros((2,N2,N1), dtype = np.float64)
    Pi_Pi_tau = np.zeros((2,N2,N1), dtype = np.float64)
    div_Pi_tau = np.zeros((N2,N1), dtype=np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
    tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
    
    pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
    t1 = time.time()
    Pi_tau = pd0.proj_neumann_dct(tau, Pi_tau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
    t2 = time.time()
    print("Duration of one Neumann Projection: ", (t2 - t1))
    if(np.max(np.abs(Pi_tau)) < 0.00001):
        print("WARNING: Pi_tau = 0")
    Pi_Pi_tau = pd0.proj_neumann_dct(Pi_tau, Pi_Pi_tau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
    np.testing.assert_array_almost_equal(Pi_Pi_tau, Pi_tau, 8)
    print("Pi Pi tau = Pi tau  passed.")
    pd0.divB(Pi_tau[0], Pi_tau[1], div_Pi_tau, h)
    np.testing.assert_array_almost_equal(div_Pi_tau, np.zeros((N2,N1), dtype=np.float64), 8)
    print("div Pi tau = 0  passed.")
    print("Project Neumann Test passed.")



def proj_with_dst_test():
    #Test, whether it is a projection (Pi tau = Pi Pi tau) and
    # whether divergence is 0 (div Pi tau = 0)

    print("Project Dirichlet Test")

    N2 = 10
    N1 = 10
    N2p = N2 + 1
    N1p = N1 + 1
    h = 1.0

    tau = np.random.random((2,N2p,N1p))
    Pi_tau = np.zeros((2,N2p,N1p), dtype = np.float64)
    Pi_Pi_tau = np.zeros((2,N2p,N1p), dtype = np.float64)
    div_Pi_tau = np.zeros((N2p,N1p), dtype=np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2p,N1p), dtype=np.float64)
    tmp_mat2 = np.zeros((N2p,N1p), dtype=np.float64)
    
    pd0.prepare_Sigma_sq_dst(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dst(Sigma2_sq, h)
    t1 = time.time()
    Pi_tau = pd0.proj_dirichlet_dst(tau, Pi_tau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
    t2 = time.time()
    print("Duration of one Dirichlet Projection: ", (t2 - t1))
    if(np.max(np.abs(Pi_tau)) < 0.00001):
        print("WARNING: Pi_tau = 0")
    Pi_Pi_tau = pd0.proj_dirichlet_dst(Pi_tau, Pi_Pi_tau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
    np.testing.assert_array_almost_equal(Pi_Pi_tau, Pi_tau, 8)
    print("Pi Pi tau = Pi tau  passed.")
    pd0.divF(Pi_tau[0], Pi_tau[1], div_Pi_tau, h)
    np.testing.assert_array_almost_equal(div_Pi_tau, np.zeros((N2p,N1p), dtype=np.float64), 8)
    print("div Pi tau = 0  passed.")
    print("Project Dirichlet Test passed.")


def main():
    print("===========================================================")
    print("TESTS proj_div_0.pyx START")
    print("===========================================================")
    dct_tests()
    inv_laplace_neum_test()
    inv_laplace_diri_test()
    diff_operator_test()
    proj_with_dct_test()
    proj_with_dst_test()
    print("============================================================")
    print("TESTS proj_div_0.pyx FINISHED SUCCESSFULLY")
    print("============================================================")
    

if __name__ == "__main__":
    main()
