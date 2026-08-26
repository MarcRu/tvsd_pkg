import copy
import math
import numpy as np
import time

import proj_div_0 as pd0
import proj_div_0_low_cap as pd0lc
import proj_div_0_low_cap_base as pd0lcb
from partition import Partition, HaloPartition


from scipy.fft import dct, idct

def dct_matrix(N):
    C = np.zeros([N,N], dtype=np.float64)
    C[0,:] = math.sqrt(1.0 / float(N)) * np.ones((1,N), dtype=float)
    for i in range(1, N):
        for j in range(N):
            C[i,j] = math.sqrt(2.0 / float(N)) * math.cos(math.pi / float(N) * float(i) * (float(j) + 0.5) )
    return C


def part_dct_test():

    print("Partial DCT Test")
    N1 = 3
    N2 = 3
    lb = 1
    rb = 2
    ub = 1
    bb = 2
    nlb = 0     #new left bound
    nrb = 2     #new right bound
    nub = 0     #new upper bound
    nbb = 2     #new bottom bound

    mat = np.zeros((N2, N1), dtype=np.float64)
    mat[ub:bb,lb:rb] = np.random.random((bb-ub,rb-lb))
    pdct_x_mat = np.zeros((bb-ub,nrb-nlb), dtype=np.float64)
    pidct_x_mat = np.zeros((bb-ub,nrb-nlb), dtype=np.float64)
    pdct_y_mat = np.zeros((nbb-nub,rb-lb), dtype=np.float64)
    pidct_y_mat = np.zeros((nbb-nub,rb-lb), dtype=np.float64)

    pd0lcb.pdct_x(mat[ub:bb,lb:rb], pdct_x_mat, N1, bb-ub, lb, rb, nlb, nrb)
    pd0lcb.pidct_x(mat[ub:bb,lb:rb], pidct_x_mat, N1, bb-ub, lb, rb, nlb, nrb)
    pd0lcb.pdct_y(mat[ub:bb,lb:rb], pdct_y_mat, N2, rb-lb, ub, bb, nub, nbb)
    pd0lcb.pidct_y(mat[ub:bb,lb:rb], pidct_y_mat, N2, rb-lb, ub, bb, nub, nbb)
    
    dct_x_mat = dct(mat, type=2, axis=1, norm='ortho')
    idct_x_mat = idct(mat, type=2, axis=1, norm='ortho')
    dct_y_mat = dct(mat, type=2, axis=0, norm='ortho')
    idct_y_mat = idct(mat, type=2, axis=0, norm='ortho')

    np.testing.assert_array_almost_equal(pdct_x_mat, dct_x_mat[ub:bb, nlb:nrb], 7)
    np.testing.assert_array_almost_equal(pidct_x_mat, idct_x_mat[ub:bb, nlb:nrb], 7)
    np.testing.assert_array_almost_equal(pdct_y_mat, dct_y_mat[nub:nbb, lb:rb], 7)
    np.testing.assert_array_almost_equal(pidct_y_mat, idct_y_mat[nub:nbb, lb:rb], 7)

    print("Partial DCT Test passed.")


def inv_laplace_sparse_part_test():
    #Test by comparison of proj_div_0_low_cap with proj_div_0. 
    #Assumes that proj_div_0 is already tested.

    N2 = 1000
    N1 = 1500
    h = 1.0

    print("Sparse Pseudo Inverse Laplace Comparison Test")
    mat = np.zeros((N2, N1), dtype=np.float64)
    m1 = 1
    m2 = 3
    lbounds = np.asarray([0, 300, 600, 900, 1200])
    ubounds = np.asarray([0, 200, 400, 600, 800, 900])
    partition = Partition(N1, N2, lbounds, ubounds)  
    mat[ubounds[m2]:(ubounds[m2]+partition.sizes2[m2]), \
        lbounds[m1]:(lbounds[m1]+partition.sizes1[m1])] \
        = np.random.random((partition.sizes2[m2],partition.sizes1[m1]))

    #ground truth
    invlapl_mat_pd0 = np.zeros((N2,N1), dtype = np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
    tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
    t3 = time.time()
    pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
    invlapl_mat_pd0 = pd0.inv_laplacian_neum(mat, invlapl_mat_pd0, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    t4 = time.time()
    invlapl_mat_pd0_part = invlapl_mat_pd0[ \
                    ubounds[m2]:(ubounds[m2]+partition.sizes2[m2]), \
                    lbounds[m1]:(lbounds[m1]+partition.sizes1[m1])]

    #test case
    t1 = time.time()
    mat_part = mat[ubounds[m2]:(ubounds[m2]+partition.sizes2[m2]), \
                    lbounds[m1]:(lbounds[m1]+partition.sizes1[m1])]
    invlaplobj = pd0lc.InvLapl(partition, h)
    invlapl_mat_pd0lc_part = invlaplobj.part_inv_sparse(mat_part, m1, m2)
    t2 = time.time()

    #print(np.array(invlapl_mat_pd0_part))   #ground truth
    #print(np.array(invlapl_mat_pd0lc_part)) #to be tested

    np.testing.assert_array_almost_equal( \
                        invlapl_mat_pd0_part, \
                        invlapl_mat_pd0lc_part, 7)

    print("Duration of Inv Lapl, proj_div_0_low_cap: ", (t2 - t1))
    print("Duration of Inv Lapl, proj_div_0: ", (t4 - t3))
    print("Sparse Pseudo Inverse Laplace Comparison Test passed.")


def total_inv_laplace_low_cap_test():

    N2 = 100
    N1 = 102
    h = 1.0

    print("Total Pseudo Inverse Laplace Neumann Test")
    mat = np.zeros((N2, N1), dtype=np.float64)
    m1 = 4
    m2 = 2
    lbounds = np.asarray([0, 30, 46, 60, 90])
    ubounds = np.asarray([0, 20, 40, 60, 80, 90])
    M1 = lbounds.shape[0]
    M2 = ubounds.shape[0]
    partition = Partition(N1, N2, lbounds, ubounds)  
    mat = np.random.random((N2, N1))

    #ground truth
    invlapl_mat_pd0 = np.zeros((N2,N1), dtype = np.float64)
    Sigma1_sq = np.zeros((N1), dtype=np.float64)
    Sigma2_sq = np.zeros((N2), dtype=np.float64)
    tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
    tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
    t3 = time.time()
    pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
    invlapl_mat_pd0 = pd0.inv_laplacian_neum(mat, invlapl_mat_pd0, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2)
    t4 = time.time()
    invlapl_mat_pd0_part = invlapl_mat_pd0[ \
                    ubounds[m2]:(ubounds[m2]+partition.sizes2[m2]), \
                    lbounds[m1]:(lbounds[m1]+partition.sizes1[m1])]

    #test case
    t1 = time.time()
    invlapl_mat_pd0lc_part = np.zeros( \
                (partition.sizes2[m2], partition.sizes1[m1]), dtype=np.float64)
    invlaplobj = pd0lc.InvLapl(partition, h)
    for k2 in range(M2):
        for k1 in range(M1):
            mat_part = mat[ubounds[k2]:partition.bbounds[k2], \
                    lbounds[k1]:partition.rbounds[k1]]
            #print("k2,k1", k2,k1)
            invlapl_mat_pd0lc_part += \
                    invlaplobj.part_inv_sparse_general( \
                            mat_part, k1, k2, m1, m2)
    t2 = time.time()

    #print(np.array(invlapl_mat_pd0_part))   #ground truth
    #print(np.array(invlapl_mat_pd0lc_part)) #to be tested

    np.testing.assert_array_almost_equal( \
                        invlapl_mat_pd0_part, \
                        invlapl_mat_pd0lc_part, 7)

    print("Duration of Inv Lapl, proj_div_0_low_cap: ", (t2 - t1))
    print("Duration of Inv Lapl, proj_div_0: ", (t4 - t3))



    print("Total Pseudo Inverse Laplace Neumann Test passed.")


def proj_low_cap_test():
    #Test, whether it is a projection (Pi tau = Pi Pi tau) and
    # whether divergence is 0 (div Pi tau = 0)

    print("Project Low Capacity Test")

    N2 = 7
    N1 = 12
    h = 1.0
    lbounds = np.array([0])
    ubounds = np.array([0, 3])
    M1 = lbounds.shape[0]
    M2 = ubounds.shape[0]
    k1 = 0
    k2 = 1
    ma1 = 0
    ma2 = 0
    partition = HaloPartition(N1, N2, lbounds, ubounds)
    proj_obj = pd0lc.ProjectorDiv0(partition, h)

    #tau lives only on domain (m2, m1)
    tau1 = np.random.random((2,partition.sizes2[k2],partition.sizes1[k1]))
    tau2 = np.random.random((2,partition.sizes2[k2],partition.sizes1[k1]))
    #Pi_tau lives on all domains!!!
    Pi_tau1 = np.ones((2,N2,N1), dtype=np.float64)
    Pi_tau2 = np.ones((2,N2,N1), dtype=np.float64)
    #Pi_Pi_tau lives on all domains,
    # but gets evaluated only for (ma2,ma1) for test purposes
    Pi_Pi_tau1 = np.zeros((2,partition.sizes2[ma2],partition.sizes1[ma1]))
    tau1_compl = np.zeros((2,N2,N1), dtype = np.float64)
    tau1_compl[:, partition.ubounds[k2]:partition.bbounds[k2], \
                    partition.lbounds[k1]:partition.rbounds[k1]] \
                = tau1
    tau2_compl = np.zeros((2,N2,N1), dtype = np.float64)
    tau2_compl[:, partition.ubounds[k2]:partition.bbounds[k2], \
                    partition.lbounds[k1]:partition.rbounds[k1]] \
                = tau2

    #Actual test
    t1 = time.time()
    for m1 in range(M1):
        for m2 in range(M2):
            Pi_tau1_m = proj_obj.proj_sparse_general(tau1, k1, k2, m1, m2)
            Pi_tau2_m = proj_obj.proj_sparse_general(tau2, k1, k2, m1, m2)
            Pi_tau1[:, partition.ubounds[m2]:partition.bbounds[m2], \
                        partition.lbounds[m1]:partition.rbounds[m1]] \
                        = Pi_tau1_m
            Pi_tau2[:, partition.ubounds[m2]:partition.bbounds[m2], \
                        partition.lbounds[m1]:partition.rbounds[m1]] \
                        = Pi_tau2_m
    t2 = time.time()
    if(np.max(np.abs(Pi_tau1)) < 0.00001):
        print("WARNING: Pi_tau1 = 0")
    if(np.max(np.abs(Pi_tau2)) < 0.00001):
        print("WARNING: Pi_tau2 = 0")
    print("Duration of one Neumann Projection for a sparse matrix: ", 0.5 * (t2 - t1))
    div_Pi_tau1 = np.ones((N2,N1))
    pd0.divB(Pi_tau1[0], Pi_tau1[1], div_Pi_tau1, h)
    np.testing.assert_array_almost_equal(div_Pi_tau1, np.zeros((N2,N1), dtype=np.float64), 8)
    print("div Pi tau1 = 0  passed.")
    for m1 in range(M1):
        for m2 in range(M2):
            Pi_tau1_m = Pi_tau1[:,\
                    partition.ubounds[m2]:partition.bbounds[m2], \
                    partition.lbounds[m1]:partition.rbounds[m1]]
            Pi_Pi_tau1 += proj_obj.proj_sparse_general(Pi_tau1_m, m1, m2, ma1, ma2)
    Pi_tau1_ma2_ma1 = Pi_tau1[: ,partition.ubounds[ma2]:partition.bbounds[ma2], \
                        partition.lbounds[ma1]:partition.rbounds[ma1]]
    np.testing.assert_array_almost_equal(Pi_Pi_tau1, Pi_tau1_ma2_ma1, 8)
    print("Pi Pi tau1 = Pi tau1  passed.")
    Pi_tau1_dot_tau2 = np.dot(Pi_tau1.flatten(), tau2_compl.flatten())
    Pi_tau2_dot_tau1 = np.dot(Pi_tau2.flatten(), tau1_compl.flatten())
    np.testing.assert_almost_equal(Pi_tau1_dot_tau2, Pi_tau2_dot_tau1, 8)
    print("<Pi tau_1, tau_2> = <tau_1, Pi tau_2> passed.")
    print("Project Low Capacity Test passed.")




def main():
    print("===========================================================")
    print("TESTS proj_div_0_low_cap.py START")
    print("===========================================================")
    part_dct_test()
    inv_laplace_sparse_part_test()
    total_inv_laplace_low_cap_test()
    proj_low_cap_test()
    print("============================================================")
    print("TESTS proj_div_0_low_cap.py FINISHED SUCCESSFULLY")
    print("============================================================")


if __name__ == "__main__":
    main()