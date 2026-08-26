
import copy
import numpy as np
from mpi4py import MPI
import pickle
import os, sys
import math
import cv2

import proj_div_0 as pd0
from partition import ThreadedHUIP
from tvsd_dd import TRTangentFieldSmoothing as Trtfs
from tvsd_dd import TvStokesDualDD
from tvsd_dd import ParallelArray, Recorder, TRIntersectingDDAlgorithm, \
                    parallelize, deparallelize
from tvsd import tv_stokes_dual, tv_stokes_dual_tfs_only, tv_stokes_dual_ir_only


# Ensure the script can find modules relative to the script’s directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


"""
conda activate tvsd_env
mpirun --hostfile hostfile -np 2 python par_tests.py
"""


def tridda_global_forw_diff_test(comm):
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    assert num_ranks >= 2

    if rank < 2:
        print("TEST tridda_global_forw_diff_test START, Rank " + str(rank))

    N2 = 8
    N1 = 8
    h = 1.0
    lbounds = np.array([0, 3])
    rbounds = np.array([5, 8])
    ubounds = np.array([0, 3])
    bbounds = np.array([5, 8])
    thread_map = np.array([[0,1],[1,0]])

    if (rank == 0):
        d_arr = {(0,0) : np.array(
            [[1, 0, 1, 0, 1], \
            [0, 1, 3, 1, 0], \
            [1, 6, 1, 0, 1], \
            [0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1]], dtype=np.float64), \
                (1,1) : np.array(
            [[1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1]], dtype=np.float64)}
        tau_gt_arr = {(0,0) : np.array( \
            [[[ 1., -1.,  -2., -1.,  1.], \
            [-1.,  -5., 2.,  1., -1.], \
            [ 1.,  5.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.]], \
            [[-1.,  1., -1.,  1., -1.], \
            [ 1.,  2.,  -2., -1.,  1.], \
            [ 5.,  -5., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.]]], dtype=np.float64), \
                (1,1) : np.array( \
            [[[ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 0.,  0., 0.,  0., 0.]], \
            [[-1.,  1., -1.,  1., 0.], \
            [ 1., -1.,  1., -1., 0.], \
            [-1.,  1., -1.,  1., 0.], \
            [ 1., -1.,  1., -1., 0.], \
            [-1.,  1., -1.,  1., 0.]]], dtype=np.float64)}
    elif (rank == 1):
        d_arr = {(0,1) : np.array(
            [[0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0]], dtype=np.float64), \
                (1,0) : np.array(
            [[0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0], \
            [1, 0, 1, 0, 1], \
            [0, 1, 0, 1, 0]], dtype=np.float64)}
        tau_gt_arr = {(0,1) : np.array( \
            [[[-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.]], \
            [[ 1., -1.,  1., -1., 0.], \
            [-1.,  1., -1.,  1., 0.], \
            [ 1., -1.,  1., -1., 0.], \
            [-1.,  1., -1.,  1., 0.], \
            [ 1., -1.,  1., -1., 0.]]], dtype=np.float64), \
                (1,0) : np.array( \
            [[[-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [0.,  0., 0.,  0., 0.]], \
            [[ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.], \
            [-1.,  1., -1.,  1., -1.], \
            [ 1., -1.,  1., -1.,  1.]]], dtype=np.float64)}
    else:
        d_arr = {}
        tau_gt_arr = {}

    d = ParallelArray(d_arr)
    tau_gt = ParallelArray(tau_gt_arr)
    #prepare DD-object
    dd = ThreadedHUIP(N1, N2, lbounds,rbounds,ubounds,bbounds,0,1,0,1,thread_map,num_ranks)
    #Apply function to be tested
    dummy_rec = Recorder.dummy()
    dummy_par = {}
    tridda_obj = TRIntersectingDDAlgorithm(dd, dummy_par, dummy_rec, comm)
    tridda_obj.h = h
    comm.Barrier()
    dx_d_l, dy_d_l = tridda_obj.global_forw_diff([d], [d])
    if rank < 2:
        tau_1, tau_2 = dy_d_l[0] * (-1.0), dx_d_l[0]
        tau = tau_1.stack(tau_2, axis=0)
    comm.Barrier()

    #Now do actual test
    if rank < 2:
        for m2, m1 in tridda_obj.dd.relevant_domains[rank]:
            try:
                np.testing.assert_array_almost_equal(tau[(m2,m1)], tau_gt[(m2,m1)], 8)
            except AssertionError as e:
                print("An AssertionError has appeared for (m2,m1) = " + str((m2,m1)) + "!\n"
                    "Message: ", e, "\n"
                    "TEST tridda_global_forw_diff_test FAILED. Rank " + str(rank))
                sys.exit()
        print("TEST tridda_global_forw_diff_test PASSED SUCCESSFULLY. Rank " + str(rank))


def tridda_global_back_diff_ext_test(comm):
    #Test, whether the result of 
    #  TRIntersectingDDAlgorithm.global_back_diff_ext
    #  gives the correct result
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST tridda_global_back_diff_ext_test START, Rank " + str(rank))

    N2 = 201
    N1 = 222
    M1 = 5
    M2 = 4
    overlap_x = 4
    overlap_y = 3
    h = 1.0

    #prepare DD-object
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)
    dd_ext = dd.duplicate_with_additional_row_and_col()

    if (rank == 0):
        img_total = np.random.random((N2,N1))
        img_arr = {}
        for m2 in range(M2):
            for m1 in range(M1):
                img_arr[(m2,m1)] = img_total[ \
                    dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]]

        #ground truth: use proj_div_0.integrate_tangentfield
        dx_gt = np.zeros((N2+1, N1+1), dtype=img_total.dtype)
        dy_gt = np.zeros((N2+1, N1+1), dtype=img_total.dtype)
        dx_gt[:-1,1:-1] = (1. / h) * (img_total[:,1:] - img_total[:,:-1])
        dy_gt[1:-1,:-1] = (1. / h) * (img_total[1:,:] - img_total[:-1,:])
        dx_gt[-1,1:-1] = np.copy(dx_gt[-2,1:-1])
        dy_gt[1:-1,-1] = np.copy(dy_gt[1:-1,-2])
        

    #Distribute image arrays to ranks
    #Create image dics for all ranks
    if (rank == 0):
        img_all_ranks = []
        for r in range(num_ranks):
            img_dic = {}
            for m2,m1 in dd.relevant_domains[r]:
                img_dic[(m2,m1)] = copy.deepcopy(img_total[ \
                        dd_ext.ubounds[m2]:dd_ext.bbounds[m2], \
                        dd_ext.lbounds[m1]:dd_ext.rbounds[m1]])
            img = ParallelArray(img_dic)
            img_all_ranks.append(img)
        img = img_all_ranks[0]
        #Distribute all data to other ranks
        for r in range(1, num_ranks):
            #Serialize data (necessary, since data structure is complicated)
            ser_data_to_r = pickle.dumps(img_all_ranks[r])
            data_size = len(ser_data_to_r)
            #send data size
            comm.isend(data_size, dest=r, tag=100+r).wait()
            #Send actual data
            comm.Isend([ser_data_to_r, MPI.BYTE], dest=r, tag=200+r).Wait()
    # Receive data on other ranks
    if (rank != 0):
        #Receive data size
        data_length = comm.irecv(source=0, tag=100+rank).wait()
        #Allocate enough space
        ser_data_to_r = bytearray(data_length)
        #Receive actual data
        comm.Irecv([ser_data_to_r, MPI.BYTE], source=0, tag=200+rank).Wait()
        #Deserialize
        img = pickle.loads(ser_data_to_r)

    #Apply function to be tested
    dummy_rec = Recorder.dummy()
    dummy_par = {}
    triddalg_obj = TRIntersectingDDAlgorithm(dd, dummy_par, dummy_rec, comm)
    triddalg_obj.h = h
    comm.Barrier()
    
    dx_img_list, dy_img_list = triddalg_obj.global_back_diff_ext([img],[img])
    dx_img = dx_img_list[0]
    dy_img = dy_img_list[0]
    comm.Barrier()

    #Put splitted array together
    #Send parts from all ranks
    if (rank != 0):
        ser_data_from_r = pickle.dumps((dx_img, dy_img))
        data_size = len(ser_data_from_r)
        #send data size
        comm.isend(data_size, dest=0, tag=100+rank).wait()
        #Send actual data
        comm.Isend([ser_data_from_r, MPI.BYTE], dest=0, tag=200+rank).Wait()
    #Collect parts from all ranks in Rank 0
    if (rank == 0):
        #create list for all ranks
        dx_all_ranks = [None for _ in range(num_ranks)]
        dy_all_ranks = [None for _ in range(num_ranks)]  
        dx_all_ranks[0] = dx_img
        dy_all_ranks[0] = dy_img
        #Receive parts from all ranks
        for r in range(1, num_ranks):
            #Receive data size
            data_length_r = comm.irecv(source=r, tag=100+r).wait()
            #Allocate enough space
            ser_data_from_r = bytearray(data_length_r)
            #Receive actual data
            comm.Irecv([ser_data_from_r, MPI.BYTE], source=r, tag=200+r).Wait()
            #Deserialize
            dx_r, dy_r = pickle.loads(ser_data_from_r)
            dx_all_ranks[r] = dx_r
            dy_all_ranks[r] = dy_r
        #Now reconstruct complete dx_img and dy_img
        dx_img = np.zeros((N2+1,N1+1))
        dy_img = np.zeros((N2+1,N1+1))
        for r in range(num_ranks):
            dd_ext.init_domains(dd_ext.relevant_domains[r])
            for m2, m1 in dd_ext.relevant_domains[r]:
                dx_img[dd_ext.ubounds[m2]:dd_ext.bbounds[m2],dd_ext.lbounds[m1]:dd_ext.rbounds[m1]] \
                    += dx_all_ranks[r][(m2,m1)] * dd_ext.theta_loc(m2, m1)
                dy_img[dd_ext.ubounds[m2]:dd_ext.bbounds[m2],dd_ext.lbounds[m1]:dd_ext.rbounds[m1]] \
                    += dy_all_ranks[r][(m2,m1)] * dd_ext.theta_loc(m2, m1)
        #Now do actual test
        try:
            np.testing.assert_array_almost_equal(dx_img, dx_gt, 8)
        except AssertionError as e:
            print("AssertionError: dx_img != dx_gt")
            print("Message: ", e)
            print("TEST test_find_backdiff_scalar_potential FAILED. Rank 0")
            return
        try:
            np.testing.assert_array_almost_equal(dy_img, dy_gt, 8)
        except AssertionError as e:
            print("AssertionError: dy_img != dy_gt")
            print("Message: ", e)
            print("TEST test_find_backdiff_scalar_potential FAILED. Rank 0")
            return
        print("TEST tridda_global_back_diff_ext_test PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST tridda_global_back_diff_ext_test routine finished. Rank " + str(rank))


def trtfs_proj_div_0_compl_test(comm):
    #Test, whether the result of TRTangentFieldSmoothing.proj_div_0_compl
    #is delivering the same result as proj_div_0.pyx
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST trtfs_proj_div_0_compl_test START, Rank " + str(rank))

    #N2 = 100
    #N1 = 102
    #M1 = 3
    #M2 = 4
    #overlap_x = 15
    #overlap_y = 10
    N2 = 10
    N1 = 5
    overlap_x = 1
    overlap_y = 3
    M1 = 1
    M2 = 2
    ext_x = 1
    ext_y = 1
    h = 1.0

    if (rank == 0):
        tau = np.random.random((2,N2,N1))
        #Ground truth
        Pi_tau_gt = np.zeros((2,N2,N1), dtype = np.float64)
        Sigma1_sq = np.zeros((N1), dtype=np.float64)
        Sigma2_sq = np.zeros((N2), dtype=np.float64)
        tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
        tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
        pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
        pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
        Pi_tau_gt = pd0.proj_neumann_dct(tau, Pi_tau_gt, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)

    #prepare DD-object
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)
    
    #Split image arrays and distribute them to ranks
    if (rank == 0):
        #Create image dics for all ranks
        tau_arrs = []
        for r in range(num_ranks):
            tau_dic = {}
            for m2,m1 in dd.relevant_domains[r]:
                tau_dic[(m2,m1)] = tau[:, \
                            dd.ubounds[m2]:dd.bbounds[m2], \
                            dd.lbounds[m1]:dd.rbounds[m1]]
            tau_arr = ParallelArray(copy.deepcopy(tau_dic))
            tau_arrs.append(tau_arr)
        tau_arr = tau_arrs[0]
        #Distribute all data to other ranks
        for r in range(1, num_ranks):
            #Serialize data (necessary, since data structure is complicated)
            ser_data_to_r = pickle.dumps(tau_arrs[r])
            data_size = len(ser_data_to_r)
            #send data size
            comm.isend(data_size, dest=r, tag=100+r).wait()
            #Send actual data
            comm.Isend([ser_data_to_r, MPI.BYTE], dest=r, tag=200+r).Wait()
    # Receive data on other ranks
    if (rank != 0):
        #Receive data size
        data_length = comm.irecv(source=0, tag=100+rank).wait()
        #Allocate enough space
        ser_data_to_r = bytearray(data_length)
        #Receive actual data
        comm.Irecv([ser_data_to_r, MPI.BYTE], source=0, tag=200+rank).Wait()
        #Deserialize
        tau_arr = pickle.loads(ser_data_to_r)

    #Apply function to be tested
    dummy_rec = Recorder.dummy()
    dummy_par = {}
    tfs_obj = Trtfs(dd, dummy_par, dummy_rec, comm)
    tfs_obj.h = h
    comm.Barrier()
    Pi_tau_arr = tfs_obj.proj_div_0_compl(tau_arr, ext_x, ext_y)
    comm.Barrier()

    #Put splitted array together
    #Send parts from all ranks
    if (rank != 0):
        ser_data_from_r = pickle.dumps(Pi_tau_arr)
        data_size = len(ser_data_from_r)
        #send data size
        comm.isend(data_size, dest=0, tag=100+rank).wait()
        #Send actual data
        comm.Isend([ser_data_from_r, MPI.BYTE], dest=0, tag=200+rank).Wait()
    #Collect parts from all ranks in Rank 0
    if (rank == 0):
        #create list for all ranks
        all_Pi_tau_ext_arr = [None for _ in range(num_ranks)]   
        all_Pi_tau_ext_arr[0] = Pi_tau_arr
        #Receive parts from all ranks
        for r in range(1, num_ranks):
            #Receive data size
            data_length_r = comm.irecv(source=r, tag=100+r).wait()
            #Allocate enough space
            ser_data_from_r = bytearray(data_length_r)
            #Receive actual data
            comm.Irecv([ser_data_from_r, MPI.BYTE], source=r, tag=200+r).Wait()
            #Deserialize
            all_Pi_tau_ext_arr[r] = pickle.loads(ser_data_from_r)
        #Now reconstruct image from all parts
        Pi_tau = np.zeros((2, N2, N1), dtype=np.float64)
        for r in range(num_ranks):
            dd.init_domains(dd.relevant_domains[r])
            for m2, m1 in dd.relevant_domains[r]:
                slice_rows = slice(None) if m2 == dd.M2 - 1 or ext_y == 0 else slice(None, -ext_y)
                slice_cols = slice(None) if m1 == dd.M1 - 1 or ext_x == 0 else slice(None, -ext_x)
                Pi_tau_arr_m_0 = all_Pi_tau_ext_arr[r][(m2,m1)][0,slice_rows,slice_cols]
                Pi_tau_arr_m_1 = all_Pi_tau_ext_arr[r][(m2,m1)][1,slice_rows,slice_cols]
                Pi_tau[0, dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                        += Pi_tau_arr_m_0 * dd.theta_loc(m2, m1)
                Pi_tau[1, dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                        += Pi_tau_arr_m_1 * dd.theta_loc(m2, m1)
        
        #Now do actual test
        np.testing.assert_array_almost_equal(Pi_tau, np.array(Pi_tau_gt), 8)

        print("TEST trtfs_proj_div_0_compl_test PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST trtfs_proj_div_0_compl_test routine finished. Rank " + str(rank))


    
def trtfs_compute_P_div_q0_test(comm):
    """
    Test, whether the result of the local function
       TRTangentFieldSmoothing.compute_P_div_q0
    is delivering the same result as computing P_div_q0 globally:
     1) sum_pl = sum_l theta_l pl
     2) P_K div sum_pl globally: div globally, P_K globally
     3) P_K div p_m globally: div globally, P_K globally
     4) P_K div (q_m^0) = P_K div(sum_pl - p_m) 
                        = P_K div sum_pl - P_K div p_m
    """
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST trtfs_compute_P_div_q0_test START, Rank " + str(rank))

    N2 = 100
    N1 = 102
    M1 = 4
    M2 = 3
    overlap_x = 15
    overlap_y = 10
    h = 1.0

    #prepare DD-object
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)

    if (rank == 0):
        p = {}
        for m2 in range(M2):
            for m1 in range(M1):
                p[(m2,m1)] = np.random.random((2,2,dd.sizes2[m2],dd.sizes1[m1]))
        
        #Ground truth: Compute for all m=(m2,m1):   P_K div q_0^m
        #       = P_K div(sum_l theta_l p_l^j) - P_K div (theta_m p_m^j)
        
        #P_K div(sum_l theta_l p_l^j)
        sum_pl = np.zeros((2,2,N2,N1), dtype=np.float64)
        for m2 in range(M2):
            for m1 in range(M1):
                dd.init_domains([(m2,m1)])
                sum_pl[0,0,dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                            += p[(m2,m1)][0,0,:,:] * dd.theta_loc(m2, m1)
                sum_pl[0,1,dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                            += p[(m2,m1)][0,1,:,:] * dd.theta_loc(m2, m1)
                sum_pl[1,0,dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                            += p[(m2,m1)][1,0,:,:] * dd.theta_loc(m2, m1)
                sum_pl[1,1,dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                            += p[(m2,m1)][1,1,:,:] * dd.theta_loc(m2, m1)
        div_sum_pl = np.zeros((2,N2,N1), dtype=np.float64)
        pd0.divB(sum_pl[0,0,:,:],sum_pl[0,1,:,:],div_sum_pl[0,:,:],h)
        pd0.divB(sum_pl[1,0,:,:],sum_pl[1,1,:,:],div_sum_pl[1,:,:],h)
        Pi_div_sum_pl = np.zeros((2,N2,N1), dtype = np.float64)
        Sigma1_sq = np.zeros((N1), dtype=np.float64)
        Sigma2_sq = np.zeros((N2), dtype=np.float64)
        tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
        tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
        pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
        pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
        Pi_div_sum_pl = pd0.proj_neumann_dct(div_sum_pl, Pi_div_sum_pl, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)

        #P_K div (theta_m p_m^j)  and  P_K div q_0^m
        Pi_div_q0_gt = {}
        for m2 in range(M2):
            for m1 in range(M1):
                #P_K div (theta_m p_m^j)
                theta_p_m = np.zeros((2,2,N2,N1))
                theta_p_m[0,0,dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]] \
                        = p[(m2,m1)][0,0,:,:] * dd.theta_loc(m2,m1)
                theta_p_m[0,1,dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]] \
                        = p[(m2,m1)][0,1,:,:] * dd.theta_loc(m2,m1)
                theta_p_m[1,0,dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]] \
                        = p[(m2,m1)][1,0,:,:] * dd.theta_loc(m2,m1)
                theta_p_m[1,1,dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]] \
                        = p[(m2,m1)][1,1,:,:] * dd.theta_loc(m2,m1)
                div_theta_p_m = np.zeros((2,N2,N1))
                pd0.divB(theta_p_m[0,0,:,:],theta_p_m[0,1,:,:],div_theta_p_m[0,:,:],h)
                pd0.divB(theta_p_m[1,0,:,:],theta_p_m[1,1,:,:],div_theta_p_m[1,:,:],h)
                Pi_div_theta_p_m = np.zeros((2,N2,N1), dtype = np.float64)
                Sigma1_sq = np.zeros((N1), dtype=np.float64)
                Sigma2_sq = np.zeros((N2), dtype=np.float64)
                tmp_mat1 = np.zeros((N2,N1), dtype=np.float64)
                tmp_mat2 = np.zeros((N2,N1), dtype=np.float64)
                pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
                pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
                Pi_div_theta_p_m = pd0.proj_neumann_dct(div_theta_p_m, Pi_div_theta_p_m, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
                
                #P_K div q_0^m
                diff = np.array(Pi_div_sum_pl) - np.array(Pi_div_theta_p_m)
                ext_r = 0 if m1 == M1 - 1 else 1
                ext_b = 0 if m2 == M2 - 1 else 1
                sl_ud = slice(dd.ubounds[m2], dd.bbounds[m2] + ext_b)
                sl_lr = slice(dd.lbounds[m1], dd.rbounds[m1] + ext_r)
                Pi_div_q0_gt[(m2,m1)] = diff[:,sl_ud,sl_lr]
    
    
        #Distribute image arrays to ranks
        #Create image dics for all ranks
        p_arrs = []
        for r in range(num_ranks):
            p_dic = {}
            for m2,m1 in dd.relevant_domains[r]:
                p_dic[(m2,m1)] = copy.deepcopy(p[(m2,m1)])
            p_arr = ParallelArray(p_dic)
            p_arrs.append(p_arr)
        p_arr = p_arrs[0]
        #Distribute all data to other ranks
        for r in range(1, num_ranks):
            #Serialize data (necessary, since data structure is complicated)
            ser_data_to_r = pickle.dumps(p_arrs[r])
            data_size = len(ser_data_to_r)
            #send data size
            comm.isend(data_size, dest=r, tag=100+r).wait()
            #Send actual data
            comm.Isend([ser_data_to_r, MPI.BYTE], dest=r, tag=200+r).Wait()
    # Receive data on other ranks
    if (rank != 0):
        #Receive data size
        data_length = comm.irecv(source=0, tag=100+rank).wait()
        #Allocate enough space
        ser_data_to_r = bytearray(data_length)
        #Receive actual data
        comm.Irecv([ser_data_to_r, MPI.BYTE], source=0, tag=200+rank).Wait()
        #Deserialize
        p_arr = pickle.loads(ser_data_to_r)

        sum_pl = None

    #Apply function to be tested
    dummy_rec = Recorder.dummy()
    dummy_par = {}
    tfs_obj = Trtfs(dd, dummy_par, dummy_rec, comm)
    tfs_obj.h = h
    comm.Barrier()
    P_div_q0_arr = tfs_obj.compute_P_div_q0(p_arr)
    comm.Barrier()

    #Put splitted array together
    #Send parts from all ranks
    if (rank != 0):
        ser_data_from_r = pickle.dumps(P_div_q0_arr)
        data_size = len(ser_data_from_r)
        #send data size
        comm.isend(data_size, dest=0, tag=100+rank).wait()
        #Send actual data
        comm.Isend([ser_data_from_r, MPI.BYTE], dest=0, tag=200+rank).Wait()
    #Collect parts from all ranks in Rank 0
    if (rank == 0):
        #create list for all ranks
        all_Pi_div_q0_arr = [None for _ in range(num_ranks)]   
        all_Pi_div_q0_arr[0] = P_div_q0_arr
        #Receive parts from all ranks
        for r in range(1, num_ranks):
            #Receive data size
            data_length_r = comm.irecv(source=r, tag=100+r).wait()
            #Allocate enough space
            ser_data_from_r = bytearray(data_length_r)
            #Receive actual data
            comm.Irecv([ser_data_from_r, MPI.BYTE], source=r, tag=200+r).Wait()
            #Deserialize
            all_Pi_div_q0_arr[r] = pickle.loads(ser_data_from_r)
        #Now collect results for all (m2,m1)
        Pi_div_q0 = {}
        for r in range(num_ranks):
            dd.init_domains(dd.relevant_domains[r])
            for m2, m1 in dd.relevant_domains[r]:
                Pi_div_q0[(m2, m1)] = all_Pi_div_q0_arr[r][(m2,m1)]
        
        #Now do actual test
        for m2 in range(M2):
            for m1 in range(M1):
                try:
                    np.testing.assert_array_almost_equal(Pi_div_q0[(m2,m1)], Pi_div_q0_gt[(m2,m1)], 8)
                except AssertionError as e:
                    print("An AssertionError has appeared for (m2,m1) = " + str((m2,m1)) + "!")
                    print("Message: ", e)
                    print("TEST trtfs_compute_P_div_q0_test FAILED. Rank 0")
                    return

        print("TEST trtfs_compute_P_div_q0_test PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST trtfs_compute_P_div_q0_test routine finished. Rank " + str(rank))


def test_tangent_field_smoothing(comm):
    #Test, whether 
    # A) the result of TRTangentFieldSmoothing.run
    #    actually delivers a valid tangent field with:
    #    div(tau) == 0
    # B) the result of TRTangentFieldSmoothing.run
    #    is equal to a global dd-tfs
    # C) the result of simple tfs without dd from here is equal
    #    to the result of tvsd.py
    # D) the result of the global dd-tfs 
    #    is similar to a tfs without dd
    DO_TEST_D = False

    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST test_tangent_field_smoothing START, Rank " + str(rank))
    M1 = 3
    M2 = 3
    overlap_x = 3
    overlap_y = 4
    num_without_dd_long = 20000      #Test D
    num_without_dd_short = 20       #Test C
    num_outer_long = 5000           #Test D
    num_outer_short = 10            #Test A,B
    num_inner = 10                  #Test A,B
    alpha = 0.25
    delta = 0.15#0.0835
    h = 1.0
    k = 0.125

    #1) Create d0 for test case
    if (rank == 0):
        #d0 = np.random.random((N2,N1))
        test_image = "beach_nz_1_480.jpg"
        ground_truth = cv2.imread("../resources/" + test_image, cv2.IMREAD_GRAYSCALE)
        ground_truth = ground_truth.astype("float64") / 255
        gaussian = np.random.normal(loc=0, scale=math.sqrt(0.01), size=ground_truth.shape) 
        d0 = np.clip(ground_truth + gaussian, 0.0, 1.0, dtype=np.float64)
        N2, N1 = d0.shape
    if (rank != 0):
        d0 = None
        N2, N1 = None, None

    
    #2) Determine tau_without_dd with classic tangent field smoothing without dd (on 1 thread)
    if (rank == 0):
        #Extended measures for tangent field
        N2t, N1t = N2+1, N1+1  
        #projection preparations
        Sigma1_sq, Sigma2_sq  = np.zeros((N1t), dtype=np.float64), np.zeros((N2t), dtype=np.float64)
        tmp_mat1, tmp_mat2 = np.zeros((N2t, N1t), dtype=np.float64), np.zeros((N2t, N1t), dtype=np.float64)
        pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
        pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
        #prepare tau0
        tau0 = np.zeros((2, N2t, N1t), dtype=np.float64)
        #tau0: inner part
        tau0[0,1:-1,:-1] = (1. / h) * (-d0[1:,:] + d0[:-1,:])
        tau0[1,:-1,1:-1] = (1. / h) * (d0[:,1:] - d0[:,:-1])
        #right/bottom boundary: mirror (=Neumann boundary)
        tau0[0,1:-1,-1] = np.copy(tau0[0,1:-1,-2])
        tau0[1,-1,1:-1] = np.copy(tau0[1,-2,1:-1])
        #prepare tau0til
        tau0til = (1.0 / delta) * tau0
        #initialization
        p = np.zeros((2, 2, N2t, N1t), dtype=np.float64)
        #outer loop
        for j in range(num_without_dd_long):    
            div_p = np.zeros((2,N2t,N1t), dtype=np.float64)
            pd0.divB(p[0,0,:,:], p[0,1,:,:], div_p[0,:,:], h)
            pd0.divB(p[1,0,:,:], p[1,1,:,:], div_p[1,:,:], h)
            div_p_tau0 = div_p - tau0til
            P_div_p_tau0 = np.zeros((2,N2t,N1t), dtype=np.float64)
            P_div_p_tau0 = pd0.proj_neumann_dct( \
                    div_p_tau0, P_div_p_tau0, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
            psi = np.zeros((2,2,N2t,N1t), dtype=np.float64)
            pd0.gradF(P_div_p_tau0[0,:,:], psi[0,0,:,:], psi[0,1,:,:], h)
            pd0.gradF(P_div_p_tau0[1,:,:], psi[1,0,:,:], psi[1,1,:,:], h)
            num00 = p[0,0,:,:] + k * psi[0,0,:,:]
            num01 = p[0,1,:,:] + k * psi[0,1,:,:]
            num10 = p[1,0,:,:] + k * psi[1,0,:,:]
            num11 = p[1,1,:,:] + k * psi[1,1,:,:]
            abs_psi0 = np.sqrt(psi[0,0,:,:] ** 2 + psi[0,1,:,:] ** 2)
            abs_psi1 = np.sqrt(psi[1,0,:,:] ** 2 + psi[1,1,:,:] ** 2)
            den0 = k * abs_psi0 + 1.0
            den1 = k * abs_psi1 + 1.0
            #update
            p[0,0,:,:] = np.divide(num00, den0, out=np.zeros_like(num00, dtype=np.float64), where=(den0!=0.0))
            p[0,1,:,:] = np.divide(num01, den0, out=np.zeros_like(num01, dtype=np.float64), where=(den0!=0.0))
            p[1,0,:,:] = np.divide(num10, den1, out=np.zeros_like(num10, dtype=np.float64), where=(den1!=0.0))
            p[1,1,:,:] = np.divide(num11, den1, out=np.zeros_like(num11, dtype=np.float64), where=(den1!=0.0))
            if j % 100 == 0 and j > 0:
                print("Without DD it " + str(j))  
            #get tau_gt from p
            if j == num_without_dd_long - 1 or j == num_without_dd_short - 1:
                div_p = np.zeros((2,N2t,N1t), dtype=np.float64)
                pd0.divB(p[0,0,:,:], p[0,1,:,:], div_p[0,:,:], h)
                pd0.divB(p[1,0,:,:], p[1,1,:,:], div_p[1,:,:], h)
                tau0_div_p = tau0 - delta * div_p
                P_tau0_div_p = np.zeros((2,N2t,N1t), dtype=np.float64)
                P_tau0_div_p = pd0.proj_neumann_dct(tau0_div_p, P_tau0_div_p, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
            if j == num_without_dd_short - 1:
                tau_without_dd_short = np.array(P_tau0_div_p)
                if DO_TEST_D == False: break
            if j == num_without_dd_long - 1:
                tau_without_dd_long = np.array(P_tau0_div_p)
        


    #3) Determine tau_without_dd_tvsdpyx with tvsd.py
    if (rank == 0):
        tfs_conf= {}
        tfs_conf["record_steps"] = 0
        tfs_conf["print_details"] = 0
        tfs_conf["h"] = h
        tfs_conf["k"] = k
        tfs_conf["delta"] = delta
        tfs_conf["projection_mode"] = "neumann"     #always in our setting
        tfs_conf["stop_criteria"] = "none"
        tfs_conf["max_steps"] = num_without_dd_short
        tfs_conf["energy_thresh_diff"] = 0.0001     #dummy
        tau_without_dd_tvsdpyx, _, tfs_energy = tv_stokes_dual_tfs_only(d0, tfs_conf, "")

    #4) Determine tau_dd_glob with a global DD-algorithm
    #prepare DD-object
    if rank == 0:
        dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
        dd_ext = dd.duplicate_with_additional_row_and_col()
    else:
        dd, dd_ext = None, None
    #perform algorithm (on 1 Thread)
    if (rank == 0):
        theta = dd_ext.create_theta_glob()
        #initialization
        p = np.zeros((2, 2, N2t, N1t), dtype=np.float64)
        v = {}
        for m1 in range(M1):
            for m2 in range(M2):
                v[(m2,m1)] = np.zeros((2, 2, N2t, N1t), dtype=np.float64)
        #outer loop
        for j in range(num_outer_long):    
            for m1 in range(M1):
                for m2 in range(M2):
                    q0m = np.copy(p)
                    for c1 in range(2):
                        for c2 in range(2):
                            q0m[c1,c2,:,:] -= theta[(m2,m1)] * p[c1,c2,:,:]
                    #inner loop
                    for n in range(num_inner):
                        v_q0 = v[(m2,m1)] + q0m
                        div_v_q0 = np.zeros((2,N2t,N1t), dtype=np.float64)
                        pd0.divB(v_q0[0,0,:,:], v_q0[0,1,:,:], div_v_q0[0,:,:], h)
                        pd0.divB(v_q0[1,0,:,:], v_q0[1,1,:,:], div_v_q0[1,:,:], h)
                        div_v_q0_tau0 = div_v_q0 - tau0til
                        P_div_v_q0_tau0 = np.zeros((2,N2t,N1t), dtype=np.float64)
                        P_div_v_q0_tau0 = pd0.proj_neumann_dct( \
                                div_v_q0_tau0, P_div_v_q0_tau0, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
                        psi = np.zeros((2,2,N2t,N1t), dtype=np.float64)
                        pd0.gradF(P_div_v_q0_tau0[0,:,:], psi[0,0,:,:], psi[0,1,:,:], h)
                        pd0.gradF(P_div_v_q0_tau0[1,:,:], psi[1,0,:,:], psi[1,1,:,:], h)
                        num00 = theta[(m2,m1)] * v[(m2,m1)][0,0,:,:] + k * theta[(m2,m1)] * psi[0,0,:,:]
                        num01 = theta[(m2,m1)] * v[(m2,m1)][0,1,:,:] + k * theta[(m2,m1)] * psi[0,1,:,:]
                        num10 = theta[(m2,m1)] * v[(m2,m1)][1,0,:,:] + k * theta[(m2,m1)] * psi[1,0,:,:]
                        num11 = theta[(m2,m1)] * v[(m2,m1)][1,1,:,:] + k * theta[(m2,m1)] * psi[1,1,:,:]
                        abs_psi0 = np.sqrt(psi[0,0,:,:] ** 2 + psi[0,1,:,:] ** 2)
                        abs_psi1 = np.sqrt(psi[1,0,:,:] ** 2 + psi[1,1,:,:] ** 2)
                        den0 = theta[(m2,m1)] + k * abs_psi0
                        den1 = theta[(m2,m1)] + k * abs_psi1
                        #inner update
                        v[(m2,m1)][0,0,:,:] = np.divide(num00, den0, out=np.zeros_like(num00, dtype=np.float64), where=(den0!=0.0))
                        v[(m2,m1)][0,1,:,:] = np.divide(num01, den0, out=np.zeros_like(num01, dtype=np.float64), where=(den0!=0.0))
                        v[(m2,m1)][1,0,:,:] = np.divide(num10, den1, out=np.zeros_like(num10, dtype=np.float64), where=(den1!=0.0))
                        v[(m2,m1)][1,1,:,:] = np.divide(num11, den1, out=np.zeros_like(num11, dtype=np.float64), where=(den1!=0.0))
            #outer update
            sum_v = np.zeros((2, 2, N2t, N1t), dtype=np.float64)
            for m1 in range(M1):
                for m2 in range(M2):
                    sum_v += v[(m2,m1)]
            if j == 0:
                p = sum_v
            else:
                p = (1.0 - alpha) * p + alpha * sum_v
            #get tau_gt from p    
            if j == num_outer_long - 1 or j == num_outer_short - 1:
                div_p = np.zeros((2,N2t,N1t), dtype=np.float64)
                pd0.divB(p[0,0,:,:], p[0,1,:,:], div_p[0,:,:], h)
                pd0.divB(p[1,0,:,:], p[1,1,:,:], div_p[1,:,:], h)
                tau0_div_p = tau0 - delta * div_p
                P_tau0_div_p = np.zeros((2,N2t,N1t), dtype=np.float64)
                P_tau0_div_p = pd0.proj_neumann_dct(tau0_div_p, P_tau0_div_p, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
            if j == num_outer_short - 1:
                tau_dd_glob_short = np.array(P_tau0_div_p)
                if DO_TEST_D == False: break
            if j == num_outer_long- 1:
                tau_dd_glob_long = np.array(P_tau0_div_p)
            if j % 100 == 0 and j > 0:
                print("DD_glob outer it " + str(j))


    #5) Determine tau from tau0 with local DD-algorithm (which shall be tested)
    d0_arr, add_data = parallelize(d0, dd, comm, additional_data=[(N2, N1)])
    #create DDs for other ranks (not just 0)
    N2, N1 = add_data[0]
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
    dd_ext = dd.duplicate_with_additional_row_and_col()
    #Prepare config such that it accepts tau0 with our parameters
    dummy_rec = Recorder.dummy()
    dummy_par = {"h" : h, "tangent_field_smoothing" : {}}
    tvsd_obj = TvStokesDualDD(dd, dummy_par, dummy_rec, comm, True, False)
    tvsd_obj.tfs.k = k
    tvsd_obj.tfs.alpha = alpha
    tvsd_obj.tfs.delta = delta
    tvsd_obj.tfs.num_outer_it_max = num_outer_short
    tvsd_obj.tfs.num_inner_it_max = num_inner
    tvsd_obj.tfs.outer_stop_criteria = "none"
    tvsd_obj.tfs.inner_stop_criteria = "none"
    tvsd_obj.tfs.energy_thresh = 0.001               #dummy
    tvsd_obj.tfs.inner_cauchy_thresh = 0.00000001    #dummy
    tvsd_obj.tfs.outer_cauchy_thresh = 0.000000001   #dummy
    #run test
    comm.Barrier()
    tau_arr = tvsd_obj.run_tfs_only(d0_arr, return_tau_ext=True)
    comm.Barrier()
    #deparallelize
    tau_dd_loc = deparallelize([tau_arr], dd_ext, comm)[0]
    
    
    #6) Now do actual tests
    if (rank == 0):
        #A) whether the result of TRTangentFieldSmoothing.run actually delivers
        #   a valid tangent field with: div(tau) == 0
        div_tau_dd_loc = np.zeros((N2t,N1t), dtype=np.float64)
        pd0.divB(tau_dd_loc[0,:,:],tau_dd_loc[1,:,:],div_tau_dd_loc,h)
        try:
            np.testing.assert_array_almost_equal(div_tau_dd_loc, np.zeros((N2t,N1t)), 8)
            print("TEST test_tangent_field_smoothing: div_tau_dd_loc==0 passed.")
        except AssertionError as e:
            print("div_tau_dd_loc is not equal to 0!!!")
            print("Message: ", e)
            print("TEST test_tangent_field_smoothing FAILED. Rank 0")
            return
        #B) the result of TRTangentFieldSmoothing.run is similar to the 
        #   result of tfs without dd
        try:
            np.testing.assert_array_almost_equal(tau_dd_loc, tau_dd_glob_short, 8)
            print("TEST test_tangent_field_smoothing: tau_dd_loc==tau_dd_glob passed.")
        except AssertionError as e:
            print("tau_dd_loc is not equal to tau_dd_glob!!!")
            print("Message: ", e)
            print("TEST test_tangent_field_smoothing FAILED. Rank 0")
            return
        #C) the result of TRTangentFieldSmoothing.run is similar to the 
        #   result of tfs without dd
        try:
            np.testing.assert_array_almost_equal(tau_without_dd_tvsdpyx, tau_without_dd_short[:,:N2,:N1], 8)
            print("TEST test_tangent_field_smoothing: tau_without_dd==tau_without_dd_tvsdpyx passed.")
        except AssertionError as e:
            print("tau_dd_glob is not equal to tau_without_dd!!!")
            print("Message: ", e)
            print("TEST test_tangent_field_smoothing FAILED. Rank 0")
            return
        #D) the result of TRTangentFieldSmoothing.run is similar to the 
        #   result of tfs without dd
        try:
            if DO_TEST_D:
                #winnames = ['au_dd_glob_long[0]', 'au_dd_glob_long[1]', \
                #            'tau_without_dd_long[0]', 'tau_without_dd_long[1]']
                #winwid = 400
                #winhei = int(winwid * N1 / N2)
                #for winname in winnames:
                #    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
                #    cv2.resizeWindow(winname, winhei, winwid)
                #cv2.imshow('tau_dd_glob_long[0]', tau_dd_glob_long[0,:,:])
                #cv2.imshow('tau_dd_glob_long[1]', tau_dd_glob_long[1,:,:])
                #cv2.imshow('tau_without_dd_long[0]', tau_without_dd_long[0,:,:])
                #cv2.imshow('tau_without_dd_long[1]', tau_without_dd_long[1,:,:])
                #while 1:
                #    if cv2.waitKey(1) == 27:    #ESC
                #        break
                #cv2.destroyAllWindows()
                np.testing.assert_array_almost_equal(tau_dd_glob_long, tau_without_dd_long, 4)
                print("TEST test_tangent_field_smoothing: tau_dd_glob==tau_without_dd passed.")
        except AssertionError as e:
            print("tau_dd_glob is not equal to tau_without_dd!!!")
            print("Message: ", e)
            print("TEST test_tangent_field_smoothing FAILED. Rank 0")
            return
        print("TEST test_tangent_field_smoothing PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST test_tangent_field_smoothing routine finished. Rank " + str(rank))


def get_Ptf(img : np.ndarray, h : float = 1.0) -> np.ndarray:
    """
    Determine projected tangent field from an image
    """
    #Preparations
    N1 = img.shape[1]
    N2 = img.shape[0]
    std_shape = (N2+1, N1+1)
    Sigma1_sq = np.zeros(std_shape[1], img.dtype)
    Sigma2_sq = np.zeros(std_shape[0], img.dtype)
    pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
    pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
    tmp_mat1 = np.zeros(std_shape, img.dtype)
    tmp_mat2 = np.zeros(std_shape, img.dtype)

    #project initial images on subspace div(tau) = 0
    tau = np.zeros((2, std_shape[0], std_shape[1]), img.dtype)
    Ptau = np.zeros((2, std_shape[0], std_shape[1]), img.dtype)
    v = Ptau[0]
    u = Ptau[1]
    #inner part
    tau[0,1:-1,:-1] = (1. / h) * (-img[1:,:] + img[:-1,:])
    tau[1,:-1,1:-1] = (1. / h) * (img[:,1:] - img[:,:-1])
    #right/bottom boundary: mirror (=Neumann boundary)
    tau[0,1:-1,-1] = np.copy(tau[0,1:-1,-2])
    tau[1,-1,1:-1] = np.copy(tau[1,-2,1:-1])
    #project this
    Ptau = pd0.proj_neumann_dct(tau, Ptau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
    return np.array(Ptau)[:,:-1,:-1]


def test_image_reconstruction_v1(comm):
    #Test, whether 
    # A) the result of TRImageReconstruction.run_v1
    #    is equal to a global dd-ir1
    # B) the result of simple ir1 without dd from here is equal
    #    to the result of tvsd.py
    # C) the result of the global dd-ir1 
    #    is similar to a tfs without dd
    DO_TEST_C = False

    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST test_image_reconstruction_v1 START, Rank " + str(rank))
    M1 = 3
    M2 = 3
    overlap_x = 3
    overlap_y = 4
    num_without_dd_long = 20000     #Test C
    num_without_dd_short = 20       #Test B
    num_outer_long = 5000           #Test C
    num_outer_short = 10            #Test A
    num_inner = 10                  #Test A
    alpha = 0.25
    mu = 0.1
    eps = 0.001
    h = 1.0
    k = 0.125

    #1) Create d0 for test case
    if (rank == 0):
        #N2, N1 = 20, 21
        #d0 = np.random.random((N2,N1))
        test_image = "beach_nz_1_480.jpg"
        ground_truth = cv2.imread("../resources/" + test_image, cv2.IMREAD_GRAYSCALE)
        ground_truth = ground_truth.astype("float64") / 255
        gaussian = np.random.normal(loc=0, scale=math.sqrt(0.01), size=ground_truth.shape) 
        d0 = np.clip(ground_truth + gaussian, 0.0, 1.0, dtype=np.float64)
        N2, N1 = d0.shape
        tau = get_Ptf(ground_truth, h)
    if (rank != 0):
        tau = None
        N2, N1 = None, None

    
    #2) Determine tau_without_dd with classic tangent field smoothing without dd (on 1 thread)
    if (rank == 0):
        #prepare xi
        xi = np.zeros((2, N2, N1), dtype=np.float64)
        abs_eps_tau = np.sqrt(tau[0,:,:] ** 2 + tau[1,:,:] ** 2 + eps)
        xi[0,:,:] = tau[1,:,:] / abs_eps_tau
        xi[1,:,:] = -tau[0,:,:] / abs_eps_tau
        #prepare d0til
        d0til = (1.0 / mu) * d0
        #initialization
        r = np.zeros((2, N2, N1), dtype=np.float64)
        #outer loop
        for j in range(num_without_dd_long):
            r_xi = r + xi
            div_r_xi = np.zeros((N2,N1), dtype=np.float64)
            pd0.divB(r_xi[0,:,:], r_xi[1,:,:], div_r_xi, h)
            div_r_xi_d0 = div_r_xi - d0til
            rho = np.zeros((2,N2,N1), dtype=np.float64)
            pd0.gradF(div_r_xi_d0, rho[0,:,:], rho[1,:,:], h)
            num0 = r[0,:,:] + k * rho[0,:,:]
            num1 = r[1,:,:] + k * rho[1,:,:]
            abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
            den = k * abs_rho + 1.0
            #update
            r[0,:,:] = np.divide(num0, den, out=np.zeros_like(num0, dtype=np.float64), where=(den!=0.0))
            r[1,:,:] = np.divide(num1, den, out=np.zeros_like(num1, dtype=np.float64), where=(den!=0.0))
            if j % 100 == 0 and j > 0:
                print("Without DD it " + str(j))  
            #get d from r
            if j == num_without_dd_long - 1 or j == num_without_dd_short - 1:
                r_xi = r + xi
                div_r_xi = np.zeros((N2,N1), dtype=np.float64)
                pd0.divB(r_xi[0,:,:], r_xi[1,:,:], div_r_xi, h)
            if j == num_without_dd_short - 1:
                d_without_dd_short = np.array(d0 - mu * div_r_xi)
                if DO_TEST_C == False: break
            if j == num_without_dd_long - 1:
                d_without_dd_long = np.array(d0 - mu * div_r_xi)
        

    #3) Determine tau_without_dd_tvsdpyx with tvsd.py
    if (rank == 0):
        ir_conf = {}
        ir_conf["variant"] = 1
        ir_conf["record_steps"] = 0
        ir_conf["print_details"] = 0
        ir_conf["h"] = h
        ir_conf["k"] = k
        ir_conf["mu"] = mu
        ir_conf["eps"] = eps
        ir_conf["diff_mode"] = "neumann"
        ir_conf["stop_criteria"] = "none"
        ir_conf["max_steps"] = num_without_dd_short
        ir_conf["energy_thresh_diff"] = 0.0001     #dummy
        d_without_dd_tvsdpyx, _, ir_energy = tv_stokes_dual_ir_only(d0, tau, ir_conf)


    #4) Determine tau_dd_glob with a global DD-algorithm
    #prepare DD-object
    if rank == 0:
        dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
    else:
        dd = None
    #perform algorithm (on 1 Thread)
    if (rank == 0):
        theta = dd.create_theta_glob()
        #initialization
        r = np.zeros((2, N2, N1), dtype=np.float64)
        w = {}
        for m1 in range(M1):
            for m2 in range(M2):
                w[(m2,m1)] = np.zeros((2, N2, N1), dtype=np.float64)
        #outer loop
        for j in range(num_outer_long):    
            for m1 in range(M1):
                for m2 in range(M2):
                    t0m = np.copy(r)
                    for c in range(2):
                        t0m[c,:,:] -= theta[(m2,m1)] * r[c,:,:]
                    #inner loop
                    for n in range(num_inner):
                        w_t0_xi = w[(m2,m1)] + t0m + xi
                        div_w_t0_xi = np.zeros((N2,N1), dtype=np.float64)
                        pd0.divB(w_t0_xi[0,:,:], w_t0_xi[1,:,:], div_w_t0_xi, h)
                        div_w_t0_xi_d0 = div_w_t0_xi - d0til
                        rho = np.zeros((2,N2,N1), dtype=np.float64)
                        pd0.gradF(div_w_t0_xi_d0, rho[0,:,:], rho[1,:,:], h)
                        num0 = theta[(m2,m1)] * w[(m2,m1)][0,:,:] + k * theta[(m2,m1)] * rho[0,:,:]
                        num1 = theta[(m2,m1)] * w[(m2,m1)][1,:,:] + k * theta[(m2,m1)] * rho[1,:,:]
                        abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
                        den = theta[(m2,m1)] + k * abs_rho
                        #inner update
                        w[(m2,m1)][0,:,:] = np.divide(num0, den, out=np.zeros_like(num0, dtype=np.float64), where=(den!=0.0))
                        w[(m2,m1)][1,:,:] = np.divide(num1, den, out=np.zeros_like(num1, dtype=np.float64), where=(den!=0.0))
            #outer update
            sum_w = np.zeros((2, N2, N1), dtype=np.float64)
            for m1 in range(M1):
                for m2 in range(M2):
                    sum_w += w[(m2,m1)]
            if j == 0:
                r = sum_w
            else:
                r = (1.0 - alpha) * r + alpha * sum_w
            #get d from r    
            if j == num_outer_long - 1 or j == num_outer_short - 1:
                r_xi = r + xi
                div_r_xi = np.zeros((N2,N1), dtype=np.float64)
                pd0.divB(r_xi[0,:,:], r_xi[1,:,:], div_r_xi, h)
            if j == num_outer_short - 1:
                d_dd_glob_short = np.array(d0 - mu * div_r_xi)
                if DO_TEST_C == False: break
            if j == num_outer_long- 1:
                d_dd_glob_long = np.array(d0 - mu * div_r_xi)
            if j % 100 == 0 and j > 0:
                print("DD_glob outer it " + str(j))

    #5) Determine tau from tau0 with local DD-algorithm (which shall be tested)
    if (rank==0):
        d0_tau = np.zeros((3,N2,N1), dtype=np.float64)      #Stack arrays to parallelize them together
        d0_tau[0,:,:] = d0
        d0_tau[1:3,:,:] = tau
    else:
        d0_tau = None
    d0_tau_arr, add_data = parallelize(d0_tau, dd, comm, additional_data=[(N2, N1)])
    d0_arr = d0_tau_arr[0,:,:]
    tau_arr = d0_tau_arr[1:3,:,:]
    #create DDs for other ranks (not just 0)
    N2, N1 = add_data[0]
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
    #Prepare config such that it accepts tau0 with our parameters
    dummy_rec = Recorder.dummy()
    ir_par = {}
    ir_par["variant"] = 1
    ir_par["k"] = k
    ir_par["alpha"] = alpha
    ir_par["mu"] = mu
    ir_par["beta"] = 0.0       #dummy for version 1
    ir_par["num_outer_it_max"] = num_outer_short
    ir_par["num_inner_it_max"] = num_inner
    ir_par["outer_stop_criteria"] = "none"
    ir_par["inner_stop_criteria"] = "none"
    ir_par["energy_thresh"] = 0.001               #dummy
    ir_par["inner_cauchy_thresh"] = 0.00000001    #dummy
    ir_par["outer_cauchy_thresh"] = 0.000000001   #dummy
    par = {"ir_variant" : 1, "h" : h, "eps" : eps, "image_reconstruction" : ir_par}
    tvsd_obj = TvStokesDualDD(dd, par, dummy_rec, comm, False, True)
    #run test
    comm.Barrier()
    d_arr, _ = tvsd_obj.run_ir_only(tau_arr, d0_arr)
    comm.Barrier()
    #deparallelize
    d_dd_loc = deparallelize([d_arr], dd, comm)[0]
    
    
    #6) Now do actual tests
    if (rank == 0):
        #A) the result of TRImageReconstruction.run_v1
        #   is equal to a global dd-ir1
        try:
            np.testing.assert_array_almost_equal(d_dd_loc, d_dd_glob_short, 8)
            print("TEST test_image_reconstruction_v1: d_dd_loc==d_dd_glob passed.")
        except AssertionError as e:
            print("d_dd_loc is not equal to d_dd_glob_short!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v1 FAILED. Rank 0")
            return
        #B) the result of simple ir1 without dd from here is equal
        #   to the result of tvsd.pyx
        try:
            np.testing.assert_array_almost_equal(d_without_dd_tvsdpyx, d_without_dd_short, 8)
            print("TEST test_image_reconstruction_v1: d_without_dd==d_without_dd_tvsdpyx passed.")
        except AssertionError as e:
            print("tau_dd_glob is not equal to tau_without_dd!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v1 FAILED. Rank 0")
            return
        #C) the result of the global dd-ir1 
        #   is similar to a tfs without dd
        try:
            if DO_TEST_C:
                np.testing.assert_array_almost_equal(d_dd_glob_long, d_without_dd_long, 4)
                print("TEST test_image_reconstruction_v1: d_dd_glob==d_without_dd passed.")
        except AssertionError as e:
            print("d_dd_glob is not equal to d_without_dd!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v1 FAILED. Rank 0")
            return
        print("TEST test_image_reconstruction_v1 PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST test_image_reconstruction_v1 routine finished. Rank " + str(rank))



def test_find_backdiff_scalar_potential(comm):
    #Test, whether the result of 
    #  TRIntersectingDDAlgorithm.find_backdiff_scalar_potential
    #is the same as the result of
    #  proj_div_0.integrate_tangentfield
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST test_find_backdiff_scalar_potential START, Rank " + str(rank))

    N2 = 340
    N1 = 243
    M1 = 3
    M2 = 4
    overlap_x = 5
    overlap_y = 6
    h = 1.0

    #prepare DD-object
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)
    dd_ext = dd.duplicate_with_additional_row_and_col()

    if (rank == 0):
        tau_raw = np.random.random((2,N2+1,N1+1))
        #project to make sure that the test tangent field is divergence-free
        tau = np.zeros((2,N2+1,N1+1), dtype = np.float64)
        Sigma1_sq, Sigma2_sq = np.zeros((N1+1), dtype=np.float64), np.zeros((N2+1), dtype=np.float64)
        pd0.prepare_Sigma_sq_dct(Sigma1_sq, h)
        pd0.prepare_Sigma_sq_dct(Sigma2_sq, h)
        tmp_mat1, tmp_mat2 = np.zeros((N2+1,N1+1), dtype=np.float64), np.zeros((N2+1,N1+1), dtype=np.float64)
        tau = pd0.proj_neumann_dct(tau_raw, tau, Sigma1_sq, Sigma2_sq, tmp_mat1, tmp_mat2, h)
        tau = np.array(tau)
        normal_total = np.stack((tau[1,:,:], -tau[0,:,:]), axis=0)
        normal_arr = {}
        for m2 in range(M2):
            for m1 in range(M1):
                normal_arr[(m2,m1)] = normal_total[:, \
                    dd_ext.ubounds[m2]:dd_ext.bbounds[m2],dd_ext.lbounds[m1]:dd_ext.rbounds[m1]]

        #ground truth: use proj_div_0.integrate_tangentfield
        g_gt = np.zeros((N2, N1), dtype=normal_total.dtype)
        g_gt = pd0.integrate_tangentfield(tau, g_gt, h, -1)
        #Test if ground truth makes sense
        try:
            g_gt = np.array(g_gt)
            dx_g_gt = (1. / h) * (g_gt[:,1:] - g_gt[:,:-1])
            dy_g_gt = (1. / h) * (g_gt[1:,:] - g_gt[:-1,:])
            np.testing.assert_array_almost_equal(dx_g_gt, tau[1,:-1,1:-1], 8)
            np.testing.assert_array_almost_equal(dy_g_gt, -tau[0,1:-1,:-1], 8)
        except AssertionError as e:
            print("AssertionError: dx_g_gt != tau[1] or dy_g_gt != -tau[0]")
            print("Message: ", e)
            print("TEST test_find_backdiff_scalar_potential FAILED. Rank 0")
            return

        #Distribute image arrays to ranks
        #Create image dics for all ranks
        normal_all_ranks = []
        for r in range(num_ranks):
            normal_dic = {}
            for m2,m1 in dd_ext.relevant_domains[r]:
                normal_dic[(m2,m1)] = copy.deepcopy(normal_arr[(m2,m1)])
            normal = ParallelArray(normal_dic)
            normal_all_ranks.append(normal)
        normal = normal_all_ranks[0]
        #Distribute all data to other ranks
        for r in range(1, num_ranks):
            #Serialize data (necessary, since data structure is complicated)
            ser_data_to_r = pickle.dumps(normal_all_ranks[r])
            data_size = len(ser_data_to_r)
            #send data size
            comm.isend(data_size, dest=r, tag=100+r).wait()
            #Send actual data
            comm.Isend([ser_data_to_r, MPI.BYTE], dest=r, tag=200+r).Wait()
    # Receive data on other ranks
    if (rank != 0):
        #Receive data size
        data_length = comm.irecv(source=0, tag=100+rank).wait()
        #Allocate enough space
        ser_data_to_r = bytearray(data_length)
        #Receive actual data
        comm.Irecv([ser_data_to_r, MPI.BYTE], source=0, tag=200+rank).Wait()
        #Deserialize
        normal = pickle.loads(ser_data_to_r)

    #Apply function to be tested
    n1 = normal[0,:,:]
    n2 = normal[1,:,:]
    dummy_rec = Recorder.dummy()
    dummy_par = {}
    triddalg_obj = TRIntersectingDDAlgorithm(dd, dummy_par, dummy_rec, comm)
    triddalg_obj.h = h
    comm.Barrier()
    g = triddalg_obj.find_backdiff_scalar_potential(n1, n2)
    comm.Barrier()

    #Put splitted array together
    #Send parts from all ranks
    if (rank != 0):
        ser_data_from_r = pickle.dumps(g)
        data_size = len(ser_data_from_r)
        #send data size
        comm.isend(data_size, dest=0, tag=100+rank).wait()
        #Send actual data
        comm.Isend([ser_data_from_r, MPI.BYTE], dest=0, tag=200+rank).Wait()
    #Collect parts from all ranks in Rank 0
    if (rank == 0):
        #create list for all ranks
        g_all_ranks = [None for _ in range(num_ranks)]   
        g_all_ranks[0] = g
        #Receive parts from all ranks
        for r in range(1, num_ranks):
            #Receive data size
            data_length_r = comm.irecv(source=r, tag=100+r).wait()
            #Allocate enough space
            ser_data_from_r = bytearray(data_length_r)
            #Receive actual data
            comm.Irecv([ser_data_from_r, MPI.BYTE], source=r, tag=200+r).Wait()
            #Deserialize
            g_all_ranks[r] = pickle.loads(ser_data_from_r)
        #Now reconstruct complete tau
        g_total = np.zeros((N2,N1))
        for r in range(num_ranks):
            dd.init_domains(dd.relevant_domains[r])
            for m2, m1 in dd.relevant_domains[r]:
                g_total[dd.ubounds[m2]:dd.bbounds[m2],dd.lbounds[m1]:dd.rbounds[m1]] \
                    += g_all_ranks[r][(m2,m1)] * dd.theta_loc(m2, m1)
        #Now do actual test
        try:
            #print("g_gt (ground truth)", g_gt)
            #print("g_total (from parallel alg)", g_total)
            np.testing.assert_array_almost_equal(g_total, g_gt, 8)
        except AssertionError as e:
            print("AssertionError: g_total != g_gt")
            print("Message: ", e)
            print("TEST test_find_backdiff_scalar_potential FAILED. Rank 0")
            return

        print("TEST test_find_backdiff_scalar_potential PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST test_find_backdiff_scalar_potential routine finished. Rank " + str(rank))


def test_image_reconstruction_v2(comm):
    #Test, whether 
    # A) the result of TRImageReconstruction.run_v2
    #    is equal to a global dd-ir2
    # B) the result of simple ir2 without dd from here is equal
    #    to the result of tvsd.py
    # C) the result of the global dd-ir2 
    #    is similar to a tfs without dd
    DO_TEST_C = False

    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("TEST test_image_reconstruction_v2 START, Rank " + str(rank))
    M1 = 3
    M2 = 3
    overlap_x = 3
    overlap_y = 4
    num_without_dd_long = 20000     #Test C
    num_without_dd_short = 20       #Test B
    num_outer_long = 5000           #Test C
    num_outer_short = 10             #Test A
    num_inner = 10                  #Test A
    alpha = 0.25
    beta = 10.0
    h = 1.0
    k = 0.125

    #1) Create d0 for test case
    if (rank == 0):
        #N2, N1 = 20, 23
        #d0 = np.random.random((N2,N1))
        test_image = "beach_nz_1_480.jpg"
        ground_truth = cv2.imread("../resources/" + test_image, cv2.IMREAD_GRAYSCALE)
        ground_truth = ground_truth.astype("float64") / 255
        gaussian = np.random.normal(loc=0, scale=math.sqrt(0.01), size=ground_truth.shape) 
        d0 = np.clip(ground_truth + gaussian, 0.0, 1.0, dtype=np.float64)
        N2, N1 = d0.shape
        tau = get_Ptf(ground_truth, h)
    if (rank != 0):
        tau = None
        N2, N1 = None, None

    
    #2) Determine tau_without_dd with classic tangent field smoothing without dd (on 1 thread)
    if (rank == 0):
        #prepare g
        g = np.zeros((N2, N1), dtype=np.float64)
        g = pd0.integrate_tangentfield(tau, g, h, -1)
        #prepare u0til
        u0til = beta * (d0 - g)
        #initialization
        r = np.zeros((2, N2, N1), dtype=np.float64)
        #outer loop
        for j in range(num_without_dd_long):
            div_r = np.zeros((N2,N1), dtype=np.float64)
            pd0.divB(r[0,:,:], r[1,:,:], div_r, h)
            div_r_j0 = div_r - u0til
            rho = np.zeros((2,N2,N1), dtype=np.float64)
            pd0.gradF(div_r_j0, rho[0,:,:], rho[1,:,:], h)
            num0 = r[0,:,:] + k * rho[0,:,:]
            num1 = r[1,:,:] + k * rho[1,:,:]
            abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
            den = k * abs_rho + 1.0
            #update
            r[0,:,:] = np.divide(num0, den, out=np.zeros_like(num0, dtype=np.float64), where=(den!=0.0))
            r[1,:,:] = np.divide(num1, den, out=np.zeros_like(num1, dtype=np.float64), where=(den!=0.0))
            if j % 100 == 0 and j > 0:
                print("Without DD it " + str(j))  
            #get d from r
            if j == num_without_dd_long - 1 or j == num_without_dd_short - 1:
                div_r = np.zeros((N2,N1), dtype=np.float64)
                pd0.divB(r[0,:,:], r[1,:,:], div_r, h)
            if j == num_without_dd_short - 1:
                d_without_dd_short = np.array(d0 - (1./beta) * div_r)
                if DO_TEST_C == False: break
            if j == num_without_dd_long - 1:
                d_without_dd_long = np.array(d0 - (1./beta) * div_r)
        

    #3) Determine tau_without_dd_tvsdpyx with tvsd.py
    if (rank == 0):
        ir_conf = {}
        ir_conf["variant"] = 2
        ir_conf["record_steps"] = 0
        ir_conf["print_details"] = 0
        ir_conf["h"] = h
        ir_conf["k"] = k
        ir_conf["beta"] = beta
        ir_conf["diff_mode"] = "neumann"
        ir_conf["stop_criteria"] = "none"
        ir_conf["max_steps"] = num_without_dd_short
        ir_conf["energy_thresh_diff"] = 0.0001     #dummy
        d_without_dd_tvsdpyx, _, ir_energy = tv_stokes_dual_ir_only(d0, tau, ir_conf)


    #4) Determine tau_dd_glob with a global DD-algorithm
    #prepare DD-object
    if rank == 0:
        dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
    else:
        dd = None
    #perform algorithm (on 1 Thread)
    if (rank == 0):
        theta = dd.create_theta_glob()
        #initialization
        r = np.zeros((2, N2, N1), dtype=np.float64)
        w = {}
        for m1 in range(M1):
            for m2 in range(M2):
                w[(m2,m1)] = np.zeros((2, N2, N1), dtype=np.float64)
        #outer loop
        for j in range(num_outer_long):    
            for m1 in range(M1):
                for m2 in range(M2):
                    t0m = np.copy(r)
                    for c in range(2):
                        t0m[c,:,:] -= theta[(m2,m1)] * r[c,:,:]
                    #inner loop
                    for n in range(num_inner):
                        w_t0 = w[(m2,m1)] + t0m
                        div_w_t0 = np.zeros((N2,N1), dtype=np.float64)
                        pd0.divB(w_t0[0,:,:], w_t0[1,:,:], div_w_t0, h)
                        div_w_t0_u0 = div_w_t0 - u0til
                        rho = np.zeros((2,N2,N1), dtype=np.float64)
                        pd0.gradF(div_w_t0_u0, rho[0,:,:], rho[1,:,:], h)
                        num0 = theta[(m2,m1)] * w[(m2,m1)][0,:,:] + k * theta[(m2,m1)] * rho[0,:,:]
                        num1 = theta[(m2,m1)] * w[(m2,m1)][1,:,:] + k * theta[(m2,m1)] * rho[1,:,:]
                        abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
                        den = theta[(m2,m1)] + k * abs_rho
                        #inner update
                        w[(m2,m1)][0,:,:] = np.divide(num0, den, out=np.zeros_like(num0, dtype=np.float64), where=(den!=0.0))
                        w[(m2,m1)][1,:,:] = np.divide(num1, den, out=np.zeros_like(num1, dtype=np.float64), where=(den!=0.0))
            #outer update
            sum_w = np.zeros((2, N2, N1), dtype=np.float64)
            for m1 in range(M1):
                for m2 in range(M2):
                    sum_w += w[(m2,m1)]
            if j == 0:
                r = sum_w
            else:
                r = (1.0 - alpha) * r + alpha * sum_w
            #get d from r    
            if j == num_outer_long - 1 or j == num_outer_short - 1:
                div_r = np.zeros((N2,N1), dtype=np.float64)
                pd0.divB(r[0,:,:], r[1,:,:], div_r, h)
            if j == num_outer_short - 1:
                d_dd_glob_short = np.array(d0 - (1./beta) * div_r)
                if DO_TEST_C == False: break
            if j == num_outer_long- 1:
                d_dd_glob_long = np.array(d0 - (1./beta) * div_r)
            if j % 100 == 0 and j > 0:
                print("DD_glob outer it " + str(j))

    #5) Determine tau from tau0 with local DD-algorithm (which shall be tested)
    if (rank==0):
        d0_tau = np.zeros((3,N2,N1), dtype=np.float64)      #Stack arrays to parallelize them together
        d0_tau[0,:,:] = d0
        d0_tau[1:3,:,:] = tau
    else:
        d0_tau = None
    d0_tau_arr, add_data = parallelize(d0_tau, dd, comm, additional_data=[(N2, N1)])
    d0_arr = d0_tau_arr[0,:,:]
    tau_arr = d0_tau_arr[1:3,:,:]
    #create DDs for other ranks (not just 0)
    N2, N1 = add_data[0]
    dd = ThreadedHUIP.create_esized_edistributed(N1, N2, M1, M2, overlap_x, overlap_y, num_ranks)    
    #Prepare config such that it accepts tau0 with our parameters
    dummy_rec = Recorder.dummy()
    ir_par = {}
    ir_par["variant"] = 2
    ir_par["k"] = k
    ir_par["alpha"] = alpha
    ir_par["mu"] = 0.0          #dummy for version 2
    ir_par["beta"] = beta
    ir_par["num_outer_it_max"] = num_outer_short
    ir_par["num_inner_it_max"] = num_inner
    ir_par["outer_stop_criteria"] = "none"
    ir_par["inner_stop_criteria"] = "none"
    ir_par["energy_thresh"] = 0.001               #dummy
    ir_par["inner_cauchy_thresh"] = 0.00000001    #dummy
    ir_par["outer_cauchy_thresh"] = 0.000000001   #dummy
    par = {"ir_variant" : 2, "h" : h, "eps" : 0.0, "image_reconstruction" : ir_par}
    tvsd_obj = TvStokesDualDD(dd, par, dummy_rec, comm, False, True)
    #run test
    comm.Barrier()
    d_arr, _ = tvsd_obj.run_ir_only(tau_arr, d0_arr)
    comm.Barrier()
    #deparallelize
    d_dd_loc = deparallelize([d_arr], dd, comm)[0]
    
    
    #6) Now do actual tests
    if (rank == 0):
        #A) the result of TRImageReconstruction.run_v2
        #   is equal to a global dd-ir2
        try:
            np.testing.assert_array_almost_equal(d_dd_loc, d_dd_glob_short, 8)
            print("TEST test_image_reconstruction_v2: d_dd_loc==d_dd_glob passed.")
        except AssertionError as e:
            print("d_dd_loc is not equal to d_dd_glob_short!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v2 FAILED. Rank 0")
            return
        #B) the result of simple ir2 without dd from here is equal
        #   to the result of tvsd.pyx
        try:
            np.testing.assert_array_almost_equal(d_without_dd_tvsdpyx, d_without_dd_short, 8)
            print("TEST test_image_reconstruction_v2: d_without_dd==d_without_dd_tvsdpyx passed.")
        except AssertionError as e:
            print("tau_dd_glob is not equal to tau_without_dd!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v2 FAILED. Rank 0")
            return
        #C) the result of the global dd-ir2
        #   is similar to a ir2 without dd
        try:
            if DO_TEST_C:
                np.testing.assert_array_almost_equal(d_dd_glob_long, d_without_dd_long, 4)
                print("TEST test_image_reconstruction_v2: d_dd_glob==d_without_dd passed.")
        except AssertionError as e:
            print("d_dd_glob is not equal to d_without_dd!!!")
            print("Message: ", e)
            print("TEST test_image_reconstruction_v2 FAILED. Rank 0")
            return
        print("TEST test_image_reconstruction_v2 PASSED SUCCESSFULLY. Rank 0")
    else:
        print("TEST test_image_reconstruction_v2 routine finished. Rank " + str(rank))




if __name__ == "__main__":
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    #print("PARALLEL TESTS for tv_stokes_dual (par_tests.py)\n"
    #        "Rank " + str(rank) + ": START")
    tridda_global_forw_diff_test(comm)
    comm.Barrier()
    tridda_global_back_diff_ext_test(comm)
    comm.Barrier()
    trtfs_proj_div_0_compl_test(comm)
    comm.Barrier()
    trtfs_compute_P_div_q0_test(comm)
    comm.Barrier()
    test_tangent_field_smoothing(comm)
    comm.Barrier()
    test_image_reconstruction_v1(comm)
    comm.Barrier()
    test_find_backdiff_scalar_potential(comm)
    comm.Barrier()
    test_image_reconstruction_v2(comm)
    #print("PARALLEL TESTS for tv_stokes_dual (par_tests.py)\n"
    #        "Rank " + str(rank) + ": FINISHED SUCCESSFULLY")
