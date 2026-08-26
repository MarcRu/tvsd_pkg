
import copy
import numpy as np

import tv_stokes_dual.proj_div_0_low_cap_base as pd0lcb
from tv_stokes_dual.partition import Partition, HaloPartition


"""
===================================================================
This Python-file provides classes to determine the partial
projection on the subspace with divB(...)=0 (and the partial
Moore-Penrose-Inverse of the discrete Laplace-operator with 
Neumann-boundary-conditions), which is needed to run the projection
parallely on low-RAM-processors.
To see a definition for divB, see proj_div_0.pyx.
The actual functionality is contained in the Cython-file
proj_div_0_low_cap_base.pyx.
This file is an interface to proj_div_0_low_cap_base.pyx.
===================================================================
"""




class InvLapl:
    """
    A class that provides methods for the partial Pseudo-Inverse of the
    Laplace operator (with Neumann boundary conditions).
    """

    def __init__(self, _partition : Partition, _h : np.float64=1.0):
        self.partition = _partition
        self.N1 = _partition.N1
        self.N2 = _partition.N2
        self.h = _h
        self.initialize()

    def initialize(self):
        self.Sigma1_sq = np.zeros(self.N1, dtype=np.float64)
        self.Sigma2_sq = np.zeros(self.N2, dtype=np.float64)
        pd0lcb.prepare_Sigma_sq(self.Sigma1_sq, self.h)
        pd0lcb.prepare_Sigma_sq(self.Sigma2_sq, self.h)
        self.M1 = self.partition.M1
        self.M2 = self.partition.M2
        ms2 = self.partition.max_sizes2
        ms1 = self.partition.max_sizes1
        self.tmp_mat1 = np.zeros((ms2, ms1), dtype=np.float64)
        self.tmp_mat2 = np.zeros((ms2, ms1), dtype=np.float64)

    def part_inv_sparse_general(self, mat : np.ndarray, \
                        k1 : np.uint64, k2 : np.uint64, \
                        m1 : np.uint64, m2 : np.uint64) -> np.ndarray:
        """
        Determines Pseudo-Inverse-Laplacian with Neumann-boundaries 
        of a N2xN1-mat, which is only non-zero in the domain (k2, k1).
        Only the domain (m2, m1) of the result is computed.
        The variable mat contains only the values in the domain (k2, k1)
        """
        assert mat.ndim == 2,  "mat must be 2-dimensional array"
        s2_in = self.partition.sizes2[k2]
        s1_in = self.partition.sizes1[k1]
        assert mat.shape[0] == s2_in and mat.shape[1] == s1_in, \
            "mat must have the size of the domain (k2, k1), which is passed to the Partition-Constructor"
        s2_out = self.partition.sizes2[m2]
        s1_out = self.partition.sizes1[m1]

        mat_inv = np.zeros((s2_out, s1_out), dtype = np.float64)

        mat_inv = pd0lcb.inv_laplacian_neum_sparse_general( \
            mat, mat_inv, self.tmp_mat1, self.tmp_mat2, \
            self.Sigma1_sq, self.Sigma2_sq, \
            self.partition.lbounds, self.partition.rbounds,
            self.partition.ubounds, self.partition.bbounds,
            self.partition.sizes1, self.partition.sizes2,
            self.partition.lbounds, self.partition.rbounds,
            self.partition.ubounds, self.partition.bbounds,
            self.partition.sizes1, self.partition.sizes2,
            self.N1, self.N2, self.M1, self.M2, k1, k2, m1, m2)
        
        mat_inv = np.asarray(mat_inv)

        return mat_inv

    def part_inv_sparse(self, mat : np.ndarray, m1 : np.uint64, m2 : np.uint64) -> np.ndarray:
        """
        Determines Pseudo-Inverse-Laplacian with Neumann-boundaries 
        of a N2xN1-mat, which is only non-zero in the domain (m2, m1).
        Only the domain (m2, m1) of the result is computed.
        The variable mat contains only the values in the domain (m2, m1) 
        """

        mat_inv = self.part_inv_sparse_general(mat, m1, m2, m1, m2)

        return mat_inv
    

class ProjectorDiv0(InvLapl):
    """
    A class that provides methods for the partial projection on the 
    subspace with divB(...) =0 using the Pseudo-Inverse of the
    Laplace operator (with Neumann boundary conditions).
    To see a definition for divB, see proj_div_0.pyx.
    """


    def __init__(self, _partition : HaloPartition, _h : np.float64=1.0):
        self.exit_partition = copy.deepcopy(_partition)
        super().__init__(_partition, _h)
        

    def constructor_2_partitions(_entry_partition : HaloPartition, \
                                 _exit_partition: HaloPartition, \
                                 _h : np.float64=1.0) -> 'ProjectorDiv0':
        """
        Needed instead of __init__, if the partition for the projected
        mat does not fit at all to the partition of the given mat.
        The number of columns and rows of rectangles (M2, M1) must be
        the same for both partitions though.
        """
        assert (_entry_partition.M1 == _exit_partition.M1)
        assert (_entry_partition.M2 == _exit_partition.M2)
        assert (_entry_partition.N1 == _exit_partition.N1)
        assert (_entry_partition.N2 == _exit_partition.N2)
        pdiv0 = ProjectorDiv0(_entry_partition, _h)
        pdiv0.exit_partition = _exit_partition
        # Allocate memory in the size of the halos of the biggest domains.
        mhs1 = max(pdiv0.partition.max_hsizes1, pdiv0.exit_partition.max_hsizes1)
        mhs2 = max(pdiv0.partition.max_hsizes2, pdiv0.exit_partition.max_hsizes2)
        pdiv0.tmp_mat1 = np.zeros((mhs2, mhs1), dtype=np.float64)
        pdiv0.tmp_mat2 = np.zeros((mhs2, mhs1), dtype=np.float64)
        pdiv0.tmp_mat3 = np.zeros((mhs2, mhs1), dtype=np.float64)
        pdiv0.tmp_mat4 = np.zeros((mhs2, mhs1), dtype=np.float64)
        return pdiv0

    
    def initialize(self):
        super().initialize()
        assert self.partition.halosize_l == 0 and \
                self.partition.halosize_r == 1 and \
                self.partition.halosize_u == 0 and \
                self.partition.halosize_b == 1, \
                "In the given Partition, it must be " + \
                "halosize_l == 0, halosize_r == 1," + \
                "halosize_u == 0 and halosize_b == 1."
        # Allocate memory in the size of the halos of the biggest domains.
        mhs1 = self.partition.max_hsizes1
        mhs2 = self.partition.max_hsizes2
        self.tmp_mat1 = np.zeros((mhs2, mhs1), dtype=np.float64)
        self.tmp_mat2 = np.zeros((mhs2, mhs1), dtype=np.float64)
        self.tmp_mat3 = np.zeros((mhs2, mhs1), dtype=np.float64)
        self.tmp_mat4 = np.zeros((mhs2, mhs1), dtype=np.float64)


    def proj_sparse_general(self, mat : np.ndarray, \
                            k1 : np.uint64, k2 : np.uint64, \
                            m1 : np.uint64, m2 : np.uint64) -> np.ndarray:
        """
        Projects 2xN2xN1-mat on subspace with divB(.)=0, 
        which is only non-zero in the domain (k2, k1).
        The discrete divB are the backward differences for the discrete R^{N2xN1}, 
        such that divB is conjugated with gradF on the discrete R^{N2xN1}.
        Only the domain (m2, m1) of the result is computed.
        The variable mat contains only the values in the domain (k2, k1).
        All other values around domain (k2, k1) are assumed to be zero.
        """
        assert mat.ndim == 3,  "mat must be 3-dimensional array"
        assert mat.shape[0] == 2, "the first dimension of mat must have size 2"
        s2_in = self.partition.sizes2[k2]
        s1_in = self.partition.sizes1[k1]
        assert mat.shape[1] == s2_in and mat.shape[2] == s1_in, \
            "mat must have shape (2, s2, s1) = " + str((2, s2_in, s1_in)) + \
            ", where (s2, s1) is the size of domain (k2, k1) = " + str((k2, k1)) + \
            ", which is passed to the Partition-Constructor. Instead:\n" + \
            "mat.shape = " + str(mat.shape)
        s2_out = self.exit_partition.sizes2[m2]
        s1_out = self.exit_partition.sizes1[m1]

        #prepare adapted partitions for block (k2,k1) and (m2,m1)
        #(these blocks are increased such that the halo is contained)
        apart_in = self.partition.modified_partition(k1, k2)
        apart_out = self.exit_partition.modified_partition(m1, m2)

        #extract bounds and sizes for C-style-routine in Cython
        labounds_in = apart_in.lbounds
        rabounds_in = apart_in.rbounds
        uabounds_in = apart_in.ubounds
        babounds_in = apart_in.bbounds
        asizes1_in = apart_in.sizes1
        asizes2_in = apart_in.sizes2
        labounds_out = apart_out.lbounds
        rabounds_out = apart_out.rbounds
        uabounds_out = apart_out.ubounds
        babounds_out = apart_out.bbounds
        asizes1_out = apart_out.sizes1
        asizes2_out = apart_out.sizes2

        #allocate output memory in correct size
        mat_proj = np.zeros((2, s2_out, s1_out), dtype = np.float64)

        mat_proj = pd0lcb.proj_div0_sparse_general( \
            mat, mat_proj, self.tmp_mat1, self.tmp_mat2, \
            self.tmp_mat3, self.tmp_mat4, self.Sigma1_sq, self.Sigma2_sq, \
            labounds_in,  rabounds_in, uabounds_in,  babounds_in, asizes1_in, asizes2_in, \
            labounds_out, rabounds_out, uabounds_out, babounds_out, asizes1_out,  asizes2_out, \
            s1_in, s2_in, s1_out, s2_out, self.N1, self.N2, \
            self.M1, self.M2, k1, k2, m1, m2, self.h)
        
        #transform cython-memory-view to python-numpy-array
        mat_proj = np.asarray(mat_proj)

        return mat_proj


    def proj_sparse(self, mat : np.ndarray, m1 : np.uint64, m2 : np.uint64) -> np.ndarray:
        """
        Projects 2xN2xN1-mat on subspace with divB(.)=0, 
        which is only non-zero in the domain (m2, m1).
        The discrete divB are the backward differences for the discrete R^{N2xN1}, 
        such that divB is conjugated with gradF on the discrete R^{N2xN1}.
        Only the domain (m2, m1) of the result is computed.
        The variable mat contains only the values in the domain (m2, m1) 
        All other values around domain (m2, m1) are assumed to be zero.
        """
        
        mat_proj = self.proj_sparse_general(mat, m1, m2, m1, m2)

        return mat_proj
