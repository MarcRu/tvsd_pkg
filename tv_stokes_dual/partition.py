import numpy as np
import numpy.matlib as nm
import copy
from typing import List, Tuple, Dict


class Partition:
    """
    Represents a (non-overlapping) subdivision of a domain of size N2xN1
    into M2xM1 rectangular pieces
    """
    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, _lbounds : np.ndarray, _ubounds : np.ndarray):
        assert _lbounds.ndim == 1 and _ubounds.ndim == 1, "_lbounds and _ubounds should be 1d-arrays"
        self.M1 = _lbounds.shape[0]
        self.M2 = _ubounds.shape[0]
        assert _lbounds[0] == 0
        assert _ubounds[0] == 0
        for m1 in range(self.M1 - 1):
            assert _lbounds[m1 + 1] > _lbounds[m1]
        for m2 in range(self.M2 - 1):
            assert _ubounds[m2 + 1] > _ubounds[m2]
        assert _lbounds[-1] < _N1 and _ubounds[-1] < _N2  
        self.N1 = _N1
        self.N2 = _N2
        self.lbounds = _lbounds.astype(np.uint32)
        self.ubounds = _ubounds.astype(np.uint32)
        self.sizes1 = np.concatenate((self.lbounds[1:] - self.lbounds[:-1], \
                    np.array([self.N1 - self.lbounds[-1]], dtype=np.uint32)))
        self.sizes2 = np.concatenate((self.ubounds[1:] - self.ubounds[:-1], \
                    np.array([self.N2 - self.ubounds[-1]], dtype=np.uint32)))
        self.rbounds = self.lbounds + self.sizes1
        self.bbounds = self.ubounds + self.sizes2
        self.max_sizes1 = np.max(self.sizes1)
        self.max_sizes2 = np.max(self.sizes2)


    def bounds_for_creation_with_fixed_block( \
            _N1 : np.uint64, _N2 : np.uint64, \
            _M1 : np.uint64, _M2 : np.uint64, \
            _m1 : np.uint64, _m2 : np.uint64, \
            _lbound : np.uint64, _rbound : np.uint64, \
            _ubound : np.uint64, _bbound : np.uint64) \
                -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates lbounds and ubounds for a Partition 
        with a fixed block (m2, m1).
        All other blocks around them are made as equal as possible.
        """

        lbounds = np.zeros((_M1), dtype=np.uint32)
        ubounds = np.zeros((_M2), dtype=np.uint32)
        assert _m1 >= 0 and _m1 < _M1, "m1 must be a value within 0,1,...,M1-1"
        assert _m2 >= 0 and _m2 < _M2, "m2 must be a value within 0,1,...,M2-1"
        assert _lbound >= 0 and _lbound < _N1
        assert _rbound > _lbound and _rbound <= _N1
        assert _ubound >= 0 and _ubound < _N2
        assert _bbound > _ubound and _bbound <= _N2

        lbounds[_m1] = _lbound
        ubounds[_m2] = _ubound
        
        #left side
        if(_m1 > 0):
            #make sure there is enough space
            assert int(_lbound / _m1) >= 1
            #distribute blocks equally
            for l1 in range(_m1):
                lbounds[l1] = int(_lbound * l1 / _m1)
        else:
            assert _lbound == 0

        #upper side
        if(_m2 > 0):
            #make sure there is enough space
            assert int(_ubound / _m2) >= 1
            #distribute blocks equally
            for l2 in range(_m2):
                ubounds[l2] = int(_ubound * l2 / _m2)
        else:
            assert _ubound == 0

        #right side
        if(_m1 < _M1 - 1):
            lbounds[_m1 + 1] = _rbound
            #make sure there is enough space
            assert int((_N1 - _rbound) / (_M1 - 1 - _m1)) >= 1
            #distribute blocks equally
            for l1 in range(_m1+1, _M1):
                lbounds[l1] = _rbound + int((l1 - 1 -_m1) * (_N1 - _rbound) / (_M1 - 1 - _m1))
        else:
            assert _m1 == _M1 - 1
            assert _rbound == _N1

        #bottom side
        if(_m2 < _M2 - 1):
            ubounds[_m2 + 1] = _bbound
            #make sure there is enough space
            assert int((_N2 - _bbound) / (_M2 - 1 - _m2)) >= 1
            #distribute blocks equally
            for l2 in range(_m2+1, _M2):
                ubounds[l2] = _bbound + int((l2 - 1 -_m2) * (_N2 - _bbound) / (_M2 - 1 - _m2))
        else:
            assert _m2 == _M2 - 1
            assert _bbound == _N2

        return lbounds, ubounds


    def create_with_fixed_block(_N1 : np.uint64, _N2 : np.uint64, \
                    _M1 : np.uint64, _M2 : np.uint64, \
                    _m1 : np.uint64, _m2 : np.uint64, \
                    _lbound : np.uint64, _rbound : np.uint64, \
                    _ubound : np.uint64, _bbound : np.uint64):
        """
        Creates Partition with a fixed block (m2, m1).
        All other blocks around them are made as equal as possible.
        """
        
        lbounds, ubounds = Partition.bounds_for_creation_with_fixed_block( \
            _N1, _N2, _M1, _M2, _m1, _m2, _lbound, _rbound, _ubound, _bbound)
        return Partition(_N1, _N2, lbounds, ubounds)


class HaloPartition(Partition):
    """
    Represents a subdivision of a domain of size N2xN1
    into M2xM1 rectangular pieces
    On top of this, every domain has a halo: a greater rectangle
    that contains it
    """
    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, _lbounds : np.ndarray, _ubounds : np.ndarray, \
                _halosize_l : np.uint32 = 0, _halosize_r : np.uint32 = 1, \
                _halosize_u : np.uint32 = 0, _halosize_b : np.uint32 = 1):
        super().__init__(_N1, _N2, _lbounds, _ubounds)
        self.halosize_l = _halosize_l
        self.halosize_r = _halosize_r
        self.halosize_u = _halosize_u
        self.halosize_b = _halosize_b
        min_sizes1 = np.min(self.sizes1)
        min_sizes2 = np.min(self.sizes2)
        assert min_sizes1 > self.halosize_l + self.halosize_r and \
            min_sizes2 > self.halosize_u + self.halosize_b, \
            "Every domain must have at least size halosize+1 in x- and y-direction."

        #define halo bounds
        if (self.M1 == 1):
            self.hlbounds = self.lbounds
            self.hrbounds = self.rbounds
        else:
            self.hlbounds = np.concatenate((np.array([0], dtype=np.uint32), \
                            self.lbounds[1:] - self.halosize_l))
            self.hrbounds = np.concatenate((self.rbounds[:-1] + self.halosize_r, \
                            np.array([self.N1], dtype=np.uint32)))
        if (self.M2 == 1):
            self.hubounds = self.ubounds
            self.hbbounds = self.bbounds
        else:
            self.hubounds = np.concatenate((np.array([0], dtype=np.uint32), \
                            self.ubounds[1:] - self.halosize_u))
            self.hbbounds = np.concatenate((self.bbounds[:-1] + self.halosize_b, \
                            np.array([self.N2], dtype=np.uint32)))
        self.hsizes1 = self.hrbounds - self.hlbounds
        self.hsizes2 = self.hbbounds - self.hubounds
        self.max_hsizes1 = np.max(self.hsizes1)
        self.max_hsizes2 = np.max(self.hsizes2)


    def modified_partition(self, m1 : np.uint64, m2 : np.uint64) -> Partition:
        """
        Creates a new Partition based on this one, but
        increases domain (m2, m1) such that includes the halo of it.
        """

        mod_part = Partition.create_with_fixed_block( \
                    self.N1, self.N2, self.M1, self.M2, m1, m2, \
                    self.hlbounds[m1], self.hrbounds[m1], \
                    self.hubounds[m2], self.hbbounds[m2])

        return mod_part
    
    def create_with_fixed_block(_N1 : np.uint64, _N2 : np.uint64, \
                    _M1 : np.uint64, _M2 : np.uint64, \
                    _m1 : np.uint64, _m2 : np.uint64, \
                    _lbound : np.uint64, _rbound : np.uint64, \
                    _ubound : np.uint64, _bbound : np.uint64) -> 'HaloPartition':
        """
        Creates HaloPartition with a fixed block (m2, m1).
        All other blocks around them are made as equal as possible.
        Furthermore halosize_l = 0, halosize_r = 1, 
        halosize_u = 0 and halosize_b = 1
        """
        
        lbounds, ubounds = Partition.bounds_for_creation_with_fixed_block( \
            _N1, _N2, _M1, _M2, _m1, _m2, _lbound, _rbound, _ubound, _bbound)
        
        return HaloPartition(_N1, _N2, lbounds, ubounds)
    


class IntersectingPartition:
    """
    Represents a subdivision of a domain of size N2xN1
    into M2xM1 OVERLAPPING rectangular pieces
    """
    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, \
                _lbounds : np.ndarray, _rbounds : np.ndarray, \
                _ubounds : np.ndarray, _bbounds : np.ndarray):
        assert _lbounds.ndim == 1 and _rbounds.ndim == 1 and \
                _ubounds.ndim == 1 and _bbounds.ndim == 1, \
                "_lbounds, _rbound, _ubounds and _bbounds should be 1d-arrays"
        assert _lbounds.shape[0] == _rbounds.shape[0], "_lbounds and _rbounds should be arrays of the same length"
        assert _ubounds.shape[0] == _bbounds.shape[0], "_ubounds and _bbounds should be arrays of the same length"
        assert _lbounds[0] == 0
        assert _ubounds[0] == 0
        assert _rbounds[-1] == _N1
        assert _bbounds[-1] == _N2
        self.lbounds = _lbounds.astype(np.uint32)
        self.ubounds = _ubounds.astype(np.uint32)
        self.M1 = self.lbounds.shape[0]
        self.M2 = self.ubounds.shape[0]
        for m1 in range(self.M1 - 1):
            assert _lbounds[m1 + 1] > _lbounds[m1]
            assert _rbounds[m1 + 1] > _rbounds[m1]
        for m2 in range(self.M2 - 1):
            assert _ubounds[m2 + 1] > _ubounds[m2]
            assert _bbounds[m2 + 1] > _bbounds[m2]
        assert _lbounds[-1] < _N1 and _ubounds[-1] < _N2
        self.N1 = _N1
        self.N2 = _N2
        assert _rbounds[0] > 0 and _bbounds[0] > 0 
        assert (_rbounds > _lbounds).all() and (_bbounds > _ubounds).all()
        assert (_rbounds[:-1] >= _lbounds[1:]).all() and (_bbounds[:-1] >= _ubounds[1:]).all()
        self.rbounds = _rbounds.astype(np.uint32)
        self.bbounds = _bbounds.astype(np.uint32)
        self.initialize()


    def create_equally_sized_parameters(_N1 : np.uint64, _N2 : np.uint64, \
                _M1 : np.uint64, _M2 : np.uint64, \
                _overlap_x : np.uint64, _overlap_y : np.uint64) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Help function to create IntersectingPartition
        with equally sized domains and a fixed number of overlaps.
        """
        lbounds = np.zeros((_M1), dtype=np.uint32)
        rbounds = np.zeros((_M1), dtype=np.uint32)
        ubounds = np.zeros((_M2), dtype=np.uint32)
        bbounds = np.zeros((_M2), dtype=np.uint32)
        rbounds[-1] = _N1
        bbounds[-1] = _N2
        assert _overlap_x * (_M1 + 1) <= _N1, "overlap_x too big"
        assert _overlap_y * (_M2 + 1) <= _N2, "overlap_y too big"
        inner_pixels_x = _N1 - _overlap_x * (_M1 + 1)
        inner_pixels_y = _N2 - _overlap_y * (_M2 + 1)
        for j in range(1,_M1):
            lbounds[j] = j * _overlap_x + int(inner_pixels_x * j / _M1)
            rbounds[j - 1] = lbounds[j] + _overlap_x
        for i in range(1,_M2):
            ubounds[i] = i * _overlap_y + int(inner_pixels_y * i / _M2)
            bbounds[i - 1] = ubounds[i] + _overlap_y
        return lbounds,rbounds,ubounds,bbounds


    def create_equally_sized(_N1 : np.uint64, _N2 : np.uint64, \
                _M1 : np.uint64, _M2 : np.uint64, \
                _overlap_x : np.uint64, _overlap_y : np.uint64) \
                    -> 'IntersectingPartition':
        """
        Alternative Constructor to create IntersectingPartition
        with equally sized domains and a fixed number of overlaps.
        """
        lbounds,rbounds,ubounds,bbounds = \
            IntersectingPartition.create_equally_sized_parameters( \
                _N1, _N2, _M1, _M2, _overlap_x, _overlap_y)
        
        return IntersectingPartition(_N1, _N2, \
                        lbounds, rbounds, ubounds, bbounds)


    def initialize(self):
        """
        Initialize further values.
        """
        self.sizes1 = self.rbounds - self.lbounds
        self.sizes2 = self.bbounds - self.ubounds
        #inner bounds
        self.ilbounds = np.concatenate((np.array([0], dtype=np.uint32), \
                                        self.rbounds[:-1]))
        self.iubounds = np.concatenate((np.array([0], dtype=np.uint32), \
                                        self.bbounds[:-1]))
        self.irbounds = np.concatenate((self.lbounds[1:], \
                                        np.array([self.N1], dtype=np.uint32)))
        self.ibbounds = np.concatenate((self.ubounds[1:], \
                                        np.array([self.N2], dtype=np.uint32)))
        self.isizes1 = self.irbounds - self.ilbounds
        self.isizes2 = self.ibbounds - self.iubounds
        #intersections
        self.interlb = self.lbounds[1:]
        self.interrb = self.rbounds[:-1]
        self.interub = self.ubounds[1:]
        self.interbb = self.bbounds[:-1]
        self.inters1 = self.interrb - self.interlb
        self.inters2 = self.interbb - self.interub
        ##initialize object that describes the surrounding intersections
        #self.init_surrounding_bounds()


    def intersection_iterator_objects(self, m1 : int, m2 : int, hs : int = 0, \
                        on_extended_domain : bool = False) -> List[Tuple[bool, int, int, slice, slice]]:
        """
        Returns iterator over all intersections:
        [left boundary, 
        right boundary, 
        upper boundary, 
        bottom boundary, 
        upper left corner, 
        upper right corner, 
        bottom left corner, 
        bottom right corner]
        Including the follwing information:
        (relevance_condition,   #Is this boundary even relevant for (m2, m1)?
        down/up,                #-1 for upper boundaries, 1 for lower boundaries, 0 else
        right/left,             #-1 for left boundaries, 1 for right boundaries, 0 else
        row slice,              #row slice for the corresponding boundary
        column slice)           #column slice for the corresponding boundary

        Parameters:
        - m1: coordinate of current domain
        - m2: coordinate of current domain
        - hs: Halosize. If hs > 0, then hs additional rows/columns will be added to each intersection for slices.
        - on_extended_domain: Only relevant, if hs > 0.
                If False, then slices will be determined with respect to unextended domain.
                If True, then slices will be determined with respect to extended domains by hs pixels.
        """
        assert hs >= 0

        #inner domain in local coordinates
        ilb_loc = self.ilbounds[m1].astype(np.int32) - self.lbounds[m1].astype(np.int32)
        irb_loc = self.irbounds[m1].astype(np.int32) - self.rbounds[m1].astype(np.int32)
        iub_loc = self.iubounds[m2].astype(np.int32) - self.ubounds[m2].astype(np.int32)
        ibb_loc = self.ibbounds[m2].astype(np.int32) - self.bbounds[m2].astype(np.int32)


        if hs == 0:     # simpler version (for better readability)
            iterator_objects = \
                [(m1 > 0,                              0, -1,  slice(None, None), slice(None, ilb_loc)), \
                (m1 < self.M1 - 1,                     0, 1,   slice(None, None), slice(irb_loc, None)), \
                (m2 > 0,                               -1, 0,   slice(None, iub_loc), slice(None, None)), \
                (m2 < self.M2 - 1,                     1, 0,  slice(ibb_loc, None), slice(None, None)), \
                (m1 > 0 and m2 > 0,                   -1, -1,  slice(None, iub_loc), slice(None, ilb_loc)), \
                (m1 < self.M1 - 1 and m2 > 0,         -1, 1,   slice(None, iub_loc), slice(irb_loc, None)), \
                (m1 > 0 and m2 < self.M2 - 1,          1, -1,  slice(ibb_loc, None), slice(None, ilb_loc)), \
                (m1 < self.M1 - 1 and m2 < self.M2-1,  1, 1,   slice(ibb_loc, None), slice(irb_loc, None))]

        elif hs > 0 and  on_extended_domain == False:
            iterator_objects = \
                [(m1 > 0,                              0, -1,  slice(None, None), slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1,                     0, 1,   slice(None, None), slice(irb_loc-hs, None)), \
                (m2 > 0,                               -1, 0,   slice(None, iub_loc+hs), slice(None, None)), \
                (m2 < self.M2 - 1,                     1, 0,  slice(ibb_loc-hs, None), slice(None, None)), \
                (m1 > 0 and m2 > 0,                   -1, -1,  slice(None, iub_loc+hs), slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1 and m2 > 0,         -1, 1,   slice(None, iub_loc+hs), slice(irb_loc-hs, None)), \
                (m1 > 0 and m2 < self.M2 - 1,          1, -1,  slice(ibb_loc-hs, None), slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1 and m2 < self.M2-1,  1, 1,   slice(ibb_loc-hs, None), slice(irb_loc-hs, None))]
        
        else:   #hs > 0 and  on_extended_domain == True
            if self.M2 == 1:
                y_limits_lr_inters = slice(None, None)
            elif m2 == 0:   #uppermost row of domains
                y_limits_lr_inters = slice(None, -hs)
            elif m2 == self.M2 - 1:  #bottommost row of domains
                y_limits_lr_inters = slice(hs, None)
            else:   #a middle domain row
                y_limits_lr_inters = slice(hs, -hs)
            
            if self.M1 == 1:
                x_limits_ub_inters = slice(None, None)
            elif m1 == 0:   #leftmost column of domains
                x_limits_ub_inters = slice(None, -hs)
            elif m1 == self.M1 - 1:  #rightmost column of domains
                x_limits_ub_inters = slice(hs, None)
            else:   #a middle domain column
                x_limits_ub_inters = slice(hs, -hs)

            iterator_objects = \
                [(m1 > 0,                              0, -1,  y_limits_lr_inters, slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1,                     0, 1,   y_limits_lr_inters, slice(irb_loc-hs, None)), \
                (m2 > 0,                               -1, 0,   slice(None, iub_loc+hs), x_limits_ub_inters), \
                (m2 < self.M2 - 1,                     1, 0,  slice(ibb_loc-hs, None), x_limits_ub_inters), \
                (m1 > 0 and m2 > 0,                   -1, -1,  slice(None, iub_loc+hs), slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1 and m2 > 0,         -1, 1,   slice(None, iub_loc+hs), slice(irb_loc-hs, None)), \
                (m1 > 0 and m2 < self.M2 - 1,          1, -1,  slice(ibb_loc-hs, None), slice(None, ilb_loc+hs)), \
                (m1 < self.M1 - 1 and m2 < self.M2-1,  1, 1,   slice(ibb_loc-hs, None), slice(irb_loc-hs, None))]
        


        return iterator_objects



    def get_partition_with_fixed_block(self, m1 : int, m2 : int) -> Partition:
        """
        Creates Partition with a fixed block (m2, m1).
        All other blocks around them are made as equal as possible.
        """
        lbounds = copy.copy(self.lbounds)
        ubounds = copy.copy(self.ubounds)
        lbounds[(m1+1):] = self.ilbounds[(m1+1):]
        ubounds[(m2+1):] = self.iubounds[(m2+1):]
        p = Partition(self.N1, self.N2, lbounds, ubounds)
        return p


    def get_halo_partition_with_fixed_block(self, m1 : int, m2 : int) -> HaloPartition:
        """
        Creates HaloPartition with a fixed block (m2, m1).
        All other blocks around them are made as equal as possible.
        Furthermore halosize_l = 0, halosize_r = 1, 
        halosize_u = 0 and halosize_b = 1
        """
        lbounds = copy.copy(self.lbounds)
        ubounds = copy.copy(self.ubounds)
        lbounds[(m1+1):] = self.ilbounds[(m1+1):]
        ubounds[(m2+1):] = self.iubounds[(m2+1):]
        hp = HaloPartition(self.N1, self.N2, lbounds, ubounds)
        return hp
    
    
    def get_halo_partition_with_fixed_extended_block(\
                    self, m1 : int, m2 : int, \
                    ext1r : int, ext2b : int) -> HaloPartition:
        """
        Creates HaloPartition with a fixed block (m2, m1).
        Adds ext1r columns on the right of block (m2, m1), if not right edge.
        Adds ext2b rows on the bottom of block (m2, m1), if not bottom edge.
        All other blocks around them are made as equal as possible.
        Furthermore halosize_l = 0, halosize_r = 1, 
        halosize_u = 0 and halosize_b = 1
        """
        lbounds = copy.copy(self.lbounds)
        ubounds = copy.copy(self.ubounds)
        lbounds[(m1+1):] = self.ilbounds[(m1+1):] + ext1r
        ubounds[(m2+1):] = self.iubounds[(m2+1):] + ext2b
        hp = HaloPartition(self.N1, self.N2, lbounds, ubounds)
        return hp




class UnityIntersectingPartition(IntersectingPartition):
    """
    Represents an IntersectingPartition.
    On top of that, this class represents the theta-weights
    for a partition of unity on this IntersectingPartition.
    It can save these theta-weights for an arbitrary list
    of domains. These must be determined by calling "init_domain"
    first.
    """

    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, \
                _lbounds : np.ndarray, _rbounds : np.ndarray, \
                _ubounds : np.ndarray, _bbounds : np.ndarray):
        super().__init__(_N1, _N2, _lbounds, _rbounds, _ubounds, _bbounds)

    def create_equally_sized(_N1 : np.uint64, _N2 : np.uint64, \
                _M1 : np.uint64, _M2 : np.uint64, \
                _overlap_x : np.uint64, _overlap_y : np.uint64):
        
        lbounds,rbounds,ubounds,bbounds = \
            IntersectingPartition.create_equally_sized_parameters( \
                _N1, _N2, _M1, _M2, _overlap_x, _overlap_y)
        
        return UnityIntersectingPartition(_N1, _N2, \
                        lbounds, rbounds, ubounds, bbounds)

    def initialize(self):
        super().initialize()
        self.theta_loc_dic = {}

    def create_theta_1d(overlap_length):
        """
        Define local pattern, in which theta is supposed to be created.
        Here linear.
        """
        theta_1d = np.linspace(0.0, 1.0, overlap_length + 2)[1:-1]
        
        return theta_1d


    def init_domains(self, domains: List[Tuple[int, int]]):
        """
        Initializes local theta-weights for partition of unity
        for the given domains. Uses local coordinates for the domains.
        THIS MUST BE CALLED BEFORE USING theta_loc, get_qj0
        OR theta_loc_dic!!!
        """
        for domain in domains:
            m2 = domain[0]
            m1 = domain[1]

            #fill theta initially with ones; inner part can stay like this
            theta = np.ones((self.sizes2[m2], self.sizes1[m1]), np.float64)

            #inner domain in local coordinates
            ilb_loc = self.ilbounds[m1] - self.lbounds[m1]
            irb_loc = self.irbounds[m1] - self.lbounds[m1]
            iub_loc = self.iubounds[m2] - self.ubounds[m2]
            ibb_loc = self.ibbounds[m2] - self.ubounds[m2]

            #left boundary intersection
            if (m1 > 0):
                theta_loc_l = UnityIntersectingPartition.create_theta_1d(self.inters1[m1 - 1])
                theta[iub_loc:ibb_loc, 0:ilb_loc] = \
                        nm.repmat(theta_loc_l, self.isizes2[m2], 1)

            #right boundary intersection in local domain coordinates
            if (m1 < self.M1 - 1):
                theta_loc_r = 1. - UnityIntersectingPartition.create_theta_1d(self.inters1[m1])
                theta[iub_loc:ibb_loc, irb_loc:] = \
                        nm.repmat(theta_loc_r, self.isizes2[m2], 1)

            #top boundary intersection in local domain coordinates
            if (m2 > 0):
                theta_loc_t = UnityIntersectingPartition.create_theta_1d(self.inters2[m2 - 1])[np.newaxis].T
                theta[0:iub_loc, ilb_loc:irb_loc] = \
                        nm.repmat(theta_loc_t, 1, self.isizes1[m1])

            #bottom boundary intersection in local domain coordinates
            if (m2 < self.M2 - 1):
                theta_loc_b = (1. - UnityIntersectingPartition.create_theta_1d(self.inters2[m2]))[np.newaxis].T
                theta[ibb_loc:, ilb_loc:irb_loc] = \
                        nm.repmat(theta_loc_b, 1, self.isizes1[m1])

            #top left boundary intersection in local domain coordinates
            if (m2 > 0 and m1 > 0):
                theta[0:iub_loc, 0:ilb_loc] = theta_loc_t * theta_loc_l

            #top right boundary intersection in local domain coordinates
            if (m2 > 0 and m1 < self.M1 - 1):
                theta[0:iub_loc, irb_loc:] = theta_loc_t * theta_loc_r

            #bottom left boundary intersection in local domain coordinates
            if (m2 < self.M2 - 1 and m1 > 0):
                theta[ibb_loc:, 0:ilb_loc] = theta_loc_b * theta_loc_l

            #bottom right boundary intersection in local domain coordinates
            if (m2 < self.M2 - 1 and m1 < self.M1 - 1):
                theta[ibb_loc:, irb_loc:] = theta_loc_b * theta_loc_r

            self.theta_loc_dic[(m2,m1)] = theta


    def reset_domains(self):
        """
        Delete all saved theta-weights.
        """
        self.theta_loc_dic = {}


    def theta_loc(self, m2 : np.uint64, m1 : np.uint64) -> np.ndarray:
        """
        Returns theta-weights on domain (m2,m1) in local domain coordinates.
        """
        
        if (m2,m1) in self.theta_loc_dic:
            return self.theta_loc_dic[(m2,m1)]
        else:
            raise Exception("UnityIntersectingPartition.theta_loc: The theta-weights for domain (m2,m1)=" + str((m2,m1)) + " have not been initialized. Call UnityIntersectingPartition.init_domains to load them.")

    def get_qj0(self, p : np.ndarray, m2 : np.uint64, m1 : np.uint64) -> np.ndarray:
        """
        This function is required in DD-Tangent-Field-Smoothing.
        Returns qj0 on domain (m2,m1) from p (dualized local tangent field)
        in local domain coordinates.
        """
        
        if (m2,m1) in self.theta_loc_dic:
            assert p.shape == (2, 2, self.sizes2[m2], self.sizes1[m1])
            qj0 = np.zeros(p.shape, p.dtype)
            
            #sum of all theta except theta of this domain (m2,m1)
            theta_conj = 1. - self.theta_loc_dic[(m2,m1)]

            qj0[0][0][:,:] = theta_conj * p[0][0]
            qj0[0][1][:,:] = theta_conj * p[0][1]
            qj0[1][0][:,:] = theta_conj * p[1][0]
            qj0[1][1][:,:] = theta_conj * p[1][1]
            return qj0
        else:
            raise Exception("UnityIntersectingPartition.get_qj0: The theta-weights for domain (m2,m1)=" + str((m2,m1)) + " have not been initialized.  Call UnityIntersectingPartition.init_domains to load them.")


    def create_theta_glob(self) -> Dict[Tuple[int,int], np.ndarray]:
        """
        Returns theta-weights for all domains (m2,m1) in global
        domain coordinates.
        """
        domain_list = []
        for m1 in range(self.M1):
            for m2 in range(self.M2):
                domain_list.append((m2,m1))
        self.init_domains(domain_list)
        theta_glob = {}
        for m1 in range(self.M1):
            for m2 in range(self.M2):
                theta_glob_m = np.zeros((self.N2, self.N1), dtype=np.float64)
                theta_glob_m[self.ubounds[m2]:self.bbounds[m2], self.lbounds[m1]:self.rbounds[m1]] \
                        = self.theta_loc_dic[(m2,m1)]
                theta_glob[(m2,m1)] = theta_glob_m
        return theta_glob



class HaloUIP(UnityIntersectingPartition):
    """
    Represents an IntersectingPartition.
    Furthermore, this class represents the theta-weights
    for a partition of unity on this IntersectingPartition.
    On top of this, every domain has a halo: a greater rectangle
    that contains it.
    """

    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, \
                _lbounds : np.ndarray, _rbounds : np.ndarray, \
                _ubounds : np.ndarray, _bbounds : np.ndarray, \
                _halosize_l : np.uint32, _halosize_r : np.uint32, \
                _halosize_u : np.uint32, _halosize_b : np.uint32):
        super().__init__(_N1, _N2, _lbounds, _rbounds, _ubounds, _bbounds)
        self.halosize_l = _halosize_l
        self.halosize_r = _halosize_r
        self.halosize_u = _halosize_u
        self.halosize_b = _halosize_b
        min_sizes1 = np.min(self.sizes1)
        min_sizes2 = np.min(self.sizes2)
        assert min_sizes1 > self.halosize_l + self.halosize_r and \
            min_sizes2 > self.halosize_u + self.halosize_b, \
            "Every domain must have at least size halosize+1 in x- and y-direction."

        #define halo bounds
        if (self.M1 == 1):
            self.hlbounds = self.lbounds
            self.hrbounds = self.rbounds
        else:
            self.hlbounds = np.concatenate((np.array([0], dtype=np.uint32), \
                            self.lbounds[1:] - self.halosize_l))
            self.hrbounds = np.concatenate((self.rbounds[:-1] + self.halosize_r, \
                            np.array([self.N1], dtype=np.uint32)))
        if (self.M2 == 1):
            self.hubounds = self.ubounds
            self.hbbounds = self.bbounds
        else:
            self.hubounds = np.concatenate((np.array([0], dtype=np.uint32), \
                            self.ubounds[1:] - self.halosize_u))
            self.hbbounds = np.concatenate((self.bbounds[:-1] + self.halosize_b, \
                            np.array([self.N2], dtype=np.uint32)))
        self.hsizes1 = self.hrbounds - self.hlbounds
        self.hsizes2 = self.hbbounds - self.hubounds
        self.max_hsizes1 = np.max(self.hsizes1)
        self.max_hsizes2 = np.max(self.hsizes2)


    def create_equally_sized(_N1 : np.uint64, _N2 : np.uint64, \
                _M1 : np.uint64, _M2 : np.uint64, \
                _overlap_x : np.uint64, _overlap_y : np.uint64, \
                _halosize_l : np.uint32 = 0, _halosize_r : np.uint32 = 1, \
                _halosize_u : np.uint32 = 0, _halosize_b : np.uint32 = 1) -> 'HaloUIP':
        
        lbounds, rbounds, ubounds, bbounds = \
            IntersectingPartition.create_equally_sized_parameters( \
                _N1, _N2, _M1, _M2, _overlap_x, _overlap_y)
        
        return HaloUIP(_N1, _N2, lbounds, rbounds, ubounds, bbounds, \
                    _halosize_l, _halosize_r, _halosize_u, _halosize_b)


    def modified_partition(self, m1 : np.uint64, m2 : np.uint64) -> Partition:
        """
        Creates a new Partition based on this one, but
        increases domain (m2, m1) such that includes the halo of it.
        The output partition is non-intersecting!
        """

        mod_part = Partition.create_with_fixed_block( \
                    self.N1, self.N2, self.M1, self.M2, m1, m2, \
                    self.hlbounds[m1], self.hrbounds[m1], \
                    self.hubounds[m2], self.hbbounds[m2])
        
        return mod_part


class ThreadedHUIP(HaloUIP):
    """
    Represents an IntersectingPartition.
    Additionally, this class represents the theta-weights
    for a partition of unity on this IntersectingPartition.
    Furthermore, every domain has a halo: a greater rectangle
    that contains it.
    On top of this, every domain has a thread-number, that is
    responsible for the domain.
    """

    def __init__(self, _N1 : np.uint64, _N2 : np.uint64, \
                _lbounds : np.ndarray, _rbounds : np.ndarray, \
                _ubounds : np.ndarray, _bbounds : np.ndarray, \
                _halosize_l : np.uint32, _halosize_r : np.uint32, \
                _halosize_u : np.uint32, _halosize_b : np.uint32, \
                _thread_map : np.ndarray, _num_threads : np.uint32):

        super().__init__(_N1, _N2, _lbounds, _rbounds, _ubounds, _bbounds, \
                         _halosize_l, _halosize_r, _halosize_u, _halosize_b)
        assert _num_threads > 0
        assert _thread_map.ndim == 2
        assert _thread_map.shape == (self.M2, self.M1)
        _thread_map = np.int32(np.floor(_thread_map))
        assert (_thread_map < _num_threads).all()
        self.num_threads = _num_threads
        self.thread_map = _thread_map.astype(np.uint32)
        self.initialize_domain_lists()


    def initialize_domain_lists(self):
        #create different help objects
        self.relevant_domains = [[] for _ in range(self.num_threads)]
        self.thread_l = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_r = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_u = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_b = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_ul = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_ur = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_bl = np.zeros((self.M2, self.M1), dtype=np.uint32)
        self.thread_br = np.zeros((self.M2, self.M1), dtype=np.uint32)
        for m2 in range(self.M2):
            for m1 in range(self.M1):
                rank = self.thread_map[m2,m1]

                #create list that saves all relevant domains for a rank
                self.relevant_domains[rank].append((m2,m1))
                
                #create maps to get threads in neighboring domains
                #  write -1 if it does not exist
                if (m1 == 0):   #left border
                    self.thread_l[m2, m1] = -1
                else:
                    self.thread_l[m2, m1] = self.thread_map[m2, m1 - 1]
                if (m2 == 0):   #upper border
                    self.thread_u[m2, m1] = -1
                else: 
                    self.thread_u[m2, m1] = self.thread_map[m2 - 1, m1]
                if (m1 == self.M1 - 1):   #right border
                    self.thread_r[m2, m1] = -1
                else:
                    self.thread_r[m2, m1] = self.thread_map[m2, m1 + 1]
                if (m2 == self.M2 - 1):   #bottom border
                    self.thread_b[m2, m1] = -1
                else: 
                    self.thread_b[m2, m1] = self.thread_map[m2 + 1, m1]
                if (m1 == 0 or m2 == 0):   #upper left corner
                    self.thread_ul[m2, m1] = -1
                else:
                    self.thread_ul[m2, m1] = self.thread_map[m2 - 1, m1 - 1]
                if (m1 == self.M1 - 1 or m2 == 0):   #upper right corner
                    self.thread_ur[m2, m1] = -1
                else:
                    self.thread_ur[m2, m1] = self.thread_map[m2 - 1, m1 + 1]
                if (m1 == 0 or m2 == self.M2 - 1):   #bottom left corner
                    self.thread_bl[m2, m1] = -1
                else:
                    self.thread_bl[m2, m1] = self.thread_map[m2 + 1, m1 - 1]
                if (m1 == self.M1 - 1 or m2 == self.M2 - 1):   #bottom right corner
                    self.thread_br[m2, m1] = -1
                else:
                    self.thread_br[m2, m1] = self.thread_map[m2 + 1, m1 + 1]
                    
                    

    def equal_thread_map(M1 : np.uint64, M2 : np.uint64, \
                                num_threads : np.uint32) -> np.ndarray:
        """
        Distribute threads equally on domains
        """
        thread_map = np.zeros((M2, M1), dtype=np.uint32)
        for i in range(0, M2):
            for j in range(0, M1):
                thread_map[i,j] = ((i * M1 + j) + 1) % num_threads

        return thread_map
    

    def create_esized_edistributed(_N1 : np.uint64, _N2 : np.uint64, \
                _M1 : np.uint64, _M2 : np.uint64, \
                _overlap_x : np.uint64, _overlap_y : np.uint64,
                _num_threads : np.uint32) -> 'ThreadedHUIP':
        """
        Create equally sized equally distributed 
        Threaded Halo Unity Intersecting Partition.
        Creates ThreadedHUIP-object with equally sized domains 
        (or almost equally sized if not possible),
        given overlaps in x- and y-directions and a halo with size 1
        on the right and bottom ends of the domains.
        It equally distributes the threads on the domains
        """


        lbounds, rbounds, ubounds, bbounds = \
            IntersectingPartition.create_equally_sized_parameters( \
                _N1, _N2, _M1, _M2, _overlap_x, _overlap_y)
        
        halosize_l = 0
        halosize_r = 1
        halosize_u = 0
        halosize_b = 1

        thread_map = ThreadedHUIP.equal_thread_map(_M1, _M2, _num_threads)

        return ThreadedHUIP(_N1, _N2, lbounds, rbounds, ubounds, bbounds, \
                    halosize_l, halosize_r, halosize_u, halosize_b, \
                    thread_map, _num_threads)
    
    def duplicate_with_additional_row_and_col(self) -> 'ThreadedHUIP':
        """
        Creates new ThreadedHUIP from this one, but adds 
        a new line to the bottom-most domains and a new column
        to the right-most domains
        """
        N1_new = self.N1 + 1
        N2_new = self.N2 + 1
        lbounds_new = copy.copy(self.lbounds)
        ubounds_new = copy.copy(self.ubounds)
        rbounds_new = copy.copy(self.rbounds)
        bbounds_new = copy.copy(self.bbounds)
        rbounds_new[-1] = N1_new
        bbounds_new[-1] = N2_new

        return ThreadedHUIP(N1_new, N2_new, \
                    lbounds_new, rbounds_new, ubounds_new, bbounds_new, \
                    self.halosize_l,self.halosize_r, self.halosize_u, self.halosize_b, \
                    self.thread_map, self.num_threads)
    

    def get_relevant_domains(self, rank : np.uint32) -> List[Tuple[int, int]]:
        """
        For a certain rank, get a list of all relevant domains.
        """
        domains = self.relevant_domains[rank]
        return domains


    def load_relevant_theta(self, rank : np.uint32):
        """
        Load all relevant theta for the partition of unity of a certain rank.
        This can be called instead of UnityIntersectingPartition.init_domains,
        WHICH MUST BE CALLED BEFORE USING theta_loc, get_qj0 OR theta_loc_dic.
        """
        self.init_domains(self.relevant_domains[rank])