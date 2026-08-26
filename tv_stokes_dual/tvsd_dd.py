
import tv_stokes_dual.partition as pa
import tv_stokes_dual.proj_div_0_low_cap as pd0lc
import tv_stokes_dual.proj_div_0_low_cap_base as pd0lcb
from tv_stokes_dual.partition import ThreadedHUIP

import copy
import os
import math
import logging
from mpi4py import MPI
import numpy as np
import pickle
import cv2
from typing import List, Callable, Dict, Tuple



class ParallelArray:
    """
    Represents dictionary of numpy-arrays
    """

    def __init__(self):
        self.val = {}

    def __init__(self, img_data : Dict[any, np.ndarray]):
        self.val = img_data

    def __eq__(self, other : 'ParallelArray'):
        """Implement equality (self == other)."""
        if type(other) == ParallelArray:
            return self.val == other.val
        else:
            return False

    def __getitem__(self, key : any) -> any:
        """Implements the bracket operator for getting an item with tuple indexing.
        Usually, the to "key" corresponding array gets returned.
        If key is a slice or contains slices, then a new ParallelArray
        is returned with all entries of this ParallelArray sliced this way.
        """
        slicing = False
        if isinstance(key, slice):
            slicing = True
        if isinstance(key, tuple):
            if any(isinstance(item, slice) for item in key):
                slicing = True
        if slicing:
            sliced_dic = {}
            for k in self.val:
                sliced_dic[k] = self.val[k][key]
            return ParallelArray(sliced_dic)
        else:
            return self.val[key]

    def __setitem__(self, key : any, value : any):
        """Implements the bracket operator for setting an item with tuple indexing.
        Usually, the to "key" corresponding array will be set.
        If key is a slice or contains slices, then value must be a
        ParallelArray and the slice of each entry of this ParallelArray 
        gets the array-part from value.
        """
        slicing = False
        if isinstance(key, slice):
            slicing = True
        if isinstance(key, tuple):
            if any(isinstance(item, slice) for item in key):
                slicing = True
        if slicing:
            assert isinstance(value, ParallelArray)
            for k in self.val:
                assert self.val[k][key].shape == value.val[k].shape
                self.val[k][key] = value.val[k]
        else:
            self.val[key] = value


    def __add__(self, other):
        """Implement addition (self + other)."""
        sum = {}
        if isinstance(other, (int, float)):  # Check if other is a number
            # Add self.value + the other number
            for key in self.val:
                sum[key] = self.val[key] + other
            return ParallelArray(sum)
        elif isinstance(other, (ParallelArray)):  # Check if other is another ParallelArray
            for key in self.val:
                assert (key in other.val)
                assert (self.val[key].shape == other.val[key].shape)
                sum[key] = self.val[key] + other.val[key]
            return ParallelArray(sum)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'ParallelArray' and '{type(other).__name__}'")


    def __sub__(self, other):
        """Implement subtraction (self - other)."""
        diff = {}
        if isinstance(other, (int, float)):  # Check if other is a number
            # Add self.value + the other number
            for key in self.val:
                diff[key] = self.val[key] - other
            return ParallelArray(diff)
        elif isinstance(other, (ParallelArray)):  # Check if other is another ParallelArray
            for key in self.val:
                assert (key in other.val)
                assert (self.val[key].shape == other.val[key].shape)
                diff[key] = self.val[key] - other.val[key]
            return ParallelArray(diff)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'ParallelArray' and '{type(other).__name__}'")


    def __mul__(self, other):
        """Implement multiplication (self * other)."""
        prod = {}
        if isinstance(other, (int, float)):  # Check if other is a number
            # Multiply self.value by the number
            for key in self.val:
                prod[key] = self.val[key] * other
            return ParallelArray(prod)
        elif isinstance(other, (ParallelArray)):  # Check if other is another ParallelArray
            for key in self.val:
                assert (key in other.val)
                assert (self.val[key].shape == other.val[key].shape)
                prod[key] = self.val[key] * other.val[key]
            return ParallelArray(prod)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'ParallelArray' and '{type(other).__name__}'")

        

    def __truediv__(self, other):
        """Implement true division (self / other)."""
        quot = {}
        if isinstance(other, (int, float)):  # Check if other is a number
            # Divide self.value by the number
            for key in self.val:
                quot[key] = self.val[key] / other
            return ParallelArray(quot)
        elif isinstance(other, (ParallelArray)):  # Check if other is another ParallelArray
            for key in self.val:
                assert (key in other.val)
                assert (self.val[key].shape == other.val[key].shape)
                quot[key] = self.val[key] / other.val[key]
            return ParallelArray(quot)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'ParallelArray' and '{type(other).__name__}'")

    def __str__(self):
        """Implement string representation (str(self))."""
        return str(self.val)

    def __repr__(self):
        """Implement official string representation (repr(self))."""
        return repr(self.val)
    

    def stack(self, other : 'ParallelArray', axis : int = 0) \
            -> 'ParallelArray':
        """
        For each dictionary-component, self and other gets stacked
        """
        conc = {}
        for key in self.val:
            assert (key in other.val)
            assert (self.val[key].shape == other.val[key].shape)
            conc[key] = np.stack((self.val[key], other.val[key]), axis=axis)
        return ParallelArray(conc)
    
    def sqrt(arr : 'ParallelArray') -> 'ParallelArray':
        """
        Apply np.sqrt(...) on each dictionary entry
        """
        sqrt_res = {}
        for key in arr.val:
            sqrt_res[key] = np.sqrt(arr.val[key])
        return ParallelArray(sqrt_res)


    def clip_last_row_and_column(self, dd : ThreadedHUIP, dim : int = 2) -> 'ParallelArray':
        """
        Remove last row of bottom-most blocks and rightest row of
        right-most blocks
        """
        M1, M2 = dd.M1, dd.M2
        new_arr = copy.deepcopy(self)
        if (dim == 4):
            cut_slice_row = (slice(None), slice(None), slice(None, -1), slice(None))
            cut_slice_col = (slice(None), slice(None), slice(None), slice(None, -1))
            cut_slice_row_col = (slice(None), slice(None), slice(None, -1), slice(None, -1))
        elif(dim == 3):
            cut_slice_row = (slice(None), slice(None, -1), slice(None))
            cut_slice_col = (slice(None), slice(None), slice(None, -1))
            cut_slice_row_col = (slice(None), slice(None, -1), slice(None, -1))
        elif(dim == 2):
            cut_slice_row = (slice(None, -1), slice(None))
            cut_slice_col = (slice(None), slice(None, -1))
            cut_slice_row_col = (slice(None, -1), slice(None, -1))
        else:
            raise NotImplementedError("ParallelArray.clip_last_row_and_column: Not implemented for dim = " + str(dim))
        for m1 in range(M1-1):
            if (M2-1, m1) in self.val:
                new_arr.val[(M2-1, m1)] = self.val[(M2-1, m1)][cut_slice_row]
        for m2 in range(M2-1):
            if (m2, M1-1) in self.val:
                new_arr.val[(m2, M1-1)] = self.val[(m2, M1-1)][cut_slice_col]
        if (M2-1,M1-1) in self.val:
            new_arr.val[(M2-1,M1-1)] = self.val[(M2-1, M1-1)][cut_slice_row_col]
        return new_arr


def parallelize(img : np.ndarray, dd_obj : ThreadedHUIP, comm : MPI.Comm, \
                additional_data : any = None) -> Tuple[ParallelArray, any]:
    """
    Transforms simple image (np.ndarray) into a ParallelArray.
    Additional data for the other ranks can be transported.
    img:                image to parallelize
    dd_obj:             Domain Decomposition of type ThreadedHUIP. Contains domains,
                        overlaps, partition of unity and distribution on threads.
    mpi_comm:           MPI-object to communicate between threads.
    additional_data:    Additional data, that shall be distributed on other threads.
    Returns on each rank a tuple of the ParallelArray and the additional_data.
    """
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    if (rank == 0):
        if img.ndim < 2 or img.ndim > 4:
            raise Exception("parallelize not implemented for img.ndim==" + str(img.ndim))

        #Create image dics for all ranks
        img_dics = []
        for r in range(num_ranks):
            img_dic = {}
            for m2,m1 in dd_obj.relevant_domains[r]:
                if img.ndim == 2:
                    img_dic[(m2,m1)] = img[ \
                            dd_obj.ubounds[m2]:dd_obj.bbounds[m2], \
                            dd_obj.lbounds[m1]:dd_obj.rbounds[m1]]
                elif img.ndim == 3:
                    img_dic[(m2,m1)] = img[:, \
                            dd_obj.ubounds[m2]:dd_obj.bbounds[m2], \
                            dd_obj.lbounds[m1]:dd_obj.rbounds[m1]]
                elif img.ndim == 4:
                    img_dic[(m2,m1)] = img[:, :, \
                            dd_obj.ubounds[m2]:dd_obj.bbounds[m2], \
                            dd_obj.lbounds[m1]:dd_obj.rbounds[m1]]
            img_dics.append(img_dic)
        img_dic_r = img_dics[0]
        #Distribute all data to other ranks
        for r in range(1, num_ranks):
            #Serialize data (necessary, since data structure is complicated)
            ser_data_to_r = pickle.dumps((img_dics[r], additional_data))
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
        ser_data_from_r = bytearray(data_length)
        #Receive actual data
        comm.Irecv([ser_data_from_r, MPI.BYTE], source=0, tag=200+rank).Wait()
        #Deserialize
        data_from_r = pickle.loads(ser_data_from_r)
        img_dic_r, additional_data = data_from_r

    comm.Barrier()

    return ParallelArray(img_dic_r), additional_data


def deparallelize(parr_arrs : List[ParallelArray], \
                    dd : ThreadedHUIP, \
                    mpi_comm : MPI.Comm) -> List[np.ndarray]:
    """
    Transforms list of ParallelArray's into list of simple images,
    represented as numpy.ndarray's. Returns these images on Rank 0.
    At all other ranks, this functions returns a list filled with None.
    dd:         Domain Decomposition of type ThreadedHUIP. Contains domains,
                overlaps, partition of unity and distribution on threads.
    mpi_comm:   MPI-object to communicate between threads.
    """
    my_rank = mpi_comm.Get_rank()
    num_ranks = mpi_comm.Get_size()
    num_arrs = len(parr_arrs)
        
    mpi_comm.Barrier()

    #Send parts from all ranks
    if (my_rank != 0):
        ser_data_from_r = pickle.dumps(parr_arrs)
        data_size = len(ser_data_from_r)
        #send data size
        mpi_comm.isend(data_size, dest=0, tag=100+my_rank).wait()
        #Send actual data
        mpi_comm.Isend([ser_data_from_r, MPI.BYTE], dest=0, tag=200+my_rank).Wait()
        
    #Collect parts from all ranks in Rank 0
    if (my_rank == 0):

        #create list for all ranks
        all_array_lists = [None for _ in range(num_ranks)]   
        all_array_lists[0] = parr_arrs

        #Receive parts from all ranks
        for rank_send in range(1, num_ranks):
            #Receive data size
            data_length_r = mpi_comm.irecv(source=rank_send, tag=100+rank_send).wait()
            #Allocate enough space
            ser_data_from_r = bytearray(data_length_r)
            #Receive actual data
            mpi_comm.Irecv([ser_data_from_r, MPI.BYTE], source=rank_send, tag=200+rank_send).Wait()
            #Deserialize
            all_array_lists[rank_send] = pickle.loads(ser_data_from_r)

        #Now reconstruct images from all parts
        reconstructed_arrays = []
        for a in range(num_arrs):
            #extract dimension and sizes of dimensions
            for r in range(num_ranks):
                try:
                    ndim = list(all_array_lists[r][a].val.values())[0].ndim
                    dimsizes = list(all_array_lists[r][a].val.values())[0].shape
                    break
                except Exception as e:
                    if r == num_ranks - 1:
                        raise Exception("Unable to extract dimension and sizes")
                    continue
            if ndim == 2:
                rec_arr = np.zeros((dd.N2, dd.N1), dtype=np.float64)
            elif ndim == 3:
                nc = dimsizes[0]    #number of channels
                rec_arr = np.zeros((nc, dd.N2, dd.N1), dtype=np.float64)
            elif ndim == 4:
                nc1, nc2 = dimsizes[0], dimsizes[1]    #number of channels
                rec_arr = np.zeros((nc1, nc2, dd.N2, dd.N1), dtype=np.float64)
            else:
                raise Exception("Invalid dimension.")
            for rank in range(num_ranks):
                if a == 0: dd.init_domains(dd.relevant_domains[rank])
                for (m2, m1), loc_data in all_array_lists[rank][a].val.items():
                    assert loc_data.ndim == ndim
                    if ndim == 2:
                        rec_arr[dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                                += loc_data * dd.theta_loc(m2, m1)
                    if ndim == 3:
                        for c in range(nc):
                            rec_arr[c, dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                                    += loc_data[c,:,:] * dd.theta_loc(m2, m1)
                    if ndim == 4:
                        for c1 in range(nc1):
                            for c2 in range(nc2):
                                rec_arr[c1, c2, dd.ubounds[m2]:dd.bbounds[m2], dd.lbounds[m1]:dd.rbounds[m1]] \
                                        += loc_data[c1,c2,:,:] * dd.theta_loc(m2, m1)
            reconstructed_arrays.append(rec_arr)

        return reconstructed_arrays
    else:
        return [None for _ in range(num_arrs)]





class Recorder:

    def dummy() -> 'Recorder':
        """Dummy constructor"""
        rec = Recorder("", "", 0, 0, 0)
        rec.active = False

        #initialize logger
        rec.log_details = 0
        rec.log_file_name = ""
        rec.logger = None

        return rec


    def __init__(self, _recordings_path : str, _name_prefix : str, \
                 _print_details : int = 1, _log_details : int = 1, \
                _record_steps : int = 2, _do_set_up_logger : bool = True, \
                _log_file_name : str = None):
        #True constructor
        self.active = True
        self.recordings_path = _recordings_path
        self.print_details = _print_details
        self.record_steps = _record_steps
        self.name_prefix = _name_prefix

        #initialize logger
        self.log_details = _log_details
        if _log_file_name == None:
            self.log_file_name = os.path.join(self.recordings_path, "printout.log")
        else:
            self.log_file_name = _log_file_name
        # Configure logging with basicConfig
        if _do_set_up_logger:
            logging.basicConfig(filename=self.log_file_name, 
                                level=logging.INFO,
                                format='%(asctime)s - %(message)s')
        # Set up the logger
        self.logger = logging.getLogger(__name__)



    def save_recording(self, img : np.ndarray, name : str, print_saved : bool = False):
        if (self.active):
            if (print_saved):
                fn = os.path.join(self.recordings_path, name + ".png")
                print(fn)
            cv2.imwrite(fn, img)


    def save_recording_rescaled(self, img : np.ndarray, name : str, print_saved : bool = False):
        if (self.active == False): return
        assert img.ndim == 2
        imgmax = img.max()
        imgmin = img.min()
        if (img.max() - img.min() < 1.e-12):
            self.save_recording(img, name, print_saved)
        else:
            img_rescaled = (img - imgmin) * 255 / (imgmax - imgmin)
            self.save_recording(img_rescaled, name, print_saved)


    def is_round_number(num : int) -> bool:
        """
        return True for num=1,2,...,10,20,...,100,200,...
        else False
        """
        n = num
        count = 0
        while(n>0):
            count=count+1
            n=n//10
        #now check criteria by rounding
        if int(round(num, -count+1)) == num:
            return True
        else:
            return False   


    def do_record_step(self, depth : np.uint8, current_step : List[int]) -> bool:
        """
        Define when images should actually be saved
        It depends on depth in the algorithm
        and the parameter record_steps
        """

        if (self.active == False): return
        assert depth <= len(current_step)

        if depth == 0:
            if (self.record_steps == 0):
                return False
            else:
                return True
        
        if (self.record_steps < 2 * depth):
            do_record_step = False
        if (self.record_steps > 2 * depth):
            do_record_step = True
        if (self.record_steps == 2 * depth):
            step = current_step[depth - 1]
            #we are within the algorithm in iteration of depth depth
            #allow recording for a round step=1,2,...,10,20,...,100,200,...
            do_record_step = Recorder.is_round_number(step)        

        return do_record_step
    

    def do_print_saved(self, depth : np.uint8) -> bool:
        if (self.active == False): return
        do_print_saved = (self.print_details > depth)
        return do_print_saved
    

    def generate_full_names(self, names : List[str], current_step : List[int], depth : np.uint8) -> List[str]:
        if (self.active == False): return []
        assert depth <= len(current_step)
        full_names = []
        for name in names:
            full_name = self.name_prefix
            if depth > 0:
                if not(full_name == ""): full_name += "_"
                full_name += "step"
            for i in range(depth):
                if not(full_name == ""): full_name += "_"
                full_name += str(current_step[i])
            if not(full_name == ""): full_name += "_"
            full_name += name
            full_names.append(full_name)
        return full_names


    def record_step(self, raw_data : any, names : List[str], current_step : List[int], depth : np.uint8, \
                    operation : Callable[[any], List[np.ndarray]] = lambda x : [x], force : bool = False, \
                    force_rank : bool = False, rank : int = 0):
        """
        raw_data:       Data, which shall be saved as images
        operation:      Function that transforms raw_data into a list of images
        names:          List of names for the images to save
        current_steps:  Iteration indices from outer to inner index
        depth:          How deep are we in the iterations
                        outer iteration = 1, iteration within outer iteration = 2,...
        force:          if True, image gets saved anyway, independent of 
                        print_details-parameter of this class
        force_rank:     if True, then step gets recorded for all ranks

        """
        
        if (self.active == False): return
        do_record_step = self.do_record_step(depth, current_step)
        if (force == True):
            do_record_step = True
        
        if (do_record_step):
            #It is important that all ranks run through this. There might be Barriers
            images = operation(raw_data)
            #Now actual recording
            if ((rank == 0) or force_rank):
                do_print_saved = self.do_print_saved(depth)
                full_names = self.generate_full_names(names, current_step, depth)
                assert len(images) <= len(full_names)
                for i in range(len(images)):
                    self.save_recording_rescaled(images[i], full_names[i], do_print_saved)


    def log(self, text : str, priority : int):
        if (self.active == False): return
        if (priority < self.print_details):
            print(text)
        #write complete printout into file
        if (priority < self.log_details):
            self.logger.info(text)



class RecordedAlgorithm:
    
    def __init__(self, _alg_parameters : dict, _rec : Recorder, \
                 _start_step : List[int] = [0,0]):
        self.rec = _rec
        self.current_step = _start_step

        #initialize algorithm specific parameters (in derived class)
        self.initialize_parameters(_alg_parameters)


    def initialize_parameters(self, alg_parameters : dict):
        """
        Abstract method to read in parameters for the algorithm
        """
        pass

    def record_step(self, raw_data : any, names : List[str], current_step : List[int], depth : np.uint8, \
                    operation : Callable[[any], List[np.ndarray]] = lambda x : [x], \
                    force : bool = False, force_rank : bool = False, rank : int = 0):
        self.rec.record_step(raw_data, names, self.current_step, depth, operation, force, force_rank, rank)


    def log(self, text : str, priority : int):
        self.rec.log(text, priority)



class ThreadedRecordedAlgorithm(RecordedAlgorithm):
    
    def __init__(self, _alg_parameters : dict, _rec : Recorder, \
                  _mpi_comm : MPI.Comm, _start_step : List[int] = [0,0]):
        self.mpi_comm = _mpi_comm
        self.rank = _mpi_comm.Get_rank()
        self.num_ranks = _mpi_comm.Get_size()
        super().__init__(_alg_parameters, _rec, _start_step)


    def initialize_parameters(self, alg_parameters : dict):
        """
        Abstract method to read in parameters for the algorithm
        """
        pass


    def log(self, text : str, priority : int):
        text = "(Rank " + str(self.rank) + ") " + text
        super().log(text, priority)


    def record_step(self, raw_data : any, names : List[str], depth : np.uint8 = 1, \
                    operation : Callable[[any], List[np.ndarray]] = lambda x : [x], \
                    force_rank : bool = False, force : bool = False):
        """
        force_rank  if True, then step gets recorded for all ranks
        force       if True, then step gets recorded independently of record_steps-parameter
        """
        
        super().record_step(raw_data, names, self.current_step, depth, operation, force, force_rank, self.rank)


    def distribute_data(self, data_send : List[any]) -> List[any]:
        """
        Takes entries in data_sends and distributes them 
        to the other ranks (threads):
        data_send[0] goes to rank 0, data_send[1] goes to rank 1 and so on ...
        Returns list data_recv with messages from other ranks (threads):
        data_recv[0] comes from rank 0, data_recv[1] comes from rank 1, ...
        This will be handled with non-blocking techniques
        to avoid dead locks.
        """
   
        assert len(data_send) == self.num_ranks, \
            "List data_in must have same length as number of threads"

        # Make sure previous operations are finished to avoid tag conflicts
        self.mpi_comm.Barrier()

        # Non-blocking sends
        reqs1_send = []
        reqs2_send = []
        for rank_targ in range(self.num_ranks):
            if rank_targ != self.rank:
                ser_data_send = pickle.dumps(data_send[rank_targ])
                data_size = len(ser_data_send)
                req1 = self.mpi_comm.isend(data_size, dest=rank_targ, tag=10000 + 100 * self.rank + rank_targ)
                req2 = self.mpi_comm.Isend([ser_data_send, MPI.BYTE], dest=rank_targ, tag=20000 + 100 * self.rank + rank_targ)
                reqs1_send.append(req1)
                reqs2_send.append(req2)

        # Non-blocking receives for data lengths
        data_lengths = [None] * (self.num_ranks)
        for rank_orig in range(self.num_ranks):
            if rank_orig != self.rank:
                data_lengths[rank_orig] = self.mpi_comm.irecv(source=rank_orig, tag=10000 + 100 * rank_orig + self.rank).wait()

        ## DON'T wait for all data send operations to complete
        ## DANGER FOR DEADLOCK

        # Non-blocking receives for actual data
        data_recv = [None] * (self.num_ranks)
        for rank_orig in range(self.num_ranks):
            if rank_orig != self.rank:
                ser_data_recv = bytearray(data_lengths[rank_orig])
                self.mpi_comm.Irecv([ser_data_recv, MPI.BYTE], source=rank_orig, tag=20000 + 100 * rank_orig + self.rank).Wait()
                data_recv[rank_orig] = pickle.loads(ser_data_recv)
            else:   
                #rank_orig == self.rank: Just copy data, no mpi needed
                data_recv[self.rank] = copy.copy(data_send[self.rank])

        # Make sure all operations are finished to avoid data corruptions
        self.mpi_comm.Barrier()

        return data_recv
    

    def collect_data_from_rank(self, data_send : any, rank_send : int) -> any:
        """
        Takes data_send (which is only relevant for 
        self.rank == rank_send) and sends it to all ranks.
        """

        if (self.rank == rank_send):

            # Send data_send from rank_send to all other ranks
            for rank_targ in range(self.num_ranks):
                if rank_targ != self.rank:
                    #Serialize data (necessary, since data structure is complicated)
                    ser_data_send = pickle.dumps(data_send)
                    data_size = len(ser_data_send)
                    self.mpi_comm.isend(data_size, dest=rank_targ, tag=100+rank_targ).wait()
                    self.mpi_comm.Isend([ser_data_send, MPI.BYTE], dest=rank_targ, tag=200+rank_targ).Wait()

            # On this rank, return simply the send data
            return data_send
        
        else:
            # Receive data_send on other ranks
            data_length = self.mpi_comm.irecv(source=rank_send, tag=100+self.rank).wait()
            ser_data_recv = bytearray(data_length)
            self.mpi_comm.Irecv([ser_data_recv, MPI.BYTE], source=rank_send, tag=200+self.rank).Wait()
            data_recv = pickle.loads(ser_data_recv)
            return data_recv


    def collect_data_at_rank(self, data_send : any, rank_recv : int) -> any:
        """
        Takes data_send at all ranks and sends it to rank_recv.
        """

        if (self.rank != rank_recv):
            # Send data_send from this rank to rank_recv
            ser_data_send = pickle.dumps(data_send)
            data_size = len(ser_data_send)
            self.mpi_comm.isend(data_size, dest=rank_recv, tag=100+self.rank).wait()
            self.mpi_comm.Isend([ser_data_send, MPI.BYTE], dest=rank_recv, tag=200+self.rank).Wait()
            # On these ranks, nothing needs to be returned
            return None
        
        else:   #self.rank == rank_recv
            data_recv = []

            for rank_orig in range(self.num_ranks):
                if rank_orig == rank_recv:
                    data_recv.append(data_send)
                else:
                    # Receive data_send from other ranks
                    data_length = self.mpi_comm.irecv(source=rank_orig, tag=100+rank_orig).wait()
                    ser_data_recv = bytearray(data_length)
                    self.mpi_comm.Irecv([ser_data_recv, MPI.BYTE], source=rank_orig, tag=200+rank_orig).Wait()
                    data_recv.append(copy.deepcopy(pickle.loads(ser_data_recv)))
            
            return data_recv
    



class TRIntersectingDDAlgorithm(ThreadedRecordedAlgorithm):
    """
    Represents an abstract algorithm with intersecting Domain Decomposition
    for 2d-images. Contains a ThreadedHUIP-object 
    ("Threaded Halo Unity Intersecting Partition"), which manages
    the distribution of the threads on the domains and helps computing
    and saving the coordinates of the domains, their inner parts,
    their intersections, their halos (needed when divB is applied)
    and the partition of unity on them.
    """

    def __init__(self, _domain_decomposition : pa.ThreadedHUIP, \
                _alg_parameters : dict, _rec : Recorder, _mpi_comm : MPI.Comm):
        """
        Constructor with given domain decomposition
        """
        
        #The start step is initially [0,0]
        #   [outer it,  inner it]
        #   DD block row and column will be adapted depending on (thread-)rank
        start_step = [0,0]
        self.dd = _domain_decomposition
        self.N1 = self.dd.N1
        self.N2 = self.dd.N2
        super().__init__(_alg_parameters, _rec, _mpi_comm, start_step)
        self.relevant_domains = self.dd.relevant_domains[self.rank]



    def initialize_parameters(self, alg_parameters : dict):
        """
        Abstract method to read in parameters for the algorithm
        """
        pass
    

    def global_back_diff(self, u : List[ParallelArray], \
                        v : List[ParallelArray]) \
            -> Tuple[List[ParallelArray], List[ParallelArray]]:
        """
        Global differentiation for backward differences for ParallelArrays.
        Uses same backward differences as Chambolle.
                                [a b c d]
        Example:    u1 = v1 =   [e f g h],
                                [i j k l]
                                [m n o p]
                         [a b-a c-b -c]           [ a   b   c   d ]
        Then:    dx u1 = [e f-e g-f -g],  dy v1 = [e-a f-b g-c h-d]
                         [i j-i k-j -k]           [i-e j-f k-g l-h]
                         [m n-m o-n -o]           [-i  -j  -k  -l ]
        Takes all (locally represented) arrays u1, u2,... from u
        and applies backwards-derivative in x-direction on them.
        Takes all (locally represented) arrays v1, v2,... from v
        and applies backwards-derivative in y-direction on them.
        Returns ([dx u1, dx u2, ...], [dy v1, dy v2, ...])
        """
        if hasattr(self, 'h'):
            h = self.h
        else:
            h = 1
            self.log("WARNING: TRIntersectingDDAlgorithm.global_back_diff being used without parameter h set.", 0)

        dx_u = [ParallelArray({}) for _ in range(len(u))]
        dy_v = [ParallelArray({}) for _ in range(len(v))]
        u_rightest_noninters = [{} for _ in range(len(u))]
        v_bottom_noninters = [{} for _ in range(len(v))]
        data_to_send = [copy.deepcopy((u_rightest_noninters, v_bottom_noninters)) \
                        for targ_rank in range(self.num_ranks)]


        #Differentiate local image. 
        #  At first, treat boundaries as if this was the whole domain.
        #  Then correct rightest column / bottom row
        #  Send rightest non-intersecting column / bottom non-intersecting row 
        #   to neighboring domain
        for i, ui_pa in enumerate(u):
            ui = ui_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = self.dd.sizes2[m2], self.dd.sizes1[m1]
                dx_ui_m = np.zeros((sy, sx), dtype=np.float64)
                pd0lcb.diffB_x(ui[(m2,m1)], dx_ui_m, sx, sy, True, h)
                if (m1 < self.dd.M1 - 1):
                    dx_ui_m[:,-1] = ui[(m2,m1)][:,-1] - ui[(m2,m1)][:,-2]
                dx_u[i][(m2,m1)] = dx_ui_m
                if (m1 < self.dd.M1 - 1):
                    m2_targ, m1_targ = m2, m1 + 1
                    rank_targ = self.dd.thread_map[(m2_targ, m1_targ)]
                    rightest_noninters_index = \
                            self.dd.lbounds[m1_targ] - self.dd.lbounds[m1] - 1
                    #0 = u_rightest_noninters
                    data_to_send[rank_targ][0][i][(m2_targ,m1_targ)] = \
                            ui[(m2,m1)][:,rightest_noninters_index]

        for i, vi_pa in enumerate(v):
            vi = vi_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = self.dd.sizes2[m2], self.dd.sizes1[m1]
                dy_vi_m = np.zeros((sy, sx), dtype=np.float64)
                pd0lcb.diffB_y(vi[(m2,m1)], dy_vi_m, sx, sy, True, h)
                if (m2 < self.dd.M2 - 1):
                    dy_vi_m[-1,:] = vi[(m2,m1)][-1,:] - vi[(m2,m1)][-2,:]
                dy_v[i][(m2,m1)] = dy_vi_m
                if (m2 < self.dd.M2 - 1):
                    m2_targ, m1_targ = m2 + 1, m1
                    rank_targ = self.dd.thread_map[(m2_targ, m1_targ)]
                    bottom_noninters_index = \
                            self.dd.ubounds[m2_targ] - self.dd.ubounds[m2] - 1
                    #1 = v_bottom_noninters
                    data_to_send[rank_targ][1][i][(m2_targ,m1_targ)] = vi[(m2,m1)][bottom_noninters_index,:]

        #Now distribute data to other threads
        data_recv = self.distribute_data(data_to_send)

        #Unpack data_recv and rightest non-interecting column of neighboring left block
        for i, ui in enumerate(u):
            for m2, m1 in self.relevant_domains:
                if (m1 > 0):
                    m2_orig, m1_orig = m2, m1 - 1
                    rank_orig = self.dd.thread_map[(m2_orig, m1_orig)]
                    #0 = rightest non-intersecting column of neighboring left block
                    dx_u[i][(m2,m1)][:, 0] -= data_recv[rank_orig][0][i][(m2, m1)]

        #Unpack data_recv and bottom non-intersecting row of neighboring upper block
        for i, vi in enumerate(v):
            for m2, m1 in self.relevant_domains:
                if (m2 > 0):
                    m2_orig, m1_orig = m2 - 1, m1
                    rank_orig = self.dd.thread_map[(m2_orig, m1_orig)]
                    #1 = bottom non-intersecting row of neighboring upper block
                    dy_v[i][(m2,m1)][0, :] -= data_recv[rank_orig][1][i][(m2, m1)]

        return (dx_u, dy_v)


    def global_back_diff_ext(self, u : List[ParallelArray], \
                        v : List[ParallelArray]) \
            -> Tuple[List[ParallelArray], List[ParallelArray]]:
        """
        Global differentiation for backward differences. Extends total 
        domain by 1 column and 1 row.
        ASSUMES NEUMANN-BOUNDARY-CONDITIONS.
        If for example u1 and v1 live on an N2xN1-pixel-array (globally), 
        the derivatives are (N2+1)x(N1+1)-dimensional. The lowest 
        and right-most blocks get added a row / a column.
                                [a b c]
        Example:    u1 = v1 =   [d e f],
                                [g h i]
                         [0 b-a c-b | 0]           [ 0   0   0  |  0 ]
        Then:    dx u1 = [0 e-d f-e | 0],  dy v1 = [d-a e-b f-c | f-c]
                         [0 h-g i-h | 0]           [g-d h-e i-f | i-f]
                         [----------+--]           [------------+----]
                         [0 h-g i-h | 0]           [ 0   0   0  |  0 ]
        Takes all (locally represented) arrays u1, u2,... from u
        and applies backwards-derivative in x-direction on them.
        Takes all (locally represented) arrays v1, v2,... from v
        and applies backwards-derivative in y-direction on them.
        Returns ([dx u1, dx u2, ...], [dy v1, dy v2, ...])
        """
        #1) Initializations
        if hasattr(self, 'h'):
            h = self.h
        else:
            h = 1
            self.log("WARNING: TRIntersectingDDAlgorithm.global_back_diff_ext being used without parameter h set.", 0)

        dx_u = [ParallelArray({}) for _ in range(len(u))]
        dy_v = [ParallelArray({}) for _ in range(len(v))]
        u_rightest_noninters = [{} for _ in range(len(u))]
        v_bottom_noninters = [{} for _ in range(len(v))]
        data_to_send = [copy.deepcopy((u_rightest_noninters, v_bottom_noninters)) \
                        for targ_rank in range(self.num_ranks)]
        #use extended domain decomposition:
        dd_ext = self.dd.duplicate_with_additional_row_and_col()

        #2) Differentiate local image. 
        #  At first, treat boundaries as if this was the whole domain.
        #  Then correct rightest column / bottom row
        #  Send rightest non-intersecting column / bottom non-intersecting row 
        #   to neighboring domain
        #2.1) X-Derivatives
        for i, ui_pa in enumerate(u):
            ui = ui_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = dd_ext.sizes2[m2], dd_ext.sizes1[m1]
                dx_ui_m = np.zeros((sy, sx), dtype=np.float64)
                #differentiation of all blocks
                if (m2 < dd_ext.M2 - 1):
                    #differentiation (back diff) of all local columns except leftest
                    if (m1 < dd_ext.M1 - 1):
                        dx_ui_m[:,1:] = (1./h) * (ui[(m2,m1)][:,1:] - ui[(m2,m1)][:,:-1])
                    #at global right-most column fill in zeros (Neumann)
                    else:   #m1 == dd_ext.M1 - 1
                        dx_ui_m[:,1:-1] = (1./h) * (ui[(m2,m1)][:,1:] - ui[(m2,m1)][:,:-1])
                    #at global left-most column fill in zeros (Neumann)
                    if (m1 > 0):
                        #at local left-most columns fill the known half of finite difference
                        dx_ui_m[:,0] = (1./h) * ui[(m2,m1)][:,0]
                        #for m1 == 0 leave zeros at the left-most column (Neumann)
                #at global bottom-most row copy 2nd bottom-most (Neumann)
                else:   #m2 == dd_ext.M2 - 1
                    if (m1 < dd_ext.M1 - 1):
                        dx_ui_m[:-1,1:] = (1./h) * (ui[(m2,m1)][:,1:] - ui[(m2,m1)][:,:-1])
                    else:   # m1 == dd_ext.M1 - 1
                        dx_ui_m[:-1,1:-1] = (1./h) * (ui[(m2,m1)][:,1:] - ui[(m2,m1)][:,:-1])
                    if (m1 > 0):
                        dx_ui_m[:-1,0] = (1./h) * ui[(m2,m1)][:,0]
                    #mirror 2nd bottom-most row
                    dx_ui_m[-1,:] = dx_ui_m[-2,:]
                
                #pack local derivative into bigger structure
                dx_u[i][(m2,m1)] = dx_ui_m

                #send required info for the local left-most columns to
                #  right neighbor block
                if (m1 < dd_ext.M1 - 1):
                    m2_targ, m1_targ = m2, m1 + 1
                    rank_targ = dd_ext.thread_map[(m2_targ, m1_targ)]
                    rightest_noninters_index = \
                            dd_ext.lbounds[m1_targ] - dd_ext.lbounds[m1] - 1
                    #0 = u_rightest_noninters
                    data_to_send[rank_targ][0][i][(m2_targ,m1_targ)] = \
                            ui[(m2,m1)][:,rightest_noninters_index]

        #2.2) Y-Derivatives
        for i, vi_pa in enumerate(v):
            vi = vi_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = dd_ext.sizes2[m2], dd_ext.sizes1[m1]
                dy_vi_m = np.zeros((sy, sx), dtype=np.float64)
                #differentiation of all blocks
                if (m1 < dd_ext.M1 - 1):
                    #differentiation (back diff) of all local rows except upper-most
                    if (m2 < dd_ext.M2 - 1):
                        dy_vi_m[1:,:] = (1./h) * (vi[(m2,m1)][1:,:] - vi[(m2,m1)][:-1,:])
                    #at global bottom-most column fill in zeros (Neumann)
                    else:   # m2 == dd_ext.M2 - 1
                        dy_vi_m[1:-1,:] = (1./h) * (vi[(m2,m1)][1:,:] - vi[(m2,m1)][:-1,:])
                    if(m2 > 0):
                        #at local upper-most rows fill the known half of finite difference
                        dy_vi_m[0,:] = (1./h) * vi[(m2,m1)][0,:]
                        #for m2 == 0 leave zeros at the uppermost row (Neumann)
                #at global rightest column copy 2nd rightest (Neumann)
                else:   #m1 == dd_ext.M1 - 1
                    if (m2 < dd_ext.M2 - 1):
                        dy_vi_m[1:,:-1] = (1./h) * (vi[(m2,m1)][1:,:] - vi[(m2,m1)][:-1,:])
                    else:   # m2 == dd_ext.M2 - 1
                        dy_vi_m[1:-1,:-1] = (1./h) * (vi[(m2,m1)][1:,:] - vi[(m2,m1)][:-1,:])
                    if (m2 > 0):
                        dy_vi_m[0,:-1] = (1./h) * vi[(m2,m1)][0,:]
                    #mirror 2nd rightest column
                    dy_vi_m[:,-1] = dy_vi_m[:,-2]
                
                #pack local derivative into bigger structure
                dy_v[i][(m2,m1)] = dy_vi_m

                #send required info for the local upper-most rows to
                #  lower neighbor block
                if (m2 < dd_ext.M2 - 1):
                    m2_targ, m1_targ = m2 + 1, m1
                    rank_targ = dd_ext.thread_map[(m2_targ, m1_targ)]
                    bottom_noninters_index = \
                            dd_ext.ubounds[m2_targ] - dd_ext.ubounds[m2] - 1
                    #1 = v_bottom_noninters
                    data_to_send[rank_targ][1][i][(m2_targ,m1_targ)] = vi[(m2,m1)][bottom_noninters_index,:]

        #3) Now distribute data to other threads
        data_recv = self.distribute_data(data_to_send)

        #4.1) Unpack data_recv and rightest non-interecting column of neighboring left block
        for i, ui in enumerate(u):
            for m2, m1 in self.relevant_domains:
                if (m1 > 0):
                    m2_orig, m1_orig = m2, m1 - 1
                    rank_orig = dd_ext.thread_map[(m2_orig, m1_orig)]
                    #0 = rightest non-intersecting column of neighboring left block
                    if (m2 < dd_ext.M2 - 1):
                        dx_u[i][(m2,m1)][:, 0] -= data_recv[rank_orig][0][i][(m2, m1)]
                    else:   #m2 == dd_ext.M2 - 1
                        dx_u[i][(m2,m1)][:-1, 0] -= data_recv[rank_orig][0][i][(m2, m1)]
                        dx_u[i][(m2,m1)][-1, 0] = dx_u[i][(m2,m1)][-2, 0]

        #4.2) Unpack data_recv and bottom non-intersecting row of neighboring upper block
        for i, vi in enumerate(v):
            for m2, m1 in self.relevant_domains:
                if (m2 > 0):
                    m2_orig, m1_orig = m2 - 1, m1
                    rank_orig = dd_ext.thread_map[(m2_orig, m1_orig)]
                    #1 = bottom non-intersecting row of neighboring upper block
                    if (m1 < dd_ext.M1 - 1):
                        dy_v[i][(m2,m1)][0, :] -= data_recv[rank_orig][1][i][(m2, m1)]
                    else:   #m1 == dd_ext.M1 - 1
                        dy_v[i][(m2,m1)][0, :-1] -= data_recv[rank_orig][1][i][(m2, m1)]
                        dy_v[i][(m2,m1)][0, -1] = dy_v[i][(m2,m1)][0, -2]
        
        return (dx_u, dy_v)


    def global_forw_diff(self, u : List[ParallelArray], \
                        v : List[ParallelArray]) \
            -> Tuple[List[ParallelArray], List[ParallelArray]]:
        """
        Global differentiation for forward differences for ParallelArrays.
        Uses same forward differences as Chambolle.
                                [a b c d]
        Example:    u1 = v1 =   [e f g h],
                                [i j k l]
                                [m n o p]
                         [b-a c-b d-c  0 ]           [e-a f-b g-c h-d]
        Then:    dx u1 = [f-e g-f h-g  0 ],  dy v1 = [i-e j-f k-g l-h]
                         [j-i k-j l-k  0 ]           [m-i n-j o-k p-l]
                         [n-m o-n p-o  0 ]           [ 0   0   0   0 ]
        Takes all (locally represented) arrays u1, u2,... from u
        and applies forward derivative in x-direction on them.
        Takes all (locally represented) arrays v1, v2,... from v
        and applies forward derivative in y-direction on them.
        Returns ([dx u1, dx u2, ...], [dy v1, dy v2, ...])
        """
        if hasattr(self, 'h'):
            h = self.h
        else:
            h = 1
            self.log("WARNING: TRIntersectingDDAlgorithm.global_forw_diff being used without parameter h set.", 0)

        dx_u = [ParallelArray({}) for _ in range(len(u))]
        dy_v = [ParallelArray({}) for _ in range(len(v))]
        u_leftest_noninters = [{} for _ in range(len(u))]
        v_top_noninters = [{} for _ in range(len(v))]
        data_to_send = [copy.deepcopy((u_leftest_noninters, v_top_noninters)) \
                        for targ_rank in range(self.num_ranks)]


        #Differentiate local image. 
        #  At first, treat boundaries as if this was the whole domain.
        #  Send leftest non-intersecting column / top non-intersecting row 
        #   to neighboring domain
        for i, ui_pa in enumerate(u):
            ui = ui_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = self.dd.sizes2[m2], self.dd.sizes1[m1]
                dx_ui_m = np.zeros((sy, sx), dtype=np.float64)
                pd0lcb.diffF_x(ui[(m2,m1)], dx_ui_m, sx, sy, True, h)
                dx_u[i][(m2,m1)] = dx_ui_m
                if (m1 > 0):
                    m2_targ, m1_targ = m2, m1 - 1
                    rank_targ = self.dd.thread_map[(m2_targ, m1_targ)]
                    leftest_noninters_index = \
                            self.dd.rbounds[m1_targ] - self.dd.lbounds[m1]
                    #0 = u_leftest_noninters
                    data_to_send[rank_targ][0][i][(m2_targ,m1_targ)] = \
                            ui[(m2,m1)][:,leftest_noninters_index]

        for i, vi_pa in enumerate(v):
            vi = vi_pa.val      #get dictionary from ParallelArray
            for m2, m1 in self.relevant_domains:
                sy, sx = self.dd.sizes2[m2], self.dd.sizes1[m1]
                dy_vi_m = np.zeros((sy, sx), dtype=np.float64)
                pd0lcb.diffF_y(vi[(m2,m1)], dy_vi_m, sx, sy, True, h)
                dy_v[i][(m2,m1)] = dy_vi_m
                if (m2 > 0):
                    m2_targ, m1_targ = m2 - 1, m1
                    rank_targ = self.dd.thread_map[(m2_targ, m1_targ)]
                    top_noninters_index = \
                            self.dd.bbounds[m2_targ] - self.dd.ubounds[m2]
                    #1 = v_top_noninters
                    data_to_send[rank_targ][1][i][(m2_targ,m1_targ)] = \
                            vi[(m2,m1)][top_noninters_index,:]

        #Now distribute data to other threads
        data_recv = self.distribute_data(data_to_send)

        #Unpack data_recv and leftest non-intersecting column of neighboring right block
        for i, ui in enumerate(u):
            for m2, m1 in self.relevant_domains:
                if (m1 < self.dd.M1 - 1):
                    m2_orig, m1_orig = m2, m1 + 1
                    rank_orig = self.dd.thread_map[(m2_orig, m1_orig)]
                    #0 = leftest non-intersecting column of neighboring right block
                    dx_u[i][(m2,m1)][:,-1] = \
                            data_recv[rank_orig][0][i][(m2, m1)] \
                            - ui[(m2,m1)][:,-1]

        #Unpack data_recv and top non-intersecting row of neighboring lower block
        for i, vi in enumerate(v):
            for m2, m1 in self.relevant_domains:
                if (m2 < self.dd.M2 - 1):
                    m2_orig, m1_orig = m2 + 1, m1
                    rank_orig = self.dd.thread_map[(m2_orig, m1_orig)]
                    #1 = top non-intersecting row of neighboring lower block
                    dy_v[i][(m2,m1)][-1, :] = \
                            data_recv[rank_orig][1][i][(m2, m1)] \
                            - vi[(m2,m1)][-1,:]

        return (dx_u, dy_v)



    def additive_intersection_exchange(self, array_par : ParallelArray, \
                                        num_channels1 : int = 1, num_channels2: int = 1, \
                                        extended_output=False) \
                                        -> ParallelArray:
        """
        The parameter array_dic stores data (2d-arrays) within all domains
        of the current rank.
        This function takes the data and sends the intersections to
        the other domains that are involved. If another rank is
        responsible, the data is sent with MPI.
        This is done in a way that deadlocks are avoided. 
        At the target domains, we start with a zero-array and 
        add up all data from the intersecting origin 2d-arrays.
        This can be made for multiple channels. If this shall be done
        num_channels1 must be greater 1 (arrays are 3-dimensional then).
        For for example 2x2 or 10x10 channels, even
        num_channels2 must be greater 1 (arrays are 4-dimensional then).
        If the parameter extended_output is set True, the return-Arrays will 
        contain an additional row/column left/above and right/below the
        domain-boundaries, if the domains are not at the boundary of the
        whole image.
        Example:
        num_channels1 = 1, num_channels2 = 1
        N1 = 5, N2 = 5, M1 = 2, M2 = 2, size of all domains: 3x3.
        num_threads = 2, thread 0 is responsible for (0,0) and (1,1)
        thread 1 is responsible for (0,1) and (1,0).
        For rank 0,         (1 1|1)                       (-4|-4 -4)
        array_dic[(0,0)] =  (1 1|1),   array_dic[(1,1)] = (--|-----)
                            (---|-)                       (-4|-4 -4)          
                            (1 1|1)                       (-4|-4 -4)
        For rank 1,         (10|10 10)                    (-40 -40|-40)
        array_dic[(0,1)] =  (10|10 10),array_dic[(1,0)] = (-------|---)
                            (--------)                    (-40 -40|-40)
                            (10|10 10)                    (-40 -40|-40)
        Then the output would be:
        For rank 0,         (  0   0| 10)                    (-29|10 10)
        out_dic[(0,0)]    = (  0   0| 10),  out_dic[(1,1)] = (---|-----)
                            (-------|---)                    (-40| 0  0)          
                            (-40 -40|-34)                    (-40| 0  0)
        For rank 1,         (  1| 0  0)                    ( 1  1| 7)
        out_dic[(0,1)] =    (  1| 0  0),  out_dic[(1,0)] = (-----|--)
                            (---------)                    ( 0  0|-4)
                            (-43|-4 -4)                    ( 0  0|-4)
        """
        
        #1) PREPARE DIMENSION AND HALO SIZE
        if num_channels2 > 1:
            dim = 4
        elif num_channels1 > 1:
            dim = 3
        else:
            dim = 2
        halosize = 0 if extended_output == False else 1


        #2) CREATE DICTIONARIES TO SEND TO OTHER DOMAINS
        array_dic = array_par.val
        idics_orig = [{} for _ in range(self.num_ranks)]
        for m2_orig, m1_orig in self.relevant_domains:
            assert np.ndim(array_dic[(m2_orig, m1_orig)]) == dim, \
                    "Incorrect shape dimension. Probably either num_channels1 or num_channels2 is wrong or arrays in array_dic have wrong dimension."
            
            #iterate over all boundaries
            b_it_objects = self.dd.intersection_iterator_objects(m1_orig, m2_orig, halosize, False)
            for condition, du, rl, row_slice, col_slice in b_it_objects:
                #check if there is intersection left/right/... of domain
                if(condition):
                    m2_targ, m1_targ = m2_orig + du, m1_orig + rl
                    rank_targ = self.dd.thread_map[m2_targ, m1_targ]
                    if not((m2_targ, m1_targ) in idics_orig[rank_targ]):
                        idics_orig[rank_targ][(m2_targ, m1_targ)] = {}
                    if (dim == 2):
                        idics_orig[rank_targ][(m2_targ, m1_targ)][(m2_orig, m1_orig)] \
                            = array_dic[(m2_orig, m1_orig)][row_slice, col_slice]
                    elif (dim == 3):
                        idics_orig[rank_targ][(m2_targ, m1_targ)][(m2_orig, m1_orig)] \
                            = array_dic[(m2_orig, m1_orig)][:num_channels1, row_slice, col_slice]
                    else:  #dim == 4
                        idics_orig[rank_targ][(m2_targ, m1_targ)][(m2_orig, m1_orig)] \
                            = array_dic[(m2_orig, m1_orig)][:num_channels2, :num_channels1, row_slice, col_slice]


        #3) DISTRIBUTION PROCESS TO OTHER RANKS
        idics_targ = self.distribute_data(idics_orig)
        

        #4) UNPACKING DICTIONARIES AT TARGET DOMAINS
        array_dic_out = {}
        for m2_targ, m1_targ in self.relevant_domains:
            #initialize target result
            #determine shapes of target result
            s1, s2 = self.dd.sizes1[m1_targ], self.dd.sizes2[m2_targ]
            if m1_targ > 0: s1 += halosize
            if m1_targ < self.dd.M1 - 1: s1 += halosize
            if m2_targ > 0: s2 += halosize
            if m2_targ < self.dd.M2 - 1: s2 += halosize
            if (dim == 2):
                shape_m2_m1 = (s2, s1)
            elif (dim == 3):
                shape_m2_m1 = (num_channels1, s2, s1)
            else: #dim == 4
                shape_m2_m1 = (num_channels2, num_channels1, s2, s1)
            array_dic_out[(m2_targ, m1_targ)] = np.zeros(shape_m2_m1, dtype=np.float64)

            #iterate over all boundaries 
            b_it_objects = self.dd.intersection_iterator_objects(m1_targ, m2_targ, halosize, True)

            for condition, du, rl, row_slice, col_slice in b_it_objects:
                if (condition):
                    m2_orig, m1_orig = m2_targ + du, m1_targ + rl
                    rank_orig = self.dd.thread_map[m2_orig, m1_orig]
                    if (dim == 2):
                        array_dic_out[(m2_targ, m1_targ)][row_slice, col_slice] \
                            += idics_targ[rank_orig][(m2_targ, m1_targ)][(m2_orig, m1_orig)]
                    elif (dim == 3):
                        array_dic_out[(m2_targ, m1_targ)][:num_channels1, row_slice, col_slice] \
                            += idics_targ[rank_orig][(m2_targ, m1_targ)][(m2_orig, m1_orig)]
                    else:   #dim == 4
                        array_dic_out[(m2_targ, m1_targ)][:num_channels2, :num_channels1, row_slice, col_slice] \
                            += idics_targ[rank_orig][(m2_targ, m1_targ)][(m2_orig, m1_orig)]

        return ParallelArray(array_dic_out)



    def divergence_from_extended_array(self, ext_array: np.ndarray, m2 : int, m1 : int) -> np.ndarray:
        """
        Computes global divergence of an array locally. Assumes the required
        additional boundary rows/columns are already added.
        This method can for example be used by calling the 
        additive_intesection_exchange-method with extended_output=True first.
        """
        assert ext_array.ndim == 3
        assert ext_array.shape[0] == 2
        if hasattr(self, 'h'):
            h = self.h
        else:
            h = 1
            self.log("WARNING: TRIntersectingDDAlgorithm.global_back_diff being used without parameter h set.", 0)

        #initialize div_arr
        s2_ext, s1_ext = ext_array.shape[1], ext_array.shape[2]
        s2_div, s1_div = s2_ext, s1_ext
        if m1 > 0: s1_div -= 1
        if m2 > 0: s2_div -= 1
        dx_arr = np.zeros(shape=(s2_div, s1_div), dtype=np.float64)
        dy_arr = np.zeros(shape=(s2_div, s1_div), dtype=np.float64)

        #Determine "global" divergence

        #inner parts:
        if (m1 > 0 and m2 > 0):
            dx_arr = (1. / h) * (ext_array[0,1:,1:] - ext_array[0,1:,:-1])
            if (m1 == self.dd.M1 - 1):  #adjustment for right boundary
                dx_arr[:,-1] = (1. / h) * (- ext_array[0,1:,-2])
            dy_arr = (1. / h) * (ext_array[1,1:,1:] - ext_array[1,:-1,1:])
            if (m2 == self.dd.M2 - 1):  #adjustment for bottom boundary
                dy_arr[-1,:] = (1. / h) * (- ext_array[1,-2,1:])
        #special treatment for left and upper boundaries:
        elif (m1 > 0 and m2 == 0):
            dx_arr = (1. / h) * (ext_array[0,:,1:] - ext_array[0,:,:-1])
            if (m1 == self.dd.M1 - 1):  #adjustment for right boundary
                dx_arr[:,-1] = (1. / h) * (- ext_array[0,:,-2])
            dy_arr[0,:] = (1. / h) * ext_array[1,0,1:]
            dy_arr[1:,:] = (1. / h) * (ext_array[1,1:,1:] - ext_array[1,:-1,1:])
            if (m2 == self.dd.M2 - 1):  #adjustment for bottom boundary
                dy_arr[-1,:] = (1. / h) * (- ext_array[1,-2,1:])
        elif (m1 == 0 and m2 > 0):
            dx_arr[:,0] = (1. / h) * ext_array[0,1:,0]
            dx_arr[:,1:] = (1. / h) * (ext_array[0,1:,1:] - ext_array[0,1:,:-1])
            if (m1 == self.dd.M1 - 1):  #adjustment for right boundary
                dx_arr[:,-1] = (1. / h) * (- ext_array[0,1:,-2])
            dy_arr = (1. / h) * (ext_array[1,1:,:] - ext_array[1,:-1,:])
            if (m2 == self.dd.M2 - 1):  #adjustment for bottom boundary
                dy_arr[-1,:] = (1. / h) * (- ext_array[1,-2,:])
        elif (m1 == 0 and m2 == 0):
            dx_arr[:,0] = (1. / h) * ext_array[0,:,0]
            dx_arr[:,1:] = (1. / h) * (ext_array[0,:,1:] - ext_array[0,:,:-1])
            if (m1 == self.dd.M1 - 1):  #adjustment for right boundary
                dx_arr[:,-1] = (1. / h) * (- ext_array[0,:,-2])
            dy_arr[0,:] = (1. / h) * ext_array[1,0,:]
            dy_arr[1:,:] = (1. / h) * (ext_array[1,1:,:] - ext_array[1,:-1,:])
            if (m2 == self.dd.M2 - 1):  #adjustment for bottom boundary
                dy_arr[-1,:] = (1. / h) * (- ext_array[1,-2,:])

        div_arr = dx_arr + dy_arr

        return div_arr



    def collect_halo(self, arr : ParallelArray) -> ParallelArray:
        """
        Takes parallel array arr and collects a column right and a row
        below every domain and returns it as arr_ext
        """
        #1) INITIALIZE EXTENDED ARRAY
        arr_ext_dict = {}
        for m2, m1 in self.dd.relevant_domains[self.rank]:
            ext_r = 0 if m1 == self.dd.M1 - 1 else 1
            ext_b = 0 if m2 == self.dd.M2 - 1 else 1
            sx, sy = self.dd.sizes1[m1], self.dd.sizes2[m2]
            shape_m2_m1_ext = (sy + ext_b, sx + ext_r)
            arr_ext_dict[(m2,m1)] = np.zeros(shape_m2_m1_ext, dtype=np.float64)
            arr_ext_dict[(m2,m1)][:sy,:sx] = arr[(m2,m1)]


        #2) CREATE DICTIONARIES TO SEND TO OTHER DOMAINS
        idics_orig = [{} for _ in range(self.num_ranks)]
        for m2_orig, m1_orig in self.dd.relevant_domains[self.rank]:
            assert np.ndim(arr.val[(m2_orig, m1_orig)]) == 2, \
                    "Incorrect shape dimension."
            
            #iterate over left and upper boundary
            ilb_loc = self.dd.ilbounds[m1_orig] - self.dd.lbounds[m1_orig]
            iub_loc = self.dd.iubounds[m2_orig] - self.dd.ubounds[m2_orig]
            it_objects = [(m1_orig > 0, 0, -1, slice(None, None), ilb_loc), \
                          (m2_orig > 0, -1, 0, iub_loc, slice(None, None)), \
                          (m1_orig > 0 and m2_orig > 0, -1, -1, iub_loc, ilb_loc)]

            #check if there is intersection left of domain / on top of domain
            for condition, ud, lr, row_slice, col_slice in it_objects:
                if(condition):
                    m2_targ, m1_targ = m2_orig + ud, m1_orig + lr
                    rank_targ = self.dd.thread_map[m2_targ, m1_targ]
                    if not((m2_targ, m1_targ) in idics_orig[rank_targ]):
                        idics_orig[rank_targ][(m2_targ, m1_targ)] = {}
                    idics_orig[rank_targ][(m2_targ, m1_targ)][(m2_orig, m1_orig)] \
                        = arr[(m2_orig, m1_orig)][row_slice, col_slice]


        #3) DISTRIBUTION PROCESS TO OTHER RANKS
        idics_targ = self.distribute_data(idics_orig)
        

        #4) UNPACKING DICTIONARIES AT TARGET DOMAINS
        for m2_targ, m1_targ in self.dd.relevant_domains[self.rank]:
            #iterate over right and bottom boundary
            rb = -1 if m1_targ < self.dd.M1 - 1 else None   #right boundary of slice
            bb = -1 if m2_targ < self.dd.M2 - 1 else None   #bottom boundary of slice
            it_objects = [(m1_targ < self.dd.M1 - 1, 0, 1, slice(None, bb), -1), \
                          (m2_targ < self.dd.M2 - 1, 1, 0, -1, slice(None, rb)), \
                          (m1_targ < self.dd.M1 - 1 and m2_targ < self.dd.M2 - 1, 1, 1, -1, -1)]

            for condition, du, rl, row_slice, col_slice in it_objects:
                if (condition):
                    m2_orig, m1_orig = m2_targ + du, m1_targ + rl
                    rank_orig = self.dd.thread_map[m2_orig, m1_orig]
                    arr_ext_dict[(m2_targ, m1_targ)][row_slice, col_slice] \
                        += idics_targ[rank_orig][(m2_targ, m1_targ)][(m2_orig, m1_orig)]

        arr_ext = ParallelArray(arr_ext_dict)
        return arr_ext



    def find_backdiff_scalar_potential(self, n1 : ParallelArray, n2 : ParallelArray) -> ParallelArray:
        """
        Finds a potential g such that
        grad^-(g) = (n1 n2).
        Note that backward-differences are used to derive.
        IT IS ALREADY ASSUMED THAT (n1,n2) IS CURL-FREE w.r.t. backward-differences:
        rot^-(n1,n2) = 0    <=>    dy^- n1 = dx^- n2
        The result g will have the measures N2xN1 of self.dd.
        The value
        g[0,0] = 0.0
        will be set.
        Independent from the fact if the size of (n1,n2) is
        2xN2xN1 or 2x(N2+1)x(N1+1), the algorithm will only use the pixels
        (0,1), (0,2), ..., (0,N1-1), (1,1),..., (N2-1,N1-1) of n1 and
        (1,0), (1,1), ..., (1,N1-1), (2,0),..., (N2-1,N1-1) of n2. 
        From these pixels, the values
        g[0,1], g[0,2], ..., g[N2-1,N1-1] will be determined.
        """
        
        M1, M2 = self.dd.M1, self.dd.M2
        g_dic = {}
        data_store1 = {}    #remember data that is sent to another
        data_store2 = {}    #  domain of the same rank
                            #  index: target domain
        reqs1_send = []     #MPI-object stores
        reqs4_send = []

        #Iterate over domains column-wise from top to bottom
        # determine pixels from upper left corner to lower left (Step 2 to 4) 
        # and then all pixels from left to right (Step 5 to 7)
        for m1 in range(M1):
            for m2 in range(M2):
                if (m2, m1) in self.dd.get_relevant_domains(self.rank):
                    sx, sy = self.dd.sizes1[m1], self.dd.sizes2[m2]
                    g_dic[(m2,m1)] = np.zeros((sy, sx), dtype=np.float64)

                    #1) Top left domain: Assume top left corner
                    if m1 == 0 and m2 == 0:
                        g_dic[(m2,m1)][0,0] = 0.0

                    #2) Left domains except upperpost: Receive info about left column from above
                    if m1 == 0 and m2 > 0:
                        #receive pixel from rank of domain (m2 - 1, m1)
                        rank_orig = self.dd.thread_map[m2 - 1, m1]
                        if rank_orig == self.rank:
                            recv_pixel = data_store1[(m2, m1)]
                        else:
                            ser_data_recv = self.mpi_comm.irecv(source=rank_orig, tag=10000+m2*100+m1).wait()
                            recv_pixel = pickle.loads(ser_data_recv)
                        #recv_pixel is the pixel above the boundary
                        g_dic[(m2,m1)][0,0] = recv_pixel \
                                + self.h * n2[(m2,m1)][0,0]                 

                    #3) Left domains: Determine left columns
                    if m1 == 0:
                        g_dic[(m2,m1)][1:,0] = g_dic[(m2,m1)][0,0] \
                                    + self.h * np.cumsum(n2[(m2,m1)][1:sy,0], axis=0)

                    #4) Left domains except bottommost: Send info about left column to lower domain
                    if m1 == 0 and m2 < M2 - 1:
                        #y-pos of pixel above boundary of lower neighbor domain
                        yab = self.dd.ibbounds[m2] - 1 - self.dd.ubounds[m2]
                        #pixel above boundary
                        send_pixel = g_dic[(m2,m1)][yab,0]
                        #send pixel to rank of domain (m2 + 1, m1)
                        rank_recv = self.dd.thread_map[m2 + 1, m1]
                        if rank_recv == self.rank:
                            data_store1[(m2 + 1, m1)] = send_pixel
                        else:
                            ser_data_send = pickle.dumps(send_pixel)
                            req1 = self.mpi_comm.isend(ser_data_send, dest=rank_recv, tag=10000+(m2+1)*100+m1)
                            reqs1_send.append(req1)

                    #5) All domains except leftest: Receive info about left column from left
                    if m1 > 0:
                        #receive recv_column from rank (m2,m1-1)
                        rank_orig = self.dd.thread_map[m2, m1 - 1]
                        if (rank_orig == self.rank):
                            recv_column = data_store2[(m2, m1)]
                        else:
                            data_length = self.mpi_comm.irecv(source=rank_orig, tag=30000+m2*100+m1).wait()
                            ser_data_recv = bytearray(data_length)
                            self.mpi_comm.Irecv([ser_data_recv, MPI.BYTE], source=rank_orig, tag=40000+m2*100+m1).Wait()
                            recv_column = pickle.loads(ser_data_recv)
                        #recv_column is the column left of left boundary
                        g_dic[(m2,m1)][:,0] = recv_column \
                                    + self.h * n1[(m2,m1)][:sy,0]

                    #6) (MAIN STEP) All domains: integrate from left to right
                    g_dic[(m2,m1)][:,1:] = g_dic[(m2,m1)][:,:1] \
                            + self.h * np.cumsum(n1[(m2,m1)][:sy,1:sx], axis=1)

                    #7) All domains except rightest: Send info about column left of 
                    #   right neighbor domain to right neighbor domain
                    if m1 < M1 - 1:
                        #x-pos of column left of boundary of right neighbor domain
                        xlb = self.dd.irbounds[m1] - 1 - self.dd.lbounds[m1]
                        #column left of left boundary of right neighbor domain
                        send_column = g_dic[(m2,m1)][:,xlb]
                        #send send_column to rank of domain (m2, m1+1)
                        rank_recv = self.dd.thread_map[m2, m1+1]
                        if rank_recv == self.rank:
                            data_store2[(m2, m1 - 1)] = send_column
                        else:
                            ser_data_send = pickle.dumps(send_column)
                            data_size = len(ser_data_send)
                            self.mpi_comm.isend(data_size, dest=rank_recv, tag=30000+m2*100+m1+1).wait()
                            req4 = self.mpi_comm.Isend([ser_data_send, MPI.BYTE], dest=rank_recv, tag=40000+m2*100+m1+1)
                            reqs4_send.append(req4)

        g = ParallelArray(g_dic)
        return g






class TRTangentFieldSmoothing(TRIntersectingDDAlgorithm):
    
    def __init__(self, _domain_decomposition : pa.ThreadedHUIP, \
                _tfs_parameters : dict, _rec : Recorder, \
                _mpi_comm : MPI.Comm):
        super().__init__(_domain_decomposition, _tfs_parameters, _rec, _mpi_comm)
        self.dd.init_domains(self.dd.relevant_domains[self.rank]) #prepare theta-weights


    def initialize_parameters(self, tfs_parameters : dict):
        if (tfs_parameters == {}):  #only possible for test purposes
            print("WARNING: TRTangentFieldSmoothing, no tfs_parameters initialized.")
            return
        self.k = tfs_parameters["k"]
        self.delta = tfs_parameters["delta"]
        self.alpha = tfs_parameters["alpha"]

        self.outer_stop_criteria = tfs_parameters["outer_stop_criteria"]
        self.inner_stop_criteria = tfs_parameters["inner_stop_criteria"]
        self.num_outer_it_max = tfs_parameters["num_outer_it_max"]
        self.num_inner_it_max = tfs_parameters["num_inner_it_max"]
        self.inner_cauchy_thresh = tfs_parameters["inner_cauchy_thresh"]
        self.outer_cauchy_thresh = tfs_parameters["outer_cauchy_thresh"]
    
    
    def proj_div_0_sparse(self, tau : np.ndarray, m1 : int, m2 : int, \
                        halo_r : int = 0, \
                        halo_b : int = 0) -> np.ndarray:
        """
        Applies global projection on subspace divB(tau)=0 on tau.
        Assumes that tau is only non-zero on domain (m2,m1).
        Only on this domain shall tau be provided.
        Only on domain (m2,m1) will the projection be evaluated 
        and returned.
        """

        halopartition = self.dd.get_halo_partition_with_fixed_extended_block(m1, m2, halo_r, halo_b)
        projobj = pd0lc.ProjectorDiv0(halopartition, self.h)
        P_tau = projobj.proj_sparse(tau, m1, m2)

        return P_tau


    def proj_div_0_compl(self, tau : ParallelArray, \
                        halo_r : int = 0, \
                        halo_b : int = 0) -> ParallelArray:
        """
        Applies global projection on subspace divB(tau)=0 on tau.
        Assumes that tau can be non-zero everywhere.
        tau must be passed as dictionary for all domains (m2,m1),
        for which this rank is responsible.
        Returns P_tau for all domains (m2,m1),
        for which this rank is responsible.
        If halo_r > 0, then halo_r columns of the result are added to
            each P_tau[(m2,m1)], if not at the right border.
        If halo_b > 0, then halo_b rows of the result are added to 
            each P_tau[(m2,m1)], if not at the right border.
        """
        #Initialize Projected tau for relevant domains of this rank
        P_tau = ParallelArray({})
        for m2, m1 in self.dd.relevant_domains[self.rank]:
            if self.dd.M1 > 1: assert self.dd.rbounds[-2] + halo_r < self.dd.N1
            if self.dd.M2 > 1: assert self.dd.bbounds[-2] + halo_b < self.dd.N2
            drb = True if m1 == self.dd.M1 - 1 else False 
            dbb = True if m2 == self.dd.M2 - 1 else False 
            ext_r = 0 if drb else halo_r
            ext_b = 0 if dbb else halo_b
            shape_m2_m1_ext = (2, self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
            P_tau[(m2,m1)] = np.zeros(shape_m2_m1_ext, dtype=np.float64)

        for k1 in range(self.dd.M1):
            for k2 in range(self.dd.M2):
                #Create HaloPartition with fixed block (k2,k1) for all (k2,k1)
                halop_entry = self.dd.get_halo_partition_with_fixed_block(k1, k2)
                halop_exit = self.dd.get_halo_partition_with_fixed_extended_block(k1, k2, halo_r, halo_b)
                #Create fitting projector object around domain (k2,k1)
                projdiv0 = pd0lc.ProjectorDiv0.constructor_2_partitions( \
                            halop_entry, halop_exit, self.h)
                #Now compute contributions to all relevant (m2,m1) from (k2,k1)
                data_to_rank_k = {}
                for m2, m1 in self.dd.relevant_domains[self.rank]:  
                    #Extract part of tau_m that shall be projected to k
                    lb_loc = halop_entry.lbounds[m1] - self.dd.lbounds[m1]
                    rb_loc = halop_entry.rbounds[m1] - self.dd.lbounds[m1]
                    ub_loc = halop_entry.ubounds[m2] - self.dd.ubounds[m2]
                    bb_loc = halop_entry.bbounds[m2] - self.dd.ubounds[m2]
                    tau_k_m_part = tau[(m2, m1)][:, ub_loc:bb_loc, lb_loc:rb_loc]
                    assert tau_k_m_part.shape == (2,bb_loc-ub_loc,rb_loc-lb_loc)
                    #Project that part
                    P_tau_k_m_part = projdiv0.proj_sparse_general( \
                                tau_k_m_part, m1, m2, k1, k2)
                    data_to_rank_k[(m2,m1)] = P_tau_k_m_part
                #Collect all the data on the rank, which is responsible for (k2,k1)
                rank_targ = self.dd.thread_map[k2,k1]
                #send data_to_rank_k to rank k
                self.mpi_comm.Barrier()
                if (self.rank != rank_targ):
                    ser_data_to_k = pickle.dumps(data_to_rank_k)
                    data_size = len(ser_data_to_k)
                    self.mpi_comm.isend(data_size, dest=rank_targ, tag=1000000+k1*10000+k2*100+self.rank).wait()
                    self.mpi_comm.Isend([ser_data_to_k, MPI.BYTE], dest=rank_targ, tag=2000000+k1*10000+k2*100+self.rank).Wait()
                #Add contributions from k to result
                if (self.rank == rank_targ):  #(m2,m1)==(k2,k1)
                    for rank_orig in range(self.num_ranks):
                        if (rank_orig == rank_targ):
                            data_from_rank_orig = data_to_rank_k
                        else:
                            #receive data_from_rank_orig
                            data_length = self.mpi_comm.irecv(source=rank_orig, tag=1000000+k1*10000+k2*100+rank_orig).wait()
                            ser_data_from_orig = bytearray(data_length)
                            self.mpi_comm.Irecv([ser_data_from_orig, MPI.BYTE], source=rank_orig, tag=2000000+k1*10000+k2*100+rank_orig).Wait()
                            data_from_rank_orig = pickle.loads(ser_data_from_orig)
                        for l2, l1 in self.dd.relevant_domains[rank_orig]:
                            P_tau_m_l_part = data_from_rank_orig[(l2, l1)]
                            P_tau[(k2,k1)] += P_tau_m_l_part    #m=k
        return P_tau
    

    def compute_P_div_q0(self, pj : ParallelArray) -> ParallelArray:
        """
        Returns (P_K div^- q^0)_m
        with q^0_k = sum_{l!=k} (theta_l pj_l) 
        for all domains m = (m2,m1), for which this rank is responsible.
        The idea is to perform the calculation like the global projection
        but to split the calculation by performing all of the
        P_K div^- (theta_l pj_l)
        separately.
        """
        
        theta_pj = ParallelArray({})
        div_theta_pj = ParallelArray({})
        P_div_q0 = ParallelArray({})
        for m2, m1 in self.dd.relevant_domains[self.rank]:
            nr, nc = self.dd.sizes2[m2], self.dd.sizes1[m1]

            #Compute theta_l pj_l
            theta_pj[(m2,m1)] = np.zeros((2,2,nr,nc), dtype=np.float64)
            theta_pj[(m2,m1)][0,0,:,:] = self.dd.theta_loc(m2, m1) * pj[(m2,m1)][0,0,:,:]
            theta_pj[(m2,m1)][0,1,:,:] = self.dd.theta_loc(m2, m1) * pj[(m2,m1)][0,1,:,:]
            theta_pj[(m2,m1)][1,0,:,:] = self.dd.theta_loc(m2, m1) * pj[(m2,m1)][1,0,:,:]
            theta_pj[(m2,m1)][1,1,:,:] = self.dd.theta_loc(m2, m1) * pj[(m2,m1)][1,1,:,:]

            #Compute extended shapes
            drb = True if m1 == self.dd.M1 - 1 else False 
            dbb = True if m2 == self.dd.M2 - 1 else False 
            ext_r = 0 if drb else 1
            ext_b = 0 if dbb else 1
            shape_ext_m2_m1 = (2, nr + ext_b, nc + ext_r)
            assert shape_ext_m2_m1 == (2, pj[(m2, m1)].shape[2] + ext_b, pj[(m2, m1)].shape[3] + ext_r)
            
            #Compute div_theta_pj from theta and pj
            div_theta_pj[(m2,m1)] = np.zeros(shape_ext_m2_m1, dtype=np.float64)
            pd0lcb.divB(theta_pj[(m2,m1)][0,:,:,:], div_theta_pj[(m2,m1)][0,:,:], \
                        self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            pd0lcb.divB(theta_pj[(m2,m1)][1,:,:,:], div_theta_pj[(m2,m1)][1,:,:], \
                        self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            
            #Initialize P_K div^-(q^0) for relevant domains of this rank
            P_div_q0[(m2,m1)] = np.zeros(shape_ext_m2_m1, dtype=np.float64)

        #Prepare halo partitions for projection
        halo_partitions = {}
        for k1 in range(self.dd.M1):
            for k2 in range(self.dd.M2):
                #Create HaloPartition with fixed block (k2,k1) for all (k2,k1)
                #  and 1 additional column and row for block, if not edge / bottom
                halo_partitions[(k2,k1)] = \
                    self.dd.get_halo_partition_with_fixed_extended_block(k1, k2, 1, 1)


        #Perform projection of div_theta_pj
        for k1 in range(self.dd.M1):
            for k2 in range(self.dd.M2):
                #Now compute contributions to all relevant (m2,m1) from (k2,k1)
                data_to_rank_k = {}
                for m2, m1 in self.dd.relevant_domains[self.rank]:
                    if (m2 == k2 and m1 == k1): continue
                    #Create fitting projector object around domain (k2,k1) and (m2,m1)
                    projdiv0 = pd0lc.ProjectorDiv0.constructor_2_partitions( \
                            _entry_partition=halo_partitions[(m2,m1)], \
                            _exit_partition=halo_partitions[(k2,k1)], \
                            _h=self.h)
                    #Perform the actual projection
                    P_div_theta_pj_m = projdiv0.proj_sparse_general( \
                                div_theta_pj[(m2,m1)], m1, m2, k1, k2)
                    data_to_rank_k[(m2,m1)] = P_div_theta_pj_m
                #Collect all the data on the rank, which is responsible for (k2,k1)
                rank_targ = self.dd.thread_map[k2,k1]
                #send data_to_rank_k to rank k
                self.mpi_comm.Barrier()
                if (self.rank != rank_targ):
                    ser_data_to_k = pickle.dumps(data_to_rank_k)
                    data_size = len(ser_data_to_k)
                    self.mpi_comm.isend(data_size, dest=rank_targ, tag=1000000+k1*10000+k2*100+self.rank).wait()
                    self.mpi_comm.Isend([ser_data_to_k, MPI.BYTE], dest=rank_targ, tag=2000000+k1*10000+k2*100+self.rank).Wait()
                #Add contributions from k to result
                if (self.rank == rank_targ):  #(m2,m1)==(k2,k1)
                    for rank_orig in range(self.num_ranks):
                        if (rank_orig == rank_targ):
                            data_from_rank_orig = data_to_rank_k
                        else:
                            #receive data_from_rank_orig
                            data_length = self.mpi_comm.irecv(source=rank_orig, tag=1000000+k1*10000+k2*100+rank_orig).wait()
                            ser_data_from_orig = bytearray(data_length)
                            self.mpi_comm.Irecv([ser_data_from_orig, MPI.BYTE], source=rank_orig, tag=2000000+k1*10000+k2*100+rank_orig).Wait()
                            data_from_rank_orig = pickle.loads(ser_data_from_orig)
                        for l2, l1 in self.dd.relevant_domains[rank_orig]:
                            if (l2 == k2 and l1 == k1): continue   #remember m=k
                            P_div_theta_pj_l = data_from_rank_orig[(l2, l1)]
                            P_div_q0[(k2,k1)] += P_div_theta_pj_l       #remember m=k
        
        return P_div_q0


    def evaluate_outer_criteria(self, pj : ParallelArray, \
                                pj_m1 : ParallelArray = None, \
                                old_o_energy : float = None, \
                                P_tau0 : ParallelArray = None) -> Tuple[bool, str, float]:
        """
        Evaluate the stopping criteria of the outer Chambolle loop.
        Args:
          pj: current dual tangent field
          pj_m1: last dual tangent field
          old_o_energy: old outer energy
          P_tau0: projected initial (noisy) tangent field
        """
        
        outer_criteria = False
        o_criteria_info = ""
        o_energy = None
        if self.outer_stop_criteria == "Cauchy":
            assert old_o_energy != None or P_tau0 != None, \
                "old_o_energy and P_tau0 must be provided for self.outer_stop_criteria == ""Cauchy""."
            
            #get Pdivpj from pj
            dx_pj_l, dy_pj_l = self.global_back_diff( \
                    [pj[0,0,:,:], pj[1,0,:,:]], [pj[0,1,:,:], pj[1,1,:,:]])
            divpj = (dx_pj_l[0] + dy_pj_l[0]).stack( \
                    dx_pj_l[1] + dy_pj_l[1], axis=0)
            Pdivpj = self.proj_div_0_compl(divpj)

            #Expression within integral
            expr = Pdivpj - P_tau0 * (1.0 / self.delta)
            
            #Sum squares everything up: Create non-overlapping DD as subsets of DD
            part = self.dd.get_partition_with_fixed_block(0,0)

            #Sum squares everything up: Create local sums
            sum_loc_expr_sq = 0.0
            for m2,m1 in self.dd.relevant_domains[self.rank]:
                #boundaries of non-overl DD locally in overl DD
                lbl = part.lbounds[m1] - self.dd.lbounds[m1]
                lbr = part.rbounds[m1] - self.dd.lbounds[m1]
                lbu = part.ubounds[m2] - self.dd.ubounds[m2]
                lbb = part.bbounds[m2] - self.dd.ubounds[m2]
                sum_loc_expr_sq += np.linalg.norm(expr[(m2,m1)][:,lbu:lbb,lbl:lbr]) ** 2
            self.mpi_comm.Barrier()
            
            #Collect sums from all ranks, add them together and determine energy
            sum_collection = self.collect_data_at_rank(sum_loc_expr_sq, 0)
            if self.rank == 0:
                sum_expr_sq = np.sum(sum_collection)
                o_energy = math.sqrt(0.5 / float(self.N1) / float(self.N2) * sum_expr_sq)
                self.log("E(p^j) = " + str(o_energy), priority=1)
            else:
                norm_diff_p = None

            #Distribute energy on all ranks
            o_energy = self.collect_data_from_rank(o_energy, 0)
            
            #Actual criteria
            norm_diff_energy = np.abs(o_energy - old_o_energy)
            if norm_diff_energy < self.outer_cauchy_thresh:
                outer_criteria = True
                o_criteria_info = "Terminated by Cauchy Criterion. " + \
                        "|E(p^{j+1})-E(p^j)| = " + str(norm_diff_energy)
        elif self.outer_stop_criteria == "norm_diff_p":
            assert pj_m1 != None, "pj_m1 must be provided for self.outer_stop_criteria == ""norm_diff_p""."
            #Create non-overlapping DD as subsets of DD
            part = self.dd.get_partition_with_fixed_block(0,0)
            norm_diff_p_part = 0.0
            for m2,m1 in self.dd.relevant_domains[self.rank]:
                #boundaries of non-overl DD locally in overl DD
                lbl = part.lbounds[m1] - self.dd.lbounds[m1]
                lbr = part.rbounds[m1] - self.dd.lbounds[m1]
                lbu = part.ubounds[m2] - self.dd.ubounds[m2]
                lbb = part.bbounds[m2] - self.dd.ubounds[m2]
                norm_diff_p_part += np.linalg.norm( \
                        pj[(m2,m1)][:,:,lbu:lbb,lbl:lbr] - pj_m1[(m2,m1)][:,:,lbu:lbb,lbl:lbr]) ** 2
            self.mpi_comm.Barrier()
            sum_collection = self.collect_data_at_rank(norm_diff_p_part, 0)
            if self.rank == 0:
                norm_diff_p = np.sum(sum_collection) \
                            / np.sqrt(4 * self.N1 * self.N2)
                self.log("|p^{j+1}-p^j| = " + str(norm_diff_p), priority=1)
            else:
                norm_diff_p = None
            norm_diff_p = self.collect_data_from_rank(norm_diff_p, 0)
            if norm_diff_p < self.outer_cauchy_thresh:
                outer_criteria = True
                o_criteria_info = "Terminated by norm_diff_p-Criterion. " + \
                        "|p^{j+1}-p^j| = " + str(norm_diff_p)
        return outer_criteria, o_criteria_info, o_energy
            

    def create_outer_record_transform(self, P_tau0 : ParallelArray) -> Callable[[ParallelArray], List[np.ndarray]]:
        """
        Returns function that transforms (outer loop) p into a value that 
        makes sense to record.
        """
        def tf_record_outer_transform(p : ParallelArray) -> List[np.ndarray]:
            #get Pdivp from p
            dx_p_l, dy_p_l = self.global_back_diff( \
                    [p[0,0,:,:], p[1,0,:,:]], [p[0,1,:,:], p[1,1,:,:]])
            divp = (dx_p_l[0] + dy_p_l[0]).stack( \
                    dx_p_l[1] + dy_p_l[1], axis=0)
            Pdivp = self.proj_div_0_compl(divp)

            #Final primal result tau of tangent field smoothing
            tau = P_tau0 - Pdivp * self.delta
            tau_dep = deparallelize([tau[0,:,:], tau[1,:,:]], \
                                    self.dd, self.mpi_comm)
            return [tau_dep[0], tau_dep[1]]
        return tf_record_outer_transform


    def create_inner_record_transform(self, omega0: ParallelArray, m1 : int, m2 : int) \
                        -> Callable[[np.ndarray], List[np.ndarray]]:
        """
        Returns function that transforms (inner loop) v into a value that 
        makes sense to record.
        """
        def tf_record_inner_transform(vn: np.ndarray) -> List[np.ndarray]:
            drb = True if m1 == self.dd.M1 - 1 else False 
            dbb = True if m2 == self.dd.M2 - 1 else False 
            ext_r = 0 if drb else 1
            ext_b = 0 if dbb else 1
            shape_ext_m2_m1 = (2, self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
            div_vn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
            pd0lcb.divB(vn[0,:,:,:], div_vn[0,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            pd0lcb.divB(vn[1,:,:,:], div_vn[1,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            P_div_vn = self.proj_div_0_sparse(div_vn, m1, m2, 1, 1)
            tau = omega0[(m2,m1)] - P_div_vn
            return [tau[0], tau[1]]

        return tf_record_inner_transform


    def run(self, tau0 : ParallelArray) -> ParallelArray:
        """
        Execute Tangent Field Smoothing.
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        tau0 is delivered only on these domains.
        """
        doms = self.dd.relevant_domains[self.rank]
        
        #Compute Projection of tau0 with one extra row and one extra column
        P_tau0_ext = self.proj_div_0_compl(tau0, 1, 1)
        #Also Extract P_tau0 from P_tau0_ext
        P_tau0 = ParallelArray({})
        for m2, m1 in doms:
            slice_rows = slice(None) if m2 == self.dd.M2 - 1 else slice(None, -1)
            slice_cols = slice(None) if m1 == self.dd.M1 - 1 else slice(None, -1)
            P_tau0[(m2, m1)] = P_tau0_ext[(m2, m1)][:,slice_rows,slice_cols]
        tau0tild_ext = P_tau0_ext * (1. / self.delta)

        #for recording outer loops
        record_outer_transform = self.create_outer_record_transform(P_tau0)
        self.log("Tangent Field Smoothing START", priority=0)

        #Initialize pj and qjhat
        pjdic = {}
        qjhatdic = {}
        for m2,m1 in doms:
            pjdic[(m2, m1)] = np.zeros((2, 2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
            qjhatdic[(m2, m1)] = np.zeros((2, 2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
        pj = ParallelArray(pjdic)
        qjhat = ParallelArray(qjhatdic)

        #Outer loop of DD
        for j in range(self.num_outer_it_max):
            self.current_step = [j,0]
            self.log("Tfs OUTER iteration " + str(j) + " START", priority=1)
            P_div_q0 = self.compute_P_div_q0(pj)
            omega0 = tau0tild_ext - P_div_q0

            for m2, m1 in doms:
                record_inner_transform = self.create_inner_record_transform(omega0, m1, m2)
                
                #Help variables that tell if (m2, m1) is at right or bottom boundary
                drb = True if m1 == self.dd.M1 - 1 else False 
                dbb = True if m2 == self.dd.M2 - 1 else False 
                ext_r = 0 if drb else 1
                ext_b = 0 if dbb else 1
                shape_m2_m1 = (2, 2, self.dd.sizes2[m2], self.dd.sizes1[m1])
                shape_ext_m2_m1 = (2, self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
                
                #Inner loop of DD
                self.log("Tfs Inner iteration (" + str(j) + ",) for domain " + str((m2,m1)) + " START", priority=1)
                vn = copy.deepcopy(qjhat[(m2,m1)])
                for n in range(self.num_inner_it_max):
                    self.current_step = [j,n]
                    self.log("Tfs Inner iteration, domain " + str((m2,m1)) + " step " + str((j,n)) + " START", priority=2)
                    div_vn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
                    pd0lcb.divB(vn[0,:,:,:], div_vn[0,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    pd0lcb.divB(vn[1,:,:,:], div_vn[1,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    P_div_vn = self.proj_div_0_sparse(div_vn, m1, m2, 1, 1)
                    P_div_vn_m_omega0 = P_div_vn - omega0[(m2,m1)]
                    psin = np.zeros(shape_m2_m1, dtype=np.float64)
                    pd0lcb.gradF(P_div_vn_m_omega0[0,:,:], psin[0,:,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    pd0lcb.gradF(P_div_vn_m_omega0[1,:,:], psin[1,:,:,:], self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    theta = self.dd.theta_loc(m2, m1)
                    numerator00 = vn[0,0,:,:] * theta + self.k * theta * psin[0,0,:,:]
                    numerator01 = vn[0,1,:,:] * theta + self.k * theta * psin[0,1,:,:]
                    numerator10 = vn[1,0,:,:] * theta + self.k * theta * psin[1,0,:,:]
                    numerator11 = vn[1,1,:,:] * theta + self.k * theta * psin[1,1,:,:]
                    abs_psi_0 = np.sqrt(psin[0,0,:,:] ** 2 + psin[0,1,:,:] ** 2)
                    abs_psi_1 = np.sqrt(psin[1,0,:,:] ** 2 + psin[1,1,:,:] ** 2)
                    denominator_0 = theta + self.k * abs_psi_0
                    denominator_1 = theta + self.k * abs_psi_1
                    vn_m1 = copy.deepcopy(vn)
                    vn[0,0,:,:] = numerator00 / denominator_0
                    vn[0,1,:,:] = numerator01 / denominator_0
                    vn[1,0,:,:] = numerator10 / denominator_1
                    vn[1,1,:,:] = numerator11 / denominator_1

                    #save current (primal version of) vn
                    self.record_step(vn, ["tau1fromVn_" + str(m2) + "_" + str(m1), \
                                        "tau2fromVn_" + str(m2) + "_" + str(m1)],
                                        depth=2, operation=record_inner_transform, \
                                        force_rank=True)

                    #inner stop criteria
                    inner_criteria = False
                    if self.inner_stop_criteria == "Cauchy":
                        old_energy = energy if n > 0 else 1000000000.0
                        shp_y, shp_x = self.dd.sizes2[m2], self.dd.sizes1[m1]
                        scaling_factor = math.sqrt(0.5 / float(shp_y) / float(shp_x))
                        energy = scaling_factor * np.linalg.norm(P_div_vn_m_omega0)
                        self.log("E(v^n) = " + str(energy), priority=2)
                        if np.abs(old_energy - energy) < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by Cauchy-Criterion. " + \
                                    "E(v^n) = " + str(energy)
                    elif self.inner_stop_criteria == "norm_diff_v":
                        norm_diff_v = np.linalg.norm(vn - vn_m1) \
                            / np.sqrt(4 * self.dd.sizes1[m1] * self.dd.sizes2[m2])
                        self.log("|v^{n+1}-v^n| = " + str(norm_diff_v), priority=2)
                        if norm_diff_v < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by norm_diff_v-Criterion. " + \
                                    "|v^{n+1}-v^n| = " + str(norm_diff_v)
                    if inner_criteria:
                        qjhat[(m2,m1)] = copy.deepcopy(vn)
                        self.log("Tfs Inner iteration for domain " + str((m2,m1)) + " TERMINATED BY CRITERIA at iteration " + str((j,n)), priority=1)
                        self.log(criteria_info, priority=1)
                        break
                    if n == self.num_inner_it_max - 1:
                        qjhat[(m2,m1)] = copy.deepcopy(vn)
                        self.log("Tfs Inner iteration for domain " + str((m2,m1)) + " FINISHED after last iteration " + str((j,n)), priority=1)

            #Determine sum over the qjhat locally
            sum_qjhat = qjhat + self.additive_intersection_exchange( \
                                qjhat, num_channels1=2, num_channels2=2)
            pj_m1 = copy.deepcopy(pj)
            if (j == 0):
                pj = sum_qjhat
            else:
                pj = pj * (1. - self.alpha) + sum_qjhat * self.alpha
            
            #save current images
            self.record_step(pj, ["tau[0]", "tau[1]"], 1, record_outer_transform)
            
            #outer stop criteria
            old_o_energy = o_energy if j > 0 else 1000000000.0
            outer_criteria, o_criteria_info, o_energy = \
                self.evaluate_outer_criteria(pj, pj_m1, old_o_energy, P_tau0)
            if outer_criteria:
                self.log("Tfs OUTER iteration TERMINATED BY CRITERIA at iteration " + str(j) + ".", priority=0)
                self.log(o_criteria_info, priority=0)
                break
            if j == self.num_outer_it_max - 1:
                self.log("Tfs OUTER iteration FINISHED after last iteration " + str(j), priority=0)

        #get Pdivpj from pj
        dx_pj_l, dy_pj_l = self.global_back_diff( \
                [pj[0,0,:,:], pj[1,0,:,:]], [pj[0,1,:,:], pj[1,1,:,:]])
        divpj = (dx_pj_l[0] + dy_pj_l[0]).stack( \
                dx_pj_l[1] + dy_pj_l[1], axis=0)
        Pdivpj = self.proj_div_0_compl(divpj)

        #Final primal result tau of tangent field smoothing
        tau = P_tau0 - Pdivpj * self.delta

        self.log("Tangent Field Smoothing FINISHED", priority=0)

        return tau



class TRImageReconstruction(TRIntersectingDDAlgorithm):
    
    def __init__(self, _domain_decomposition : pa.ThreadedHUIP, \
                _ir_parameters : dict, _rec : Recorder, \
                _mpi_comm : MPI.Comm):
        super().__init__(_domain_decomposition, _ir_parameters, _rec, _mpi_comm)


    def initialize_parameters(self, ir_parameters : dict):   
        self.k = ir_parameters["k"]
        self.delta = ir_parameters["mu"]
        self.alpha = ir_parameters["alpha"]
        self.beta = ir_parameters["beta"]
        self.mu = ir_parameters["mu"]
        self.variant = ir_parameters["variant"]

        self.outer_stop_criteria = ir_parameters["outer_stop_criteria"]
        self.inner_stop_criteria = ir_parameters["inner_stop_criteria"]
        self.num_outer_it_max = ir_parameters["num_outer_it_max"]
        self.num_inner_it_max = ir_parameters["num_inner_it_max"]
        self.inner_cauchy_thresh = ir_parameters["inner_cauchy_thresh"]
        self.outer_cauchy_thresh = ir_parameters["outer_cauchy_thresh"]


    def evaluate_outer_criteria(self, rj : ParallelArray, \
                                rj_m1 : ParallelArray, \
                                old_o_energy : float = None, \
                                d0 : ParallelArray = None, \
                                xi : ParallelArray = None, \
                                u0tild : ParallelArray = None) \
                                -> Tuple[bool, str, float]:
        """
        Evaluate the stopping criteria of the outer Chambolle loop.
        Args:
          rj: current dual denoised image
          rj_m1: last dual denoised image
          old_o_energy: old outer energy
          xi (only Ir Variant 1): normalized normal field
          u0tild: (only Ir Variant 2): adapted data term
        """
        outer_criteria = False
        o_criteria_info = ""
        o_energy = None
        if self.outer_stop_criteria == "Cauchy":
            assert old_o_energy != None or P_tau0 != None, \
                "old_o_energy and P_tau0 must be provided for self.outer_stop_criteria == ""Cauchy""."
            if self.variant == 1:
                #get divrj from rj
                rj_xi = rj + xi
                dx_rj_xi, dy_rj_xi = self.global_back_diff( \
                        [rj_xi[0,:,:]], [rj_xi[1,:,:]])
                div_rj_xi = dx_rj_xi[0] + dy_rj_xi[0]
                #Expression within integral
                expr = div_rj_xi - d0 * (1.0 / self.mu)
            elif self.variant == 2:
                #get divrj from rj
                dx_r_l, dy_r_l = self.global_back_diff( \
                        [rj[0,:,:]], [rj[1,:,:]])
                divr = dx_r_l[0] + dy_r_l[0]
                #Expression within integral
                expr = divr - u0tild
            else:
                raise Exception("Ir Variant neither 1 nor 2.")

            #Sum squares everything up: Create non-overlapping DD as subsets of DD
            part = self.dd.get_partition_with_fixed_block(0,0)

            #Sum squares everything up: Create local sums
            sum_loc_expr_sq = 0.0
            for m2,m1 in self.dd.relevant_domains[self.rank]:
                #boundaries of non-overl DD locally in overl DD
                lbl = part.lbounds[m1] - self.dd.lbounds[m1]
                lbr = part.rbounds[m1] - self.dd.lbounds[m1]
                lbu = part.ubounds[m2] - self.dd.ubounds[m2]
                lbb = part.bbounds[m2] - self.dd.ubounds[m2]
                sum_loc_expr_sq += np.linalg.norm(expr[(m2,m1)][lbu:lbb,lbl:lbr]) ** 2
            self.mpi_comm.Barrier()
   
            #Collect sums from all ranks, add them together and determine energy
            sum_collection = self.collect_data_at_rank(sum_loc_expr_sq, 0)
            if self.rank == 0:
                sum_expr_sq = np.sum(sum_collection)
                o_energy = math.sqrt(1.0 / float(self.N1) / float(self.N2) * sum_expr_sq)
                self.log("E(r^j) = " + str(o_energy), priority=1)
            else:
                norm_diff_r = None

            #Distribute energy on all ranks
            o_energy = self.collect_data_from_rank(o_energy, 0)
            
            #Actual criteria
            norm_diff_energy = np.abs(o_energy - old_o_energy)
            if norm_diff_energy < self.outer_cauchy_thresh:
                outer_criteria = True
                o_criteria_info = "Terminated by Cauchy Criterion. " + \
                        "|E(r^{j+1})-E(r^j)| = " + str(norm_diff_energy)
        elif self.outer_stop_criteria == "norm_diff_r":
            #Create non-overlapping DD as subsets of DD
            part = self.dd.get_partition_with_fixed_block(0,0)
            norm_diff_r_part = 0.0
            for m2,m1 in self.dd.relevant_domains[self.rank]:
                #boundaries of non-overl DD locally in overl DD
                lbl = part.lbounds[m1] - self.dd.lbounds[m1]
                lbr = part.rbounds[m1] - self.dd.lbounds[m1]
                lbu = part.ubounds[m2] - self.dd.ubounds[m2]
                lbb = part.bbounds[m2] - self.dd.ubounds[m2]
                norm_diff_r_part += np.linalg.norm( \
                        rj[(m2,m1)][:,lbu:lbb,lbl:lbr] - rj_m1[(m2,m1)][:,lbu:lbb,lbl:lbr]) ** 2
            self.mpi_comm.Barrier()
            sum_collection = self.collect_data_at_rank(norm_diff_r_part, 0)
            if self.rank == 0:
                norm_diff_r = np.sum(sum_collection) \
                            / np.sqrt(4 * self.N1 * self.N2)
                self.log("|r^{j+1}-r^j| = " + str(norm_diff_r), priority=1)
            else:
                norm_diff_r = None
            norm_diff_r = self.collect_data_from_rank(norm_diff_r, 0)
            if norm_diff_r < self.outer_cauchy_thresh:
                outer_criteria = True
                o_criteria_info = "Terminated by norm_diff_r-Criterion. " + \
                        "|r^{j+1}-r^j| = " + str(norm_diff_r)
        return outer_criteria, o_criteria_info, o_energy
    


    def create_v1_outer_record_transform(self, d0 : ParallelArray, xi : ParallelArray) -> Callable[[ParallelArray], List[np.ndarray]]:
        """
        Returns function that transforms (outer loop) p into a value that 
        makes sense to record.
        This function can be used for Image Reconstruction Version 1.
        """
        def ir1_record_outer_transform(rj : ParallelArray) -> List[np.ndarray]:
            #get divrj from rj
            rj_xi = rj + xi
            dx_rj_xi, dy_rj_xi = self.global_back_diff( \
                    [rj_xi[0,:,:]], [rj_xi[1,:,:]])
            div_rj_xi = dx_rj_xi[0] + dy_rj_xi[0]

            #Final primal result d of image reconstruction
            d = d0 - div_rj_xi * self.mu
            
            d_dep = deparallelize([d], self.dd, self.mpi_comm)
            return [d_dep[0]]
        return ir1_record_outer_transform


    def create_v2_outer_record_transform(self, d0 : ParallelArray) -> Callable[[ParallelArray], List[np.ndarray]]:
        """
        Returns function that transforms (outer loop) p into a value that 
        makes sense to record.
        This function can be used for Image Reconstruction Version 2.
        """
        def ir2_record_outer_transform(r : ParallelArray) -> List[np.ndarray]:
            #get divrj from rj
            dx_r_l, dy_r_l = self.global_back_diff( \
                    [r[0,:,:]], [r[1,:,:]])
            divr = dx_r_l[0] + dy_r_l[0]

            #Final primal result d of image reconstruction
            d = d0 - divr * (1. / self.beta)
            
            d_dep = deparallelize([d], self.dd, self.mpi_comm)
            return [d_dep[0]]
        return ir2_record_outer_transform
    
    
    def create_v1_inner_record_transform(self, v0m : np.ndarray, m1 : int, m2 : int) \
                        -> Callable[[np.ndarray], List[np.ndarray]]:
        """
        Returns function that transforms (inner loop) v into a value that 
        makes sense to record. Used for version 1 of Image Reconstruction.
        """
        def ir1_record_inner_transform(wn: np.ndarray) -> List[np.ndarray]:
            drb = True if m1 == self.dd.M1 - 1 else False 
            dbb = True if m2 == self.dd.M2 - 1 else False 
            ext_r = 0 if drb else 1
            ext_b = 0 if dbb else 1
            shape_ext_m2_m1 = (self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
            div_wn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
            pd0lcb.divB(wn, div_wn, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            d = self.mu * (v0m - div_wn)
            return [d]

        return ir1_record_inner_transform


    def create_v2_inner_record_transform(self, v0m: ParallelArray, m1 : int, m2 : int) \
                        -> Callable[[np.ndarray], List[np.ndarray]]:
        """
        Returns function that transforms (inner loop) v into a value that 
        makes sense to record. Used for version 2 of Image Reconstruction.
        """
        def ir2_record_inner_transform(wn: np.ndarray) -> List[np.ndarray]:
            drb = True if m1 == self.dd.M1 - 1 else False 
            dbb = True if m2 == self.dd.M2 - 1 else False 
            ext_r = 0 if drb else 1
            ext_b = 0 if dbb else 1
            shape_ext_m2_m1 = (self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
            div_wn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
            pd0lcb.divB(wn, div_wn, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
            d = (1. / self.beta) * (v0m - div_wn)
            return [d]

        return ir2_record_inner_transform


    def run_v1(self, d0 : ParallelArray, xi : ParallelArray) -> ParallelArray:
        """
        Execute Image Reconstruction Variant 1.
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        d0 and xi are delivered only on these domains.
        """
        doms = self.dd.relevant_domains[self.rank]
        
        #Data term for Chambolle iteration
        d0tild = d0 / self.mu
        dx_list, dy_list = self.global_back_diff([xi[0,:,:]],[xi[1,:,:]])
        dx_xi1, dy_xi2 = dx_list[0], dy_list[0]
        div_xi = dx_xi1 + dy_xi2
        d0_div_xi = d0tild - div_xi
        d0_div_xi_ext = self.collect_halo(d0_div_xi)

        #for recording outer loops
        record_outer_transform = self.create_v1_outer_record_transform(d0, xi)
        self.log("Image Reconstruction Variant 1 START", priority=0)

        #Initialize dual variable rj, weights theta and help variables
        rjdic = {}
        thetadic = {}
        thetarjdic = {}
        tjhatdic = {}
        self.dd.load_relevant_theta(self.rank)
        for m2,m1 in doms:
            rjdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
            thetadic[(m2, m1)] = self.dd.theta_loc(m2,m1)
            thetarjdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
            tjhatdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
        rj = ParallelArray(rjdic)
        theta = ParallelArray(thetadic)
        theta_rj = ParallelArray(thetarjdic)
        tjhat = ParallelArray(tjhatdic)

        #Outer loop of DD
        for j in range(self.num_outer_it_max):
            self.current_step = [j,0]
            self.log("Ir1 OUTER iteration " + str(j) + " START", priority=1)
            theta_rj[0,:,:] = rj[0,:,:] * theta
            theta_rj[1,:,:] = rj[1,:,:] * theta
            tm0_ext = self.additive_intersection_exchange(theta_rj, num_channels1=2,extended_output=True)

            for m2, m1 in doms:
                #Evaluate vm0 ("data term" in inner iteration)
                div_tm0 = self.divergence_from_extended_array(tm0_ext[(m2,m1)], m2, m1)
                v0m = d0_div_xi_ext[(m2,m1)] - div_tm0
                
                record_inner_transform = self.create_v1_inner_record_transform(v0m, m1, m2)

                #Help variables that tell if (m2, m1) is at right or bottom boundary
                drb = True if m1 == self.dd.M1 - 1 else False 
                dbb = True if m2 == self.dd.M2 - 1 else False 
                ext_r = 0 if drb else 1
                ext_b = 0 if dbb else 1
                shape_m2_m1 = (2, self.dd.sizes2[m2], self.dd.sizes1[m1])
                shape_ext_m2_m1 = (self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
                
                #Inner loop of DD
                self.log("Ir1 Inner iteration (" + str(j) + ",) for domain " + str((m2,m1)) + " START", priority=1)
                wn = copy.deepcopy(tjhat[(m2,m1)])
                for n in range(self.num_inner_it_max):
                    self.current_step = [j,n]
                    self.log("Ir1 Inner iteration, domain " + str((m2,m1)) + " step " + str((j,n)) + " START", priority=2)
                    div_wn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
                    pd0lcb.divB(wn, div_wn, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    div_wn_tm0_xi_d0 = div_wn - v0m
                    rhon = np.zeros(shape_m2_m1, dtype=np.float64)
                    pd0lcb.gradF(div_wn_tm0_xi_d0, rhon, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    theta_loc = theta[(m2,m1)]
                    numerator0 = wn[0,:,:] * theta_loc + self.k * theta_loc * rhon[0,:,:]
                    numerator1 = wn[1,:,:] * theta_loc + self.k * theta_loc * rhon[1,:,:]
                    abs_rho = np.sqrt(rhon[0,:,:] ** 2 + rhon[1,:,:] ** 2)
                    denominator = theta_loc + self.k * abs_rho
                    wn_m1 = copy.deepcopy(wn)
                    wn[0,:,:] = numerator0 / denominator
                    wn[1,:,:] = numerator1 / denominator

                    #save current (primal version of) wn
                    self.record_step(wn, ["dFromWn_" + str(m2) + "_" + str(m1)],
                                        depth=2, operation=record_inner_transform, \
                                        force_rank=True)

                    #inner stop criteria
                    inner_criteria = False
                    if self.inner_stop_criteria == "Cauchy":
                        old_energy = energy if n > 0 else 1000000000.0
                        shp_y, shp_x = self.dd.sizes2[m2], self.dd.sizes1[m1]
                        scaling_factor = math.sqrt(1.0 / float(shp_y) / float(shp_x))
                        energy = scaling_factor * np.linalg.norm(div_wn_tm0_xi_d0)
                        self.log("E(w^n) = " + str(energy), priority=2)
                        if np.abs(old_energy - energy) < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by Cauchy-Criterion. " + \
                                    "E(w^n) = " + str(energy)
                    elif self.inner_stop_criteria == "norm_diff_w":
                        norm_diff_w = np.linalg.norm(wn - wn_m1) \
                            / np.sqrt(4 * self.dd.sizes1[m1] * self.dd.sizes2[m2])
                        self.log("|w^{n+1}-w^n| = " + str(norm_diff_w), priority=2)
                        if norm_diff_w < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by norm_diff_w-Criterion. " + \
                                    "|w^{n+1}-w^n| = " + str(norm_diff_w)
                    if inner_criteria:
                        tjhat[(m2,m1)] = copy.deepcopy(wn)
                        self.log("Ir1 Inner iteration for domain " + str((m2,m1)) + " TERMINATED BY CRITERIA at iteration " + str((j,n)), priority=1)
                        self.log(criteria_info, priority=1)
                        break
                    if n == self.num_inner_it_max - 1:
                        tjhat[(m2,m1)] = copy.deepcopy(wn)
                        self.log("Ir1 Inner iteration for domain " + str((m2,m1)) + " FINISHED after last iteration " + str((j,n)), priority=1)               

                        
            #Determine sum over the tjhat locally
            sum_tjhat = tjhat + self.additive_intersection_exchange( \
                                tjhat, num_channels1=2)
            rj_m1 = copy.deepcopy(rj)
            if (j == 0):
                rj = sum_tjhat
            else:
                rj = rj * (1. - self.alpha) + sum_tjhat * self.alpha

            #save current images
            self.record_step(rj, ["d"], 1, record_outer_transform)

            #outer stop criteria
            old_o_energy = o_energy if j > 0 else 1000000000.0
            outer_criteria, o_criteria_info, o_energy = \
                        self.evaluate_outer_criteria(rj, rj_m1, old_o_energy, d0, xi)

            if outer_criteria:
                self.log("Ir1 OUTER iteration TERMINATED BY CRITERIA at iteration " + str(j),priority=0)
                self.log(o_criteria_info, priority=0)
                break
            if j == self.num_outer_it_max - 1:
                self.log("Ir1 OUTER iteration FINISHED after last iteration " + str(j), priority=0)

        #get divrj from rj
        rj_xi = rj + xi
        dx_rj_xi, dy_rj_xi = self.global_back_diff( \
                [rj_xi[0,:,:]], [rj_xi[1,:,:]])
        div_rj_xi = dx_rj_xi[0] + dy_rj_xi[0]

        #Final primal result d of image reconstruction
        d = d0 - div_rj_xi * self.mu
        self.log("Image Reconstruction Variant 1 FINISHED", priority=0)

        return d
    

    def run_v2(self, d0 : ParallelArray, g : ParallelArray) -> ParallelArray:
        """
        Execute Image Reconstruction Variant 2.
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        d0 and g are delivered only on these domains.
        """
        doms = self.dd.relevant_domains[self.rank]

        #Data term for Chambolle iteration
        u0tild = (d0 - g) * self.beta
        u0tild_ext = self.collect_halo(u0tild)
        #for recording outer loops
        record_outer_transform = self.create_v2_outer_record_transform(d0)
        self.log("Image Reconstruction Variant 2 START", priority=0)

        #Initialize dual variable rj, weights theta and help variables
        rjdic = {}
        thetadic = {}
        thetarjdic = {}
        tjhatdic = {}
        self.dd.load_relevant_theta(self.rank)
        for m2,m1 in doms:
            rjdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
            thetadic[(m2, m1)] = self.dd.theta_loc(m2,m1)
            thetarjdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
            tjhatdic[(m2, m1)] = np.zeros((2, self.dd.sizes2[m2], self.dd.sizes1[m1]),dtype=np.float64)
        rj = ParallelArray(rjdic)
        theta = ParallelArray(thetadic)
        theta_rj = ParallelArray(thetarjdic)
        tjhat = ParallelArray(tjhatdic)

        #Outer loop of DD
        for j in range(self.num_outer_it_max):
            self.current_step = [j,0]
            self.log("Ir2 OUTER iteration " + str(j) + " START", priority=1)
            theta_rj[0,:,:] = rj[0,:,:] * theta
            theta_rj[1,:,:] = rj[1,:,:] * theta
            tm0_ext = self.additive_intersection_exchange(theta_rj, num_channels1=2,extended_output=True)

            for m2, m1 in doms:
                #Evaluate vm0 ("data term" in inner iteration)
                div_tm0 = self.divergence_from_extended_array(tm0_ext[(m2,m1)], m2, m1)
                v0m = u0tild_ext[(m2,m1)] - div_tm0

                record_inner_transform = self.create_v2_inner_record_transform(v0m, m1, m2)

                #Help variables that tell if (m2, m1) is at right or bottom boundary
                drb = True if m1 == self.dd.M1 - 1 else False 
                dbb = True if m2 == self.dd.M2 - 1 else False 
                ext_r = 0 if drb else 1
                ext_b = 0 if dbb else 1
                shape_m2_m1 = (2, self.dd.sizes2[m2], self.dd.sizes1[m1])
                shape_ext_m2_m1 = (self.dd.sizes2[m2] + ext_b, self.dd.sizes1[m1] + ext_r)
                
                #Inner loop of DD
                self.log("Ir2 Inner iteration (" + str(j) + ",) for domain " + str((m2,m1)) + " START", priority=1)
                wn = copy.deepcopy(tjhat[(m2,m1)])
                for n in range(self.num_inner_it_max):
                    self.current_step = [j,n]
                    self.log("Ir2 Inner iteration, domain " + str((m2,m1)) + " step " + str((j,n)) + " START", priority=2)
                    div_wn = np.zeros(shape_ext_m2_m1, dtype=np.float64)
                    pd0lcb.divB(wn, div_wn, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    div_wn_v0m = div_wn - v0m
                    rhon = np.zeros(shape_m2_m1, dtype=np.float64)
                    pd0lcb.gradF(div_wn_v0m, rhon, self.dd.sizes1[m1], self.dd.sizes2[m2], drb, dbb, self.h)
                    theta_loc = theta[(m2,m1)]
                    numerator0 = wn[0,:,:] * theta_loc + self.k * theta_loc * rhon[0,:,:]
                    numerator1 = wn[1,:,:] * theta_loc + self.k * theta_loc * rhon[1,:,:]
                    abs_rho = np.sqrt(rhon[0,:,:] ** 2 + rhon[1,:,:] ** 2)
                    denominator = theta_loc + self.k * abs_rho
                    wn_m1 = copy.deepcopy(wn)
                    wn[0,:,:] = numerator0 / denominator
                    wn[1,:,:] = numerator1 / denominator

                    #save current (primal version of) wn
                    self.record_step(wn, ["dFromWn_" + str(m2) + "_" + str(m1)],
                                        depth=2, operation=record_inner_transform, \
                                        force_rank=True)

                    #inner stop criteria
                    inner_criteria = False
                    if self.inner_stop_criteria == "Cauchy":
                        old_energy = energy if n > 0 else 1000000000.0
                        shp_y, shp_x = self.dd.sizes2[m2], self.dd.sizes1[m1]
                        scaling_factor = math.sqrt(1.0 / float(shp_y) / float(shp_x))
                        energy = scaling_factor * np.linalg.norm(div_wn_v0m)
                        self.log("E(w^n) = " + str(energy), priority=2)
                        if np.abs(old_energy - energy) < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by Cauchy-Criterion. " + \
                                    "E(w^n) = " + str(energy)
                    elif self.inner_stop_criteria == "norm_diff_w":
                        norm_diff_w = np.linalg.norm(wn - wn_m1) \
                            / np.sqrt(4 * self.dd.sizes1[m1] * self.dd.sizes2[m2])
                        self.log("|w^{n+1}-w^n| = " + str(norm_diff_w), priority=2)
                        if norm_diff_w < self.inner_cauchy_thresh:
                            inner_criteria = True
                            criteria_info = "Terminated by norm_diff_w-Criterion. " + \
                                    "|w^{n+1}-w^n| = " + str(norm_diff_w)
                    if inner_criteria:
                        tjhat[(m2,m1)] = copy.deepcopy(wn)
                        self.log("Ir2 Inner iteration for domain " + str((m2,m1)) + " TERMINATED BY CRITERIA at iteration " + str((j,n)),priority=1)
                        self.log(criteria_info, priority=1)
                        break
                    if n == self.num_inner_it_max - 1:
                        tjhat[(m2,m1)] = copy.deepcopy(wn)
                        self.log("Ir2 Inner iteration for domain " + str((m2,m1)) + " FINISHED after last iteration " + str((j,n)), priority=1)

            #Determine sum over the tjhat locally
            sum_tjhat = tjhat + self.additive_intersection_exchange( \
                                tjhat, num_channels1=2)
            rj_m1 = copy.deepcopy(rj)
            if (j == 0):
                rj = sum_tjhat
            else:
                rj = rj * (1. - self.alpha) + sum_tjhat * self.alpha
            
            #save current images
            self.record_step(rj, ["d"], 1, record_outer_transform)

            #outer stop criteria
            old_o_energy = o_energy if j > 0 else 1000000000.0
            outer_criteria, o_criteria_info, o_energy = \
                    self.evaluate_outer_criteria(rj, rj_m1, old_o_energy, u0tild=u0tild)

            if outer_criteria:
                self.log("Ir2 OUTER iteration TERMINATED BY CRITERIA at iteration " + str(j),priority=0)
                self.log(o_criteria_info, priority=0)
                break
            if j == self.num_outer_it_max - 1:
                self.log("Ir2 OUTER iteration FINISHED after last iteration " + str(j), priority=0)

        #get divrj from rj
        dx_rj_l, dy_rj_l = self.global_back_diff( \
                [rj[0,:,:]], [rj[1,:,:]])
        divrj = dx_rj_l[0] + dy_rj_l[0]

        #Final primal result d of image reconstruction
        d = d0 - divrj * (1. / self.beta)
        self.log("Image Reconstruction Variant 2 FINISHED", priority=0)
        return d
    
        



class TvStokesDualDD(TRIntersectingDDAlgorithm):
    
    def __init__(self, _domain_decomposition : pa.ThreadedHUIP, \
                _tvsd_parameters : dict, _rec : Recorder, _mpi_comm : MPI.Comm, \
                _include_tfs : bool = True, _include_ir : bool = True):
        self.use_tfs = _include_tfs
        self.use_ir = _include_ir        
        super().__init__(_domain_decomposition, _tvsd_parameters, _rec, _mpi_comm)


    def initialize_parameters(self, tvsd_parameters : dict):
        self.h = tvsd_parameters["h"]
        if self.use_tfs:
            self.tfs_par = tvsd_parameters["tangent_field_smoothing"]
            tfs_rec = copy.copy(self.rec)
            if tfs_rec.name_prefix != "": tfs_rec.name_prefix += "_"
            tfs_rec.name_prefix += "tfs"
            #save extended (N2+1)x(N1+1)-partition in Tfs-object
            tfs_dd = self.dd.duplicate_with_additional_row_and_col()
            self.tfs = TRTangentFieldSmoothing(tfs_dd, self.tfs_par, tfs_rec, self.mpi_comm)
            self.tfs.h = self.h
        if self.use_ir:
            self.eps = tvsd_parameters["eps"]
            self.ir_par = tvsd_parameters["image_reconstruction"]
            self.ir_variant = self.ir_par["variant"]
            ir_rec = copy.copy(self.rec)
            if ir_rec.name_prefix != "": ir_rec.name_prefix += "_"
            ir_rec.name_prefix += "ir"
            self.ir = TRImageReconstruction(self.dd, self.ir_par, ir_rec, self.mpi_comm)
            self.ir.h = self.h
            

    def get_deparallelizer(self) -> Callable[[List[ParallelArray]], List[np.ndarray]]:
        """
        Returns function that transforms a list of ParallelArray's
        into a list of ndarray's.
        """
        def deparallelize_list(arrs : List[ParallelArray]) -> List[np.ndarray]:
            arrs_dep = deparallelize(arrs, self.dd, self.mpi_comm)
            return arrs_dep
        return deparallelize_list


    def run_tfs_only(self, d0 : ParallelArray, return_tau_ext = False) -> ParallelArray:
        """
        Execute only Tangent Field Smoothing of TV Stokes algorithm
        on noisy image d0. 
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        d0 is delivered only on these domains.
        Returns: Tangent Field tau
         """
        assert self.use_tfs == True, \
                "Tangent Field Smoothing got deactivated during instantiation of this class. " \
                + "Set _include_tfs=True when instantiating this class."
        dx_d0_ext, dy_d0_ext = self.global_back_diff_ext([d0], [d0])
        tau0_1_ext, tau0_2_ext = dy_d0_ext[0] * (-1.0), dx_d0_ext[0]
        tau0_ext = tau0_1_ext.stack(tau0_2_ext, axis=0)
        tau0_1 = tau0_1_ext.clip_last_row_and_column(self.dd)
        tau0_2 = tau0_2_ext.clip_last_row_and_column(self.dd)
        deparallelizer = self.get_deparallelizer()
        self.record_step([d0, tau0_1, tau0_2], ["d0", "tau01", "tau02"], depth=0, operation=deparallelizer)
        #Tangent field Smoothing
        tau_ext = self.tfs.run(tau0_ext)
        tau = tau_ext.clip_last_row_and_column(self.dd, dim=3)
        self.record_step([tau[0,:,:], tau[1,:,:]], ["tau1_final", "tau2_final"], depth=0, operation=deparallelizer)
        if return_tau_ext == False:
            return tau
        else:
            return tau_ext


    def run_ir_only(self, tau : ParallelArray, d0 : ParallelArray) -> Tuple[ParallelArray, ParallelArray]:
        """
        Execute only Image Reconstruction of TV Stokes algorithm 
        on predetermined Tangent Field tau. The noisy image d0 is 
        required too. Whether Image Reconstruction Version 1 or 2 
        is used is specified in the config that is passed when 
        instantiating this class.
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        tau and d0 is delivered only on these domains.
        Returns:
        Tuple (d, g) of
        - d: Reconstructed denoised image
        - g: Integrated Tangent Field (in case of ir_variant==2, else None)
        """
        assert self.use_ir == True, \
                "Image Reconstruction got deactivated during instantiation of this class. " \
                + "Set _include_ir=True when instantiating this class."
        if (self.ir_variant == 1):
            xi_den = ParallelArray.sqrt( \
                    tau[0,:,:] * tau[0,:,:] + \
                    tau[1,:,:] * tau[1,:,:] + self.eps)
            xi = copy.deepcopy(tau)
            xi[0,:,:] = tau[1,:,:] / xi_den
            xi[1,:,:] = (tau[0,:,:] / xi_den) * (-1.0)
            deparallelizer = self.get_deparallelizer()
            self.record_step([xi[0,:,:], xi[1,:,:]], ["xi1", "xi2"], depth=0, operation=deparallelizer)
            #Image Reconstruction, Version 1
            d = self.ir.run_v1(d0, xi)
            self.record_step([d], ["d_final"], depth=0, operation=deparallelizer)
            return d, None
        elif (self.ir_variant == 2):
            n1 = tau[1,:,:]
            n2 = tau[0,:,:] * (-1.0)
            g = self.find_backdiff_scalar_potential(n1, n2)
            deparallelizer = self.get_deparallelizer()
            self.record_step([g], ["g"], depth=0, operation=deparallelizer)
            #Image Reconstruction, Version 2
            d = self.ir.run_v2(d0, g)
            self.record_step([d], ["d_final"], depth=0, operation=deparallelizer)
            return d, g
        else:
            raise NotImplementedError("TvStokesDualDD: Image Reconstruction"
                    "is not implemented for self.ir_variant == " + str(self.ir_variant))


    def run(self, d0 : ParallelArray) -> Tuple[ParallelArray, ParallelArray, ParallelArray]:
        """
        Execute complete TV Stokes algorithm on noisy image d0.
        This thread is only responsible for the domains listed in
        self.dd.relevant_domains.
        d0 is delivered only on these domains.
        Returns:
        Tuple (d, tau, g) of
        - d: Reconstructed denoised image
        - tau: Tangent Field
        - g: Integrated Tangent Field (in case of ir_variant==2, else None)
        """
        assert self.use_tfs == True, \
                "Tangent Field Smoothing got deactivated during instantiation of this class. " \
                + "Use either ""run_ir_only(...)"" instead of ""run(...)"" or set _include_tfs=True when instantiating this class."
        assert self.use_ir == True, \
                "Image Reconstruction got deactivated during instantiation of this class. " \
                + "Use either ""run_tfs_only(...)"" instead of ""run(...)"" or set _include_ir=True when instantiating this class."
        
        tau = self.run_tfs_only(d0)
        d, g = self.run_ir_only(tau, d0)

        return d, tau, g
        
