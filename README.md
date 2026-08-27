***************************************
Marc Runft, Lund, 2025-04-14
***************************************

Please cite this paper if you use this code in your research:
https://arxiv.org/abs/2602.17494

This package contains two different versions
of the TV-Stokes-algorithm: 
- A simple one (non-parallel)
- A parallel Domain-Decomposition one, evaluated locally 
  on different threads.


1) SETUP

Make sure that all libraries written in "requirements.txt"
are installed with compatible versions.
This package has only been tested on Ubuntu 20.04 with
Python 3.8.10 64-bit.
Both implementations contain Cython-files. To make sure
they are built properly, run setup_all.sh.
To rebuild the files, run clear_all.sh first (if the
files were not built for Python 3.8.10 64-bit, that
shell file might have to be adapted).


2) RUN A BASIC EXAMPLE FOR THE SIMPLE DUAL TV-STOKES-IMPLEMENATION

To test if the simple TV-Stokes-implementation works,
simply run:
python example_tv_stokes_dual.py
The parameters for this example can be adapted in the configuration
file:
tv_stokes_dual/config_tvsd/example_config.yaml
For a description of the parameters see comments in the file.


3) RUN A BASIC EXAMPLE FOR THE PARALLEL DUAL DD-TV-STOKES

In order to test the parallel DD algorithm, MPI must be used.
You need to decide in advance, how many threads you want to use.
If the number of threads is lower than the number of domains,
some threads will take care of multiple domains.
So, if you want to run the example-file with 4 threads, run:
mpirun --hostfile hostfile -np 4 python example_tv_stokes_dual_dd.py
Make sure that the number of threads in the hostfiles
-hostfile
-tv_stokes_dual/hostfile 
is at least 4 (or more if you want more threads).
The parameters of this example (including the number of domains)
can be adapted in the configuration file:
tv_stokes_dual/config_tvsd_dd/example_config.yaml
For a description of the parameters see comments in the file.


4) STRUCTURE OF SIMPLE DUAL TV-STOKES

The main file for the simple version is 
tv_stokes_dual/tvsd.py.
The example file example_tv_tokes_dual.py shows how to call it.
The example images are taken from the resources-folder and the
outputs (recordings) will be saved in the recordings-folder.
Both can be adapted though.
The main file tvsd.py relies on the Cython-file proj_div_0.pyx,
which contains important functions like differential operators,
the projection operator on div(...)=0 and the integration function
to gain the scalar function g from the tangent field tau.


5) STRUCTURE OF PARALLEL DUAL DD-TV-STOKES

The main file for the local parallel DD-version is
tv_stokes_dual/tvsd_dd.py
The example file example_tv_tokes_dual_dd.py shows how to call it.
The example images are taken from the resources-folder and the
outputs (recordings) will be saved in the recordings-folder.
Both can be adapted though.
The main file tvsd_dd.py relies on the Python files partition.py 
and proj_div_0_low_cap.py. The file partition.py provides classes
to conveniently handle Domain Decompositions (overlapping and 
non-overlapping). The file proj_div_0_low_cap.py can be seen as 
interface to the Cython-file proj_div_low_cap_base.pyx,
which contains the projection operator on div(...)=0, but adapted
in a way that in can be called locally. Since the structure of 
those Cython-functions is complicated, it is recommended to only
use proj_div_0_low_cap.py.


6) TEST FILES

Furthermore, in the folder tv_stokes_dual, there are the test files
test_proj_div_0.py, test_proj_div_0_low_cap.py and par_tests.py.
They are only there to validate (and proof the correctness) of the
implementations. They can be executed by uncommenting the
corresponding lines in setup_all.sh.
