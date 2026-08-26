
#Add this path to PYTHONPATH to make packages of this folder accessible from python
CURRENTPATH=${PWD}
export PYTHONPATH="${CURRENTPATH}"

#Add this line to ~/.bashrc (if not already added) for future purposes
LINE="export PYTHONPATH=${CURRENTPATH}"
FILE="${HOME}/.bashrc"
grep -xF -- "$LINE" "$FILE" || echo "$LINE" >> "$FILE"

#Build Cython-files for tv_stokes_dual
pushd tv_stokes_dual
  python3 setup.py build_ext --inplace
popd


#Uncomment to run tests
pushd tv_stokes_dual
  #python3 test_proj_div_0.py
  #python3 test_proj_div_0_low_cap.py
  #mpirun --hostfile hostfile -np 2 python par_tests.py
popd
