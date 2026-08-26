
import cv2
import datetime
import math
import numpy as np
import os
import time
import yaml
from mpi4py import MPI

import tv_stokes_dual.tvsd_dd as tvsd_dd
from tv_stokes_dual.tvsd_dd import parallelize, deparallelize
import tv_stokes_dual.partition as part


"""
Execute the file with the following command.
Adapt the number of processes. Adapt hostfile if more processes needed.

mpirun --hostfile hostfile -np 4 python example_tv_stokes_dual_dd.py
"""

def create_dd(dd_parameters, N1, N2, num_threads):
    #number of domains in x-direction (M1) and y-direction (M2)
    M1 = dd_parameters["M1"]    
    M2 = dd_parameters["M2"]    
    #overlap size of domains
    overlap_x = dd_parameters["overlap_x"]
    overlap_y = dd_parameters["overlap_y"]
    dd = part.ThreadedHUIP.create_esized_edistributed( \
        N1, N2, M1, M2, overlap_x, overlap_y, num_threads)
    return dd


def prepare_rec(tvsd_parameters):
    #prepare parameters for Recorder object
    rec_parameters = tvsd_parameters["recorder"]
    name_prefix = ""
    print_details = rec_parameters["print_details"]
    record_steps = rec_parameters["record_steps"]
    log_details = rec_parameters["log_details"]
    time_now = datetime.datetime.now()
    recordings_folder_name = 'tvsd_dd_par_' + time_now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    if not(os.path.exists("recordings")):
        os.mkdir("recordings")
    os.chdir("recordings")
    if not(os.path.exists(recordings_folder_name)):
        os.mkdir(recordings_folder_name)
    os.chdir("..")
    recordings_path = os.getcwd() + '/recordings/' + recordings_folder_name + '/'
    rec_param = recordings_path, name_prefix, \
                     print_details, log_details, record_steps

    return rec_param



def main():
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    if (rank == 0):
        #prepare test image
        test_img_number = 2
        test_images = ["phantom_al.png", "phantom_mr.png", \
                        "beach_nz_1_480.jpg"]
        ground_truth = cv2.imread("resources/" + test_images[test_img_number], cv2.IMREAD_GRAYSCALE)
        ground_truth = ground_truth.astype("float64") / 255
        with open('tv_stokes_dual/config_tvsd_dd/example_config.yaml', 'r') as file:
            tvsd_parameters = yaml.safe_load(file)
        noise_var = 0.01
        gaussian = np.random.normal(loc=0, scale=math.sqrt(noise_var), size=ground_truth.shape)
        noisy_img = np.clip(ground_truth + gaussian, 0.0, 1.0, dtype=np.float64)
        #Domain Decomposition
        dd_parameters = tvsd_parameters["domain_decomposition"]
        dd_obj = create_dd(dd_parameters, noisy_img.shape[1], noisy_img.shape[0], num_ranks)
        #prepare recording parameters
        rec_param = prepare_rec(tvsd_parameters)
    else:
        noisy_img, dd_obj, tvsd_parameters, rec_param = None, None, None, None

    #from now on consider only parallel images
    additional_data = dd_obj, tvsd_parameters, rec_param
    noisy_img_parr, additional_data = \
            parallelize(noisy_img, dd_obj, comm, additional_data)
    dd_obj, tvsd_parameters, rec_param = additional_data
    
    #create instance of Recorder-class
    recordings_path, name_prefix, print_details, \
            log_details, record_steps = rec_param
    rec_obj = tvsd_dd.Recorder(recordings_path, name_prefix, \
                     print_details, log_details, record_steps)

    #create instance of TvStokesDualDD-class
    tvsd_obj = tvsd_dd.TvStokesDualDD(dd_obj, tvsd_parameters, rec_obj, comm)

    #RUN ALGORITHM
    t1 = time.time()
    d_parr, tau_parr, g_parr = tvsd_obj.run(noisy_img_parr)
    t2 = time.time()

    if (g_parr == None):
        results = deparallelize( \
                [d_parr, tau_parr[0,:,:], tau_parr[1,:,:]], \
                dd_obj, comm)
        d, tau0, tau1 = results
        g = None
    else:
        results = deparallelize( \
                [d_parr, tau_parr[0,:,:], tau_parr[1,:,:], g_parr], \
                dd_obj, comm)
        d, tau0, tau1, g = results

    if (rank == 0):
        print("Duration of Tv Stokes DD in Seconds (rank " + str(rank) + "): ", (t2 - t1))
        if (True):
            winnames = ['ground truth', 'noisy image', 'integrated normal field g', \
                        'reconstructed tangent field tau[0]', 'reconstructed tangent field tau[1]', \
                        'TV Stokes Dual denoised image']
            winwid = 400
            winhei = int(winwid * ground_truth.shape[1] / ground_truth.shape[0])
            for winname in winnames:
                cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(winname, winhei, winwid)
            cv2.imshow('ground truth', ground_truth)
            cv2.imshow('noisy image', noisy_img)
            import os
            path = recordings_path
            with open(os.path.join(path, 'noisy_img.bin'), 'wb') as f:
                noisy_img.tofile(f)
            if (type(g) == np.ndarray):
                cv2.imshow('integrated normal field g', g + 0.5 * np.ones(g.shape))
                with open(os.path.join(path, 'g.bin'), 'wb') as f:
                    g.tofile(f)
            cv2.imshow('reconstructed tangent field tau[0]', tau0 + 0.5 * np.ones(tau0.shape))
            cv2.imshow('reconstructed tangent field tau[1]', tau1 + 0.5 * np.ones(tau1.shape))
            cv2.imshow('TV Stokes Dual denoised image', d)
            with open(os.path.join(path, 'tau0.bin'), 'wb') as f:
                tau0.tofile(f)
            with open(os.path.join(path, 'tau1.bin'), 'wb') as f:
                tau1.tofile(f)
            with open(os.path.join(path, 'denoised.bin'), 'wb') as f:
                d.tofile(f)
            while 1:
                if cv2.waitKey(1) == 27:    #ESC
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()