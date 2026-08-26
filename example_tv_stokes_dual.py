
import cv2
import math
import numpy as np
import time
import yaml
    

from tv_stokes_dual.tvsd import tv_stokes_dual


def main():
    test_img_number = 1
    test_images = ["phantom_al.png", "phantom_mr.png", \
                    "beach_nz_1_480.jpg"]
    ground_truth = cv2.imread("resources/" + test_images[test_img_number], cv2.IMREAD_GRAYSCALE)
    ground_truth = ground_truth.astype("float64") / 255
    with open('tv_stokes_dual/config_tvsd/example_config.yaml', 'r') as file:
        tvsd_parameters = yaml.safe_load(file)
    noise_var = 0.01
    gaussian = np.random.normal(loc=0, scale=math.sqrt(noise_var), size=ground_truth.shape) 
    noisy_img = np.clip(ground_truth + gaussian, 0.0, 1.0, dtype=np.float64)
    #RUN ALGORITHM
    t1 = time.time()
    d, tau, tau0, arr2, _, _ = tv_stokes_dual(noisy_img, tvsd_parameters)
    t2 = time.time()
    print("Duration of Tv Stokes in Seconds: ", (t2 - t1))
    winnames_v1 = ['ground truth', 'noisy image d0', 'tangent field tau0[0]', 'tangent field tau0[1]', \
                        'reconstructed tangent field tau[0]', 'reconstructed tangent field tau[1]', \
                        'integrated normal field xi[0]', 'integrated normal field xi[1]', \
                        'TV Stokes Dual denoised image d*']
    winnames_v2 = ['ground truth', 'noisy image d0', 'tangent field tau0[0]', 'tangent field tau0[1]', \
                        'reconstructed tangent field tau[0]', 'reconstructed tangent field tau[1]', \
                        'integrated normal field g', 'TV Stokes Dual denoised image d*']
    winnames = winnames_v1 if tvsd_parameters["image_reconstruction"]["variant"] == 1 else winnames_v2
    winwid = 400
    winhei = int(winwid * ground_truth.shape[1] / ground_truth.shape[0])
    for winname in winnames:
        cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, winhei, winwid)
    cv2.imshow('ground truth', ground_truth)
    cv2.imshow('noisy image d0', noisy_img)
    cv2.imshow('tangent field tau0[0]', tau0[0,:,:] + 0.5 * np.ones(tau0[0,:,:].shape))
    cv2.imshow('tangent field tau0[1]', tau0[1,:,:] + 0.5 * np.ones(tau0[1,:,:].shape))
    cv2.imshow('reconstructed tangent field tau[0]', tau[0,:,:] + 0.5 * np.ones(tau[0,:,:].shape))
    cv2.imshow('reconstructed tangent field tau[1]', tau[1,:,:] + 0.5 * np.ones(tau[1,:,:].shape))
    #Binaries (only necessary for experiments)
    import os
    path = tvsd_parameters["recordings_path"]
    with open(os.path.join(path, 'noisy_img.bin'), 'wb') as f:
        noisy_img.tofile(f)
    with open(os.path.join(path, 'tau0.bin'), 'wb') as f:
        tau0.tofile(f)
    with open(os.path.join(path, 'tau.bin'), 'wb') as f:
        tau.tofile(f)
    if tvsd_parameters["image_reconstruction"]["variant"] == 1:
        cv2.imshow('integrated normal field xi[0]', arr2[0,:,:] + 0.5 * np.ones(arr2[0,:,:].shape))
        cv2.imshow('integrated normal field xi[1]', arr2[1,:,:] + 0.5 * np.ones(arr2[1,:,:].shape))
        with open(os.path.join(path, 'xi.bin'), 'wb') as f:
            arr2.tofile(f)
    elif tvsd_parameters["image_reconstruction"]["variant"] == 2:
        cv2.imshow('integrated normal field g', arr2 + 0.5 * np.ones(arr2.shape))
        with open(os.path.join(path, 'g.bin'), 'wb') as f:
            arr2.tofile(f)
    cv2.imshow('TV Stokes Dual denoised image d*', d)
    with open(os.path.join(path, 'denoised_img.bin'), 'wb') as f:
            d.tofile(f)
    while 1:
        if cv2.waitKey(1) == 27:    #ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
