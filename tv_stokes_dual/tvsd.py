
import tv_stokes_dual.proj_div_0 as pd0

import math
import numpy as np
import logging
from typing import Tuple

#only necessary for recordings
import cv2
import datetime
import os


"""
Simple implementation of dual TV-Stokes-algorithm.
Call tv_stokes_dual for the whole algorithm
and call tv_stokes_dual_tfs_only or tv_stokes_dual_ir_only
for only Tangent Field Smoothing / Image Reconstruction.
Valid configuration parameters are required as dictionary to
call the function. An example for a valid config-dictionary
is written down in config_tvsd/example_config.yaml.
It only has to be written in with a simple call:

import yaml
with open(<FILENAME_WITH_PATH>, 'r') as file:
        tvsd_parameters = yaml.safe_load(file)
"""


#==================================================================

#LOGGER
#If the printouts shall be dumped into a file or hidden, adapt this line.

def setup_logger(recordings_path : str):
    #logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
    logging.basicConfig(
        filename=os.path.join(recordings_path, 'output.log'),
        filemode='w',                     # 'w' = overwrite each run, 'a' = append
        level=logging.DEBUG,              # Minimum log level to capture
        format='%(levelname)s:%(message)s'
    )

#===================================================================


def save_recording(img : np.ndarray, 
                   name : str, 
                   recordings_path : str, 
                   print_saved : bool = False):
    """
    Save image img with name name at location recordings_path.
    Set print_saved=True for logging this.
    """
    full_filename = os.path.join(recordings_path, (name + ".png"))
    if (print_saved):
        logging.info(full_filename)
    cv2.imwrite(full_filename, img)


def save_recording_rescaled(img : np.ndarray, 
                   name : str, 
                   recordings_path : str, 
                   print_saved : bool = False):
    """
    Save image img with name name at location recordings_path.
    Rescale the color values to increase contrasts.
    Set print_saved=True for logging this.
    """
    imgmax = img.max()
    imgmin = img.min()
    if (img.max() - img.min() < 1.e-12):
        save_recording(img, name, recordings_path, print_saved)
    else:
        img_rescaled = (img - imgmin) * 255 / (imgmax - imgmin)
        save_recording(img_rescaled, name, recordings_path, print_saved)


def prepare_recordings(parameters : dict, recordings_folder_name : str):
    """
    Saves recordings_path in parameters-dictionary, if
    paramters["record_steps"] > 0;
    otherwise it saves None
    """ 

    if (parameters["record_steps"] > 0):   
        time_now = datetime.datetime.now()
        if (recordings_folder_name == ""):
            recordings_folder_name = 'tvsd_' + time_now.strftime("%Y-%m-%d_%H-%M-%S-%f")
        if not(os.path.exists("recordings")):
            os.mkdir("recordings")
        os.chdir("recordings")
        if not(os.path.exists(recordings_folder_name)):
            os.mkdir(recordings_folder_name)
        os.chdir("..")
        recordings_path = os.getcwd() + '/recordings/' + recordings_folder_name + '/'
        parameters["recordings_path"] = recordings_path
        
    else:
        parameters["recordings_path"] = None


#===================================================================


class DifferentiatorObject:
    """
    Provides grad- and div-function, such that grad and -div are adjoint.
    There are two modes:
    - "neumann": grad^+ and div^-
    - "dirichlet": grad^- and div^+
    The term "neumann" refers to that the corresponding laplace operator
    div grad would have Neumann-boundary conditions.
    """
    def __init__(self, N1 : int, N2 : int, h : float = 1.0, diff_mode : str = "neumann", dtype : np.dtype = np.float64):
        self.N2 = N2
        self.N1 = N1
        self.h = h
        self.dm = diff_mode
        self.dtype = dtype
    
    def div(self, q : np.ndarray) -> np.ndarray:
        """
        Applies divB if Neumann-boundary-projection is used
        and divF if Dirichlet-boundary-projection is used
        """
        div_q = np.zeros((self.N2,self.N1), self.dtype)
        if (self.dm == "dirichlet"):
            pd0.divF(q[0,:,:], q[1,:,:], div_q, self.h)
        else:   #dm == "neumann"
            pd0.divB(q[0,:,:], q[1,:,:], div_q, self.h)
        return np.array(div_q)

    def grad(self, expr : np.ndarray) -> np.ndarray:
        """
        Applies gradF if Neumann-boundary-projection is used
        and gradB if Dirichlet-boundary-projection is used
        """
        grad_expr = np.zeros((2,self.N2,self.N1), self.dtype)
        if (self.dm == "dirichlet"):
            pd0.gradB(expr, grad_expr[0,:,:], grad_expr[1,:,:], self.h)
        else:   #dm == "neumann"
            pd0.gradF(expr, grad_expr[0,:,:], grad_expr[1,:,:], self.h)
        return np.array(grad_expr)


class ProjectorObject(DifferentiatorObject):
    """
    Provides, additionally to the DifferentiatorObject, the possiblity
    to project a tangent field tau on a subspace with div tau = 0.
    If projection_mode == "neumann", then it will be projected on the 
    subspace with div^- tau = 0, if projection_mode == "dirichlet" then 
    it will be projected on subspace with div^+ tau = 0.
    """
    def __init__(self, N1 : int, N2 : int, h : float = 1.0, projection_mode : str = "neumann", dtype : np.dtype = np.float64):
        super().__init__(N1, N2, h, projection_mode, dtype)
        self.pm = self.dm

        #prepare help matrices to apply Cython-routines for projections

        #allocate space for projections to operate on
        self.tmp_mat1 = np.zeros((self.N2, self.N1), self.dtype)
        self.tmp_mat2 = np.zeros((self.N2, self.N1), self.dtype)

        #prepare Sigma_sq and temporary matrices for inverse laplacian
        if (self.pm == "dirichlet"):
            self.Sigma1_sq = np.zeros(self.N1, self.dtype)
            self.Sigma2_sq = np.zeros(self.N2, self.dtype)
            pd0.prepare_Sigma_sq_dst(self.Sigma1_sq, self.h)
            pd0.prepare_Sigma_sq_dst(self.Sigma2_sq, self.h)
        else:   #pm == "neumann" 
            self.Sigma1_sq = np.zeros(self.N1, self.dtype)
            self.Sigma2_sq = np.zeros(self.N2, self.dtype)
            pd0.prepare_Sigma_sq_dct(self.Sigma1_sq, self.h)
            pd0.prepare_Sigma_sq_dct(self.Sigma2_sq, self.h)

    def project(self, tau : np.ndarray) -> np.ndarray:
        """
        Projects tau on subspace with div(tau) = 0
        """
        P_tau = np.zeros((2,self.N2,self.N1), self.dtype)

        if (self.pm == "dirichlet"):
            P_tau = pd0.proj_dirichlet_dst(tau, P_tau, \
                        self.Sigma1_sq, self.Sigma2_sq, self.tmp_mat1, self.tmp_mat2, self.h)
        else:   #pm == "neumann"
            P_tau = pd0.proj_neumann_dct(tau, P_tau, \
                        self.Sigma1_sq, self.Sigma2_sq, self.tmp_mat1, self.tmp_mat2, self.h)

        return np.array(P_tau)
    
    
#===================================================================


def get_tau_from_p(p : np.ndarray,
                P_tau0 : np.ndarray, 
                delta : float, 
                po : ProjectorObject) \
                -> np.ndarray:
    """
    Performs primal tangent field tau (size 2xN2xN1) 
    from dual tangent field p (size 2x2xN2xN1)
    Args:
        p: dual tangent field
        P_tau0: projected noisy (initial) tangent field
        delta: regularization parameter of Tangent Field Smoothing
        po: projection object to apply projection
    Returns:  
        tau: primal tangent field
    """
    N2, N1 = p.shape[2], p.shape[3]
    div_p = np.zeros((2, N2, N1), p.dtype)
    div_p[0,:,:] = po.div(p[0,:,:,:])
    div_p[1,:,:] = po.div(p[1,:,:,:])
    P_div_p = po.project(div_p)
    tau = P_tau0 - delta * P_div_p
    return tau


def tangent_field_smoothing(tau0 : np.ndarray,
                            tfs_parameters: dict) \
                    -> Tuple[np.ndarray, float]:
    """
    Performs Tangent Field Smoothing on noisy tangent field tau0
    by computing the dual tangent field via Chambolle iteration.
    Args:
        tau0: noisy tangent field
        tfs_parameters: Dictionary containing all required parameters
            for the Tangent Field Smoothing. For a detailed description
            of the parameters, see 
            tv_stokes_dual/config_tvsd/example_config.yaml.
    Returns:  
        tau: final primal tangent field
        energy: final energy l2norm(P_K div p - delta^(-1) tau_0)
    """
    print_details = tfs_parameters["print_details"]    
    h = tfs_parameters["h"]    
    k = tfs_parameters["k"]
    delta = tfs_parameters["delta"]
    pm = tfs_parameters["projection_mode"]
    stop_criteria = tfs_parameters["stop_criteria"]   
    max_steps = tfs_parameters["max_steps"]    
    energy_thresh_diff = tfs_parameters["energy_thresh_diff"]
    recordings_path = tfs_parameters["recordings_path"]
    
    N1 = tau0.shape[2]
    N2 = tau0.shape[1]
    std_size = (float)(2.0 * N2 * N1)

    if (print_details > 0):
        logging.info("Tangent Field Smoothing starting.")    
    
    po = ProjectorObject(N1, N2, h, pm, tau0.dtype)

    #project tau0 on subspace div(tau0) = 0
    Ptau0 = po.project(tau0)
    tau0til = (1.0 / delta) * np.array(Ptau0)

    if (tfs_parameters["record_steps"] > 0):
        print_saved = (tfs_parameters["print_details"] > 0)
        save_recording_rescaled(tau0[0]  * 255, "tau0[0]", recordings_path, print_saved)
        save_recording_rescaled(tau0[1]  * 255, "tau0[1]", recordings_path, print_saved)

    #prepare loop variables
    p = np.zeros((2,2,N2,N1), tau0.dtype)
    energy = 1.e+10     #initial default energy (very high)
    for st in range(max_steps):
        div_p = np.zeros((2,N2,N1), tau0til.dtype)
        div_p[0,:,:] = po.div(p[0,:,:,:])
        div_p[1,:,:] = po.div(p[1,:,:,:])
        P_div_p = po.project(div_p)
        P_div_p_tau = P_div_p - tau0til
        psi = np.zeros((2,2,N2,N1), tau0til.dtype)
        psi[0,:,:,:] = po.grad(P_div_p_tau[0,:,:])
        psi[1,:,:,:] = po.grad(P_div_p_tau[1,:,:])
        abs_psi0 = np.sqrt(psi[0,0,:,:] ** 2 + psi[0,1,:,:] ** 2)
        abs_psi1 = np.sqrt(psi[1,0,:,:] ** 2 + psi[1,1,:,:] ** 2)
        den0 = abs_psi0 * k + 1.0
        den1 = abs_psi1 * k + 1.0
        p[0,0,:,:] = (p[0,0,:,:] + k * psi[0,0,:,:]) / den0
        p[0,1,:,:] = (p[0,1,:,:] + k * psi[0,1,:,:]) / den0
        p[1,0,:,:] = (p[1,0,:,:] + k * psi[1,0,:,:]) / den1
        p[1,1,:,:] = (p[1,1,:,:] + k * psi[1,1,:,:]) / den1

        old_energy = energy
        energy = np.linalg.norm(P_div_p_tau) / math.sqrt(std_size)

        if (print_details > 1 and \
                (st % 100 == 0 or \
                 (st % 10 == 0 and st < 10000) or \
                 (st < 1000))):
            logging.info("Step: " + str(st))
            tau = get_tau_from_p(p, Ptau0, delta, po)
            div_tau = po.div(tau)
            norm_div_tau = np.linalg.norm(div_tau)
            logging.info("Divergence l2norm(div(tau)): " + str(norm_div_tau))
            logging.info("Energy l2norm(Pi_k div p - delta^(-1) tau_0): " + str(energy))

        if (stop_criteria == "Cauchy" and st > 1):
            if (old_energy - energy < energy_thresh_diff):
                if (print_details > 0):
                    logging.info("Tangent Field Smoothing terminating after " + str(st) + " steps.")
                    logging.info("Final energy = " + str(energy))
                break

    #calculate primal tangent field tau from dual variable p
    tau = get_tau_from_p(p, Ptau0, delta, po)

    if (print_details > 0):
        logging.info("Tangent Field Smoothing finishing.")

    return tau, energy


#===================================================================


def image_reconstruction_v1(d0 : np.ndarray,
                            xi : np.ndarray,
                            ir_parameters : dict) \
                    -> Tuple[np.ndarray, float]:
    """
    Performs Image Reconstruction Version 1 on noisy image d0
    by reconstructing the dual image via Chambolle iteration.
    The normalized normal field xi will be used, which is typically
    determined from the tangent field tau (which should be 
    reconstructed first).
    Args:
        d0: noisy image
        xi: normalized normal field n/|n| with n = (tau2, -tau1)
        ir_parameters: Dictionary containing all required parameters
            for the Image Reconstruction. For a detailed description
            of the parameters, see 
            tv_stokes_dual/config_tvsd/example_config.yaml.
    Returns:  
        d: final primal tangent field
        energy: final energy l2norm(P_K div p - delta^(-1) tau_0)
    """

    #prepare parameters
    energy_thresh_diff = ir_parameters["energy_thresh_diff"]
    max_steps = ir_parameters["max_steps"]
    stop_criteria = ir_parameters["stop_criteria"]
    h = ir_parameters["h"]
    k = ir_parameters["k"]
    diff_mode = ir_parameters["diff_mode"]
    N1 = d0.shape[1]
    N2 = d0.shape[0]
    mu = ir_parameters["mu"]
    print_details = ir_parameters["print_details"]
    recordings_path = ir_parameters["recordings_path"]


    if (print_details > 0):
        logging.info("Image Reconstruction (variant 1) starting.")

    

    #differentiator object
    do = DifferentiatorObject(N1, N2, h, diff_mode, d0.dtype)

    #prepare loop variables
    r = np.zeros((2,N2,N1), d0.dtype)   
    energy = 1.e+10     #initial default energy (very high)
    for st in range(max_steps):
        r_xi = r + xi
        div_r_xi = do.div(r_xi)
        div_r_xi_d0 = div_r_xi - (1 / mu) * d0
        rho = do.grad(div_r_xi_d0)
        abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
        den = abs_rho * k + 1.0
        r[0,:,:] = (r[0,:,:] + k * rho[0,:,:]) / den
        r[1,:,:] = (r[1,:,:] + k * rho[1,:,:]) / den

        old_energy = energy
        energy = np.linalg.norm(div_r_xi_d0) / math.sqrt(N1 * N2)
        if (print_details > 1 and \
                (st % 100 == 0 or \
                 (st % 10 == 0 and st < 10000) or \
                 (st < 1000))):
            logging.info("Energy l2norm(div(r + n/|n|) - d0/mu) after step " + str(st) + ":")
            logging.info(str(energy))
        if (stop_criteria == "energy_thresh" and st > 1):
            if (old_energy - energy < energy_thresh_diff):
                if (print_details > 0):
                    logging.info("Image Reconstruction terminating after " + str(st) + " steps.")
                    logging.info("Final energy = " + str(energy))
                break

    #recover d from r
    r_xi = r + xi
    div_r_xi = do.div(r_xi)
    d = d0 - mu * div_r_xi

    if (ir_parameters["record_steps"] > 0):
        print_saved = (ir_parameters["print_details"] > 0)
        save_recording(d * 255, "denoised_img", recordings_path, print_saved)

    if (print_details > 0):
        logging.info("Image Reconstruction finishing.")

    return d, energy


def image_reconstruction_v2(d0 : np.ndarray,
                            g : np.ndarray,
                            ir_parameters : dict) \
                    -> Tuple[np.ndarray, float]:
    """
    Performs Image Reconstruction Version 2 on noisy image d0
    by reconstructing the dual image via Chambolle iteration.
    The integrated tangent field g will be used, which is typically
    determined from the tangent field tau (which should be 
    reconstructed first).
    Args:
        d0: noisy image
        g: integrated tangent field: grad(g) = (tau2, -tau1)
        ir_parameters: Dictionary containing all required parameters
            for the Image Reconstruction. For a detailed description
            of the parameters, see 
            tv_stokes_dual/config_tvsd/example_config.yaml.
    Returns:  
        d: final primal tangent field
        energy: final energy l2norm(P_K div p - delta^(-1) tau_0)
    """

    #prepare parameters
    energy_thresh_diff = ir_parameters["energy_thresh_diff"]
    max_steps = ir_parameters["max_steps"]
    stop_criteria = ir_parameters["stop_criteria"]
    h = ir_parameters["h"]
    beta = ir_parameters["beta"]
    k = ir_parameters["k"]
    diff_mode = ir_parameters["diff_mode"]
    print_details = ir_parameters["print_details"]
    recordings_path = ir_parameters["recordings_path"]
    N2, N1 = d0.shape

    if (print_details > 0):
        logging.info("Image Reconstruction (variant 2) starting.")
     
    #Differentiator object
    do = DifferentiatorObject(N1, N2, h, diff_mode, d0.dtype)

    #Prepare data term u0
    u0 = d0 - g

    #prepare r
    r = np.zeros((2,N2, N1), d0.dtype)
    energy = 1.e+10     #initial default energy (very high)
    for st in range(max_steps):
        div_r = do.div(r)
        div_r_u0 = div_r - beta * u0
        rho = do.grad(div_r_u0)
        abs_rho = np.sqrt(rho[0,:,:] ** 2 + rho[1,:,:] ** 2)
        den = abs_rho * k + 1.0
        r[0,:,:] = (r[0,:,:] + k * rho[0,:,:]) / den
        r[1,:,:] = (r[1,:,:] + k * rho[1,:,:]) / den

        old_energy = energy
        energy = np.linalg.norm(div_r_u0) / math.sqrt(N1 * N2)
        if (print_details > 1 and \
                (st % 100 == 0 or \
                 (st % 10 == 0 and st < 10000) or \
                 (st < 1000))):
            logging.info("Energy l2norm(div(r) - beta*(d0-g)) after step " + str(st) + ":")
            logging.info(str(energy))
        if (stop_criteria == "energy_thresh" and st > 1):
            if (old_energy - energy < energy_thresh_diff):
                if (print_details > 0):
                    logging.info("Image Reconstruction terminating after " + str(st) + " steps.")
                    logging.info("Final energy = " + str(energy))
                break

    #recover d from r
    div_r = do.div(r)
    d = d0 - (1. / beta) * div_r

    if (ir_parameters["record_steps"] > 0):
        print_saved = (ir_parameters["print_details"] > 0)
        save_recording(d * 255, "denoised_img", recordings_path, print_saved)

    if (print_details > 0):
        logging.info("Image Reconstruction finishing.")

    return d, energy


#===================================================================


def img_to_tangent_field(d0 : np.ndarray, h : float = 1.0) -> np.ndarray:
    """
    Determine noisy tangent field tau0 from noisy image d0
    (before tangent field smoothing)
    """
    N2, N1 = d0.shape
    N2t, N1t = N2 + 1, N1 + 1

    tau0 = np.zeros((2, N2t, N1t), d0.dtype)
    #inner part
    tau0[0,1:-1,:-1] = (1. / h) * (-d0[1:,:] + d0[:-1,:])
    tau0[1,:-1,1:-1] = (1. / h) * (d0[:,1:] - d0[:,:-1])
    #right/bottom boundary: mirror (=Neumann boundary)
    tau0[0,1:-1,-1] = np.copy(tau0[0,1:-1,-2])
    tau0[1,-1,1:-1] = np.copy(tau0[1,-2,1:-1])

    return tau0


def tau_to_xi(tau : np.ndarray, eps : float) -> np.ndarray:
    """
    prepare normalized normal vector xi (before Image Reconstruction Variant 1)
    """
    abs_tau = np.sqrt(tau[0,:,:] ** 2 + tau[1,:,:] ** 2 + eps)
    xi = np.zeros(tau.shape, tau.dtype)
    xi[0,:,:] = tau[1,:,:] / abs_tau
    xi[1,:,:] = (-1.0) * tau[0,:,:] / abs_tau

    return xi
    

def tau_to_g(tau : np.ndarray, int_mode : int = -1, h : float = 1.0) \
                -> np.ndarray:
    """
    Prepare potential function g from tangent fiesld tau
    (before Image Reconstruction Variant 2)
    """
    
    g = np.zeros((tau.shape[1],tau.shape[2]), tau.dtype)
    g = pd0.integrate_tangentfield(tau, g, h, int_mode)  
    return np.array(g)


#=====================================================================================


def tv_stokes_dual_tfs_only(d0 : np.ndarray,
                            tfs_parameters : dict, 
                            recordings_folder_name : str = "") \
                -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs dual TV Stokes algorithm (tvsd), but only Tangent Field Smoothing
    Args:
        d0: noisy image
        tfs_parameters: parameter dictionary for tangent field smoothing
        recordings_folder_name: folder name for possible recordings
    Returns:  
        tau: final tangent field
        tau0: initial noisy tangent field
        tfs_energy: final Tangent Field Smoothing energy
    """

    #Prepare recordings
    if not("recordings_path" in tfs_parameters):
        prepare_recordings(tfs_parameters, recordings_folder_name)
    recordings_path = tfs_parameters["recordings_path"]

    if not(recordings_path == None):
        print_saved = (tfs_parameters["record_steps"] > 0)
        save_recording(d0  * 255, "noisy_img", recordings_path, print_saved)

    #Tangent Field Smoothing
    h = tfs_parameters["h"]
    tau0 = img_to_tangent_field(d0, h)      #tau0 on extended (dual) coordinates
    tau, tfs_energy = tangent_field_smoothing(tau0, tfs_parameters)
    tau, tau0 = tau[:,:-1,:-1], tau0[:,:-1,:-1]     #back from extended to primal coordinates
    
    if (tfs_parameters["record_steps"] > 0):
        print_saved = (tfs_parameters["print_details"] > 0)
        save_recording_rescaled(tau[0,:,:] * 255, "tau[0]", recordings_path, print_saved)
        save_recording_rescaled(tau[1,:,:] * 255, "tau[1]", recordings_path, print_saved)
    
    return tau, tau0, tfs_energy


def tv_stokes_dual_ir_only(d0 : np.ndarray, 
                        tau : np.ndarray, 
                        ir_parameters : dict,
                        projection_mode : str = "neumann", recordings_folder_name : str = "",
                        save_start_img : bool =False) \
                -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs dual TV Stokes algorithm (tvsd), but only Image Reconstruction
    Args:
        d0: noisy image
        tau: tangent field
        ir_parameters: parameter dictionary for image reconstruction
        projection_mode: "neumann" or "dirichlet"
                            Set "neumann", if div^-(tau) = 0, 
                            set "dirichlet", if div^+(tau) = 0.
                            Only important for Image Reconstruction Variant 2.
        recordings_folder_name: folder name for possible recordings
    Returns:  
        d: reconstructed image
        xi/g: normalized normal field xi (only Variant 1) / integrated tangent field g (only Variant 2)
        ir_energy: final image reconstruction energy
    """

    #Prepare recordings
    if not("recordings_path" in ir_parameters):
        prepare_recordings(ir_parameters, recordings_folder_name)
    recordings_path = ir_parameters["recordings_path"]

    if (save_start_img):
        print_saved = (ir_parameters["record_steps"] > 0)
        save_recording(d0 * 255, "noisy_img", recordings_path, print_saved)
        save_recording_rescaled(tau[0,:,:] * 255, "tau[0]", recordings_path, print_saved)
        save_recording_rescaled(tau[1,:,:] * 255, "tau[1]", recordings_path, print_saved)

    if (ir_parameters["variant"] == 1):
        eps = ir_parameters["eps"]
        xi = tau_to_xi(tau, eps)
        if (ir_parameters["record_steps"] > 0):
            print_saved = (ir_parameters["print_details"] > 0)
            save_recording_rescaled(xi[0,:,:] * 255, "xi[0]", recordings_path, print_saved)
            save_recording_rescaled(xi[1,:,:] * 255, "xi[1]", recordings_path, print_saved)
        
        d, ir_energy = image_reconstruction_v1(d0, xi, ir_parameters)
        
        return d, xi, ir_energy
    elif (ir_parameters["variant"] == 2):
        int_mode = 1 if projection_mode == "dirichlet" else -1
        h = ir_parameters["h"]
        g = tau_to_g(tau, int_mode, h)
        if (ir_parameters["record_steps"] > 0):
            print_saved = (ir_parameters["print_details"] > 0)
            save_recording_rescaled(g * 255, "g", recordings_path, print_saved)
        
        d, ir_energy = image_reconstruction_v2(d0, g, ir_parameters)

        return d, g, ir_energy
    else:
        raise Exception("Invalid image_reconstruction variant.")
    

def tv_stokes_dual(d0 : np.ndarray, 
                   tvsd_parameters : dict, 
                   recordings_folder_name : str = "") \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Performs complete dual TV Stokes algorithm (tvsd)
    Args:
        d0: noisy image
        tvsd_parameters: parameter dictionary for all parameters of
                        dual TV Stokes algorithm; see
                        tv_stokes_dual/config_tvsd/example_config.yaml
                        for an example
        recordings_folder_name: folder name for possible recordings
    Returns:  
        d: reconstructed image
        tau: reconstructed tangent field
        tau0: noisy tangent field
        arr2: normalized normal field xi (only Variant 1) or integrated tangent field g (only Variant 2)
        tfs_energy: final Tangent Field Smoothing energy
        ir_energy: final Image reconstruction energy
    """

    tfs_parameters = tvsd_parameters["tangent_field_smoothing"]
    ir_parameters = tvsd_parameters["image_reconstruction"]

    #Prepare recordings
    tvsd_parameters["record_steps"] = max(tfs_parameters["record_steps"], ir_parameters["record_steps"])
    prepare_recordings(tvsd_parameters, recordings_folder_name)
    recordings_path = tvsd_parameters["recordings_path"]
    setup_logger(recordings_path)
    tfs_parameters["recordings_path"] = recordings_path
    ir_parameters["recordings_path"] = recordings_path


    #Tangent Field Smoothing
    tau, tau0, tfs_energy = tv_stokes_dual_tfs_only(d0, tfs_parameters, recordings_folder_name)


    #Image Reconstruction
    d, arr2, ir_energy = tv_stokes_dual_ir_only(d0, tau, ir_parameters, tfs_parameters["projection_mode"], \
                                                recordings_folder_name, save_start_img=False)

    return d, tau, tau0, arr2, tfs_energy, ir_energy