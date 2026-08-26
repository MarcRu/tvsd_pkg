
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re


ir_variant = 1         #1/2
if (ir_variant == 1):
    non_dd_folder = "tvsd_2025-08-12_12-17-41-785362"
    dd_folder = "tvsd_dd_par_2025-08-08_14-41-58-514121"
    dd_folder_rerun = "tvsd_dd_par_2025-08-13_12-26-29-020237"
if (ir_variant == 2):
    non_dd_folder = "tvsd_2025-08-11_14-26-23-332007"
    dd_folder = "tvsd_dd_par_2025-08-08_14-42-14-302186"
    dd_folder_rerun = "tvsd_dd_par_2025-08-13_12-27-36-006128"

def img_analysis():
    # Load binary files back
    width = 480
    height = 270

    #test
    d0_ndd = np.fromfile(os.path.join(non_dd_folder, 'noisy_img.bin'), \
                          dtype=np.float64).reshape((height, width))
    d0_dd = np.fromfile(os.path.join(dd_folder, 'noisy_img.bin'), \
                          dtype=np.float64).reshape((height, width))
    np.testing.assert_almost_equal(d0_ndd, d0_dd, 15)

    #comparisons
    tau_ndd = np.fromfile(os.path.join(non_dd_folder, 'tau.bin'), \
                            dtype=np.float64).reshape((2, height, width))
    tau_dd = np.zeros((2, height, width), dtype=np.float64)
    tau_dd[0,:,:] = np.fromfile(os.path.join(dd_folder, 'tau0.bin'), \
                          dtype=np.float64).reshape((height, width))
    tau_dd[1,:,:] = np.fromfile(os.path.join(dd_folder, 'tau1.bin'), \
                          dtype=np.float64).reshape((height, width))
    if (ir_variant == 2):
        g_ndd = np.fromfile(os.path.join(non_dd_folder, 'g.bin'), \
                              dtype=np.float64).reshape((height, width))
        g_dd = np.fromfile(os.path.join(dd_folder, 'g.bin'), \
                              dtype=np.float64).reshape((height, width))
    d_ndd = np.fromfile(os.path.join(non_dd_folder, 'denoised_img.bin'), \
                          dtype=np.float64).reshape((height, width))
    d_dd = np.fromfile(os.path.join(dd_folder, 'denoised.bin'), \
                          dtype=np.float64).reshape((height, width))
    err = np.abs(d_ndd - d_dd)
    print("infty-norm: ", np.max(err))
    print("average err: ", np.mean(err))
    print("median err: ", np.median(err))
    print("0.9 quantile err: ", np.quantile(err, 0.9))
    print("0.99 quantile err: ", np.quantile(err, 0.99))
    #argmax_diff = np.unravel_index(np.argmax(err), d_dd.shape)
    #print("tau bei argmax(err), non-dd:", tau_ndd[:,argmax_diff[0],argmax_diff[1]])
    #print("tau bei argmax(err), dd:", tau_dd[:,argmax_diff[0],argmax_diff[1]])
    return d_ndd, d_dd

#save images in useful format
def save_img_useful(d_ndd, d_dd):
    img_fn_png = non_dd_folder + "_non_dd.png"
    img_fn_png_dd = dd_folder + "_dd.png"
    cv2.imwrite(img_fn_png, d_ndd * 255)
    cv2.imwrite(img_fn_png_dd, d_dd * 255)


def make_plot1(xvals, yvals, refval, datalabel, reflabel, xaxis_str, yaxis_str, title,
              yvals2=None, datalabel2=None, show_reference=True):
    # Plot the energies
    plt.plot(xvals, yvals, markersize=2, label=datalabel, linestyle='-')
    if (yvals2!=None):
        plt.plot(xvals, yvals2, markersize=2, label=datalabel2, linestyle='--')
    
    # Draw a horizontal line at the reference energy
    if show_reference:
        plt.axhline(y=refval, color='red', linestyle='--', linewidth=1, label=reflabel)
    
    
    # Labels and legend
    plt.xlabel(xaxis_str, fontsize=15, labelpad=2)
    #plt.ylabel(yaxis_str, fontsize=14, labelpad=-8)
    plt.xscale('log')
    plt.yscale('log')
    #plt.ylim([0,10**2])
    plt.title(yaxis_str, fontsize=17)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(datalabel+'.png')
    plt.show()
    
def make_plot2(xvals, yvals, refval, datalabel, reflabel, xaxis_str, yaxis_str, title,
              yvals2=None, datalabel2=None, show_reference=True):
    # Plot the energies
    plt.plot(xvals, yvals, markersize=2, label=datalabel, linestyle='-')
    if (yvals2!=None):
        plt.plot(xvals, yvals2, markersize=2, label=datalabel2, linestyle='--')
    
    # Draw a horizontal line at the reference energy
    if show_reference:
        plt.axhline(y=refval, color='red', linestyle='--', linewidth=1, label=reflabel)
    
    
    # Labels and legend
    plt.xlabel(xaxis_str, fontsize=15, labelpad=2)
    #plt.ylabel(yaxis_str, fontsize=14, labelpad=-8)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0,10**2])
    plt.title(yaxis_str, fontsize=17)
    plt.legend(fontsize=14, loc='upper left')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(datalabel+'.png')
    plt.show()


def energy_analysis():
    #load log files
    outer_iterations_tfs = []
    epj_values = []
    outer_iterations_ir = []
    erj_values = []
    outer_iterations_ir_rerun = []
    erj_values_rerun = []
    
    #baseline energy result non-dd (ADAPT!!!)
    reference_epj = round(0.03220318 ** 2, 8)
    if (ir_variant == 1):
        reference_erj = round(5.808410382 ** 2, 8)
    if (ir_variant == 2):
        reference_erj = round(5.542328233436 ** 2, 8)

    #energy development dd
    with open(os.path.join(dd_folder, "printout.log"), "r", encoding="utf-8") as f:
        for line in f:
            # Match outer iteration number
            match_outer_tfs = re.search(r"\(Rank 0\) Tfs OUTER iteration (\d+)", line)
            if match_outer_tfs:
                outer_iterations_tfs.append(int(match_outer_tfs.group(1)))
            # Match E(p^j) value
            match_epj = re.search(r"E\(p\^j\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if match_epj:
                epj_values.append(float(match_epj.group(1)))
            # Match outer iteration number
            match_outer_ir = re.search(r"\(Rank 0\) Ir[12] OUTER iteration (\d+)", line)
            if match_outer_ir:
                outer_iterations_ir.append(int(match_outer_ir.group(1)))
            # Match E(r^j) value
            match_erj = re.search(r"E\(r\^j\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if match_erj:
                erj_values.append(float(match_erj.group(1)))
    with open(os.path.join(dd_folder_rerun, "printout.log"), "r", encoding="utf-8") as f:
        for line in f:
            # Match outer iteration number
            match_outer_ir = re.search(r"\(Rank 0\) Ir[12] OUTER iteration (\d+)", line)
            if match_outer_ir:
                outer_iterations_ir_rerun.append(int(match_outer_ir.group(1)))
            # Match E(r^j) value
            match_erj = re.search(r"E\(r\^j\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if match_erj:
                erj_values_rerun.append(float(match_erj.group(1)))

    #get energy developments by parsing logfiles
    np.testing.assert_equal(len(outer_iterations_tfs), len(epj_values))
    np.testing.assert_equal(len(outer_iterations_ir), len(erj_values))
    np.testing.assert_equal(len(outer_iterations_ir_rerun), len(erj_values_rerun))
    outer_iterations_ir = outer_iterations_ir[:len(outer_iterations_ir_rerun)]
    erj_values = erj_values[:len(erj_values_rerun)]

    MODE = 2
    if (MODE == 1):         #standard diagrams
        #square are values
        epj_values = list(np.array(epj_values) ** 2)
        erj_values = list(np.array(erj_values) ** 2)
        erj_values_rerun = list(np.array(erj_values_rerun) ** 2)

        #Visualize epj_values and erj_values squared
        en_tfs_str = r'$\frac{1}{2|\tilde\Omega^h|}\mathcal{D}^h_{\mathrm{TFS}}(\vec{p}^{h,n})$'
        ref_en_tfs_str = r'Reference $\frac{1}{2|\tilde\Omega^h|}\mathcal{D}^h_{\mathrm{TFS}}(\vec{p}^{h*}) = $' + f'{reference_epj}'
        en_ir_str = r'$\frac{1}{|\Omega^h|}\mathcal{D}^h_{\mathrm{IRV' + str(ir_variant) + r'}}(\vec{r}^{h,n})$ from Tangent Field of DD-run'
        en_ir_str_rerun = r'$\frac{1}{|\Omega^h|}\mathcal{D}^h_{\mathrm{IRV' + str(ir_variant) + r'}}(\vec{r}^{h,n})$ from reference Tangent Field'
        ref_en_ir_str = r'Reference $\frac{1}{|\Omega^h|}\mathcal{D}^h_{\mathrm{IRV' + str(ir_variant) + r'}}(\vec{r}^{h*}) = $' + f'{reference_erj}'
        make_plot(outer_iterations_tfs, epj_values, reference_epj, \
                en_tfs_str, ref_en_tfs_str, \
                "Outer Iteration DD", "Value", "Tangent Field Smoothing Energy Development")
        make_plot(outer_iterations_ir, erj_values, reference_erj, \
                en_ir_str, ref_en_ir_str, \
                "Outer Iteration DD", "Value", "Image Reconstruction Variant " + str(ir_variant) + " Energy Development", \
                erj_values_rerun, en_ir_str_rerun) 
    
    elif (MODE == 2):         #logarithms of errors
        #square are values
        epj_values = list(np.abs(reference_epj - np.array(epj_values) ** 2))
        erj_values = list(np.abs(reference_erj - np.array(erj_values) ** 2))
        erj_values_rerun = list(np.abs(reference_erj - np.array(erj_values_rerun) ** 2))

        #Visualize epj_values and erj_values squared
        en_tfs_expr = r'$\frac{1}{2|\tilde\Omega^h|}\left|\mathcal{D}^h_{\mathrm{TFS}}(\vec{p}^{h,n})-\mathcal{D}^h_{\mathrm{TFS}}(\vec{p}^{h*})\right|$'
        en_ir_expr = r'$\frac{1}{|\Omega^h|}\left|\mathcal{D}^h_{\mathrm{IRV' + str(ir_variant) + r'}}(\vec{p}^{h,n})-\mathcal{D}^h_{\mathrm{IRV' + str(ir_variant) + r'}}(\vec{p}^{h*})\right|$'
        en_ir_str = r'DD-IRV' + str(ir_variant) + r', tangent field from DD-run'
        en_ir_str_rerun = r'DD-IRV' + str(ir_variant) + r', reference tangent field used'
        make_plot1(outer_iterations_tfs, epj_values, reference_epj, \
                "DD-TFS", "", \
                "Outer Iteration DD", en_tfs_expr, "Tangent Field Smoothing Energy Development",show_reference=False)
        make_plot2(outer_iterations_ir, erj_values, reference_erj, \
                en_ir_str, "", \
                "Outer Iteration DD", en_ir_expr, "Image Reconstruction Variant " + str(ir_variant) + " Energy Development", \
                erj_values_rerun, en_ir_str_rerun,show_reference=False) 


#d_ndd, d_dd = img_analysis()
#save_img_useful(d_ndd, d_dd)
energy_analysis()
