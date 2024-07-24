#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:44:35 2023

@author: wuye
"""

import argparse
import os
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti
from dipy.reconst import mapmri
from dipy.data import get_sphere


def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit MAP-MRI model with Dipy",
        epilog="Written by Ye Wu, dr.yewu@outlook.com.\"")
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.0',
        help="Show program's version number and exit")
    parser.add_argument(
        'subjectDirectory',
        help='A directory of study subjects.')
    parser.add_argument(
        'dwiFile',
        help='Name of DWI.')
    parser.add_argument(
        'bvalFile',
        help='Name of b-value.')
    parser.add_argument(
        'bvecFile',
        help='Name of gradiet vectory.')
    parser.add_argument(
        'maskFile',
        help='Name of brain mask.')
    parser.add_argument(
        '-model', action="store", dest="model", type=str, default="anisoMAPL",
        help='anisoMAPL, anisoCMAP, anisoCMAPL, anisoMAP+, isoMAPL, isoCMAP, isoCMAPL, isoMAP+, (default: anisoMAPL).')
    parser.add_argument(
        '-big_delta', action="store", dest="big_delta", type=float, default=0.0218,
        help='time between pulses [s].')
    parser.add_argument(
        '-small_delta', action="store", dest="small_delta", type=float, default=0.0129,
        help='pulses duration in [s].')
    
    args = parser.parse_args()
    
    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)
        
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)
    gtab = gradient_table(bvalFile,bvecFile,
                          big_delta=args.big_delta,
                          small_delta=args.small_delta)
    
    match args.model:
        case "anisoMAPL":
            radial_order = 6
            map_model = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting="GCV",
                                            cvxpy_solver='MOSEK')
    
        case "anisoCMAP":
            radial_order = 6
            map_model = mapmri.MapmriModel(gtab,
                                            radial_order=radial_order,
                                            laplacian_regularization=False,
                                            positivity_constraint=True,
                                            cvxpy_solver='MOSEK')
            
        case "anisoCMAPL":
            radial_order = 6
            map_model = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting="GCV",
                                            positivity_constraint=True,
                                            cvxpy_solver='MOSEK')
    
        case "anisoMAP+":
            radial_order = 6
            map_model = mapmri.MapmriModel(gtab,
                                            radial_order=radial_order,
                                            laplacian_regularization=False,
                                            positivity_constraint=True,
                                            global_constraints=True,
                                            cvxpy_solver='MOSEK')
    
        case "isoMAPL":
            radial_order = 8
            map_model = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting="GCV",
                                            anisotropic_scaling=False,
                                            cvxpy_solver='MOSEK')
            
        case "isoCMAP":
            radial_order = 8
            map_model = mapmri.MapmriModel(gtab,
                                            radial_order=radial_order,
                                            laplacian_regularization=False,
                                            positivity_constraint=True,
                                            anisotropic_scaling=False,
                                            cvxpy_solver='MOSEK')
            
        case "isoCMAPL":
            radial_order = 8
            map_model = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting="GCV",
                                            positivity_constraint=True,
                                            anisotropic_scaling=False,
                                            cvxpy_solver='MOSEK')
            
        case "isoMAP+":
            radial_order = 8
            map_model = mapmri.MapmriModel(gtab,
                                            radial_order=radial_order,
                                            laplacian_regularization=False,
                                            positivity_constraint=True,
                                            global_constraints=True,
                                            anisotropic_scaling=False,
                                            cvxpy_solver='MOSEK')

    mapfit = map_model.fit(dwi_data, mask=mask_data)
    sphere = get_sphere('repulsion724')

    MSD   =  mapfit.msd()
    QIV   =  mapfit.qiv()
    RTOP  =  mapfit.rtop()
    RTAP  =  mapfit.rtap()
    RTPP  =  mapfit.rtpp()
    NG    =  mapfit.ng()
    NGper =  mapfit.ng_perpendicular()
    NGpar =  mapfit.ng_parallel()
    PDF   =  mapfit.pdf()
    NOLS  =  mapfit.norm_of_laplacian_signal()
    ISF   =  mapfit.isotropic_scale_factor()
    COEF  =  mapfit.mapmri_coeffs()

    RTOP_cortex_norm = RTOP / RTOP[mask_data>0].mean()
    RTAP_cortex_norm = RTAP / RTAP[mask_data>0].mean()
    RTPP_cortex_norm = RTPP / RTPP[mask_data>0].mean()

    SH = mapfit.odf_sh(s=2)
    ODF = mapfit.odf(sphere, s=2)

    outdir = os.path.join(subjectDirectory,'anisoMAPL')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    save_nifti(os.path.join(outdir,'MSD.nii.gz'), MSD, dwi_affine)
    save_nifti(os.path.join(outdir,'QIV.nii.gz'), QIV, dwi_affine)
    save_nifti(os.path.join(outdir,'RTOP.nii.gz'), RTOP, dwi_affine)
    save_nifti(os.path.join(outdir,'RTAP.nii.gz'), RTAP, dwi_affine)
    save_nifti(os.path.join(outdir,'RTPP.nii.gz'), RTPP, dwi_affine)
    save_nifti(os.path.join(outdir,'NG.nii.gz'), NG, dwi_affine)
    save_nifti(os.path.join(outdir,'NGper.nii.gz'), NGper, dwi_affine)
    save_nifti(os.path.join(outdir,'NGpar.nii.gz'), NGpar, dwi_affine)
    save_nifti(os.path.join(outdir,'ODF.nii.gz'), ODF, dwi_affine)
    save_nifti(os.path.join(outdir,'RTOP_cortex_norm.nii.gz'), RTOP_cortex_norm, dwi_affine)
    save_nifti(os.path.join(outdir,'RTAP_cortex_norm.nii.gz'), RTAP_cortex_norm, dwi_affine)
    save_nifti(os.path.join(outdir,'RTPP_cortex_norm.nii.gz'), RTPP_cortex_norm, dwi_affine)
    save_nifti(os.path.join(outdir,'PDF.nii.gz'), PDF, dwi_affine)
    save_nifti(os.path.join(outdir,'NOLS.nii.gz'), NOLS, dwi_affine)
    save_nifti(os.path.join(outdir,'ISF.nii.gz'), ISF, dwi_affine)
    save_nifti(os.path.join(outdir,'SH.nii.gz'), SH, dwi_affine)
    save_nifti(os.path.join(outdir,'COEF.nii.gz'), COEF, dwi_affine)

if __name__ == '__main__':
    main()