#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:31:53 2023

@author: wuye
"""

import argparse
import os
from dipy.reconst import qtdmri
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti
 
def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit WMTI model with Dipy",
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
    
    args = parser.parse_args()
    
    subjectDirectory = args.subjectDirectory
    dwiFile = os.path.join(subjectDirectory, args.dwiFile)
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)
        
    dwi_data, dwi_affine = load_nifti(dwiFile, return_img=False)
    mask_data, mask_affine = load_nifti(maskFile, return_img=False)
    gtab = gradient_table(bvalFile,bvecFile)
        
    qtdmri_mod = qtdmri.QtdmriModel(
        gtab, radial_order=6, time_order=2,
        laplacian_regularization=True, laplacian_weighting='GCV',
        l1_regularization=True, l1_weighting='CV'
    )
    qtdmri_fit = qtdmri_mod.fit(dwi_data, mask=mask_data)
    RTOP = qtdmri_fit.rtop
    RTAP = qtdmri_fit.rtap
    RTPP = qtdmri_fit.rtpp
    QIV = qtdmri_fit.qiv
    MSD = qtdmri_fit.msd
  
    outdir = os.path.join(subjectDirectory,'QTDMRI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    save_nifti(os.path.join(outdir,'RTOP.nii.gz'), RTOP, dwi_affine)
    save_nifti(os.path.join(outdir,'RTAP.nii.gz'), RTAP, dwi_affine)
    save_nifti(os.path.join(outdir,'RTPP.nii.gz'), RTPP, dwi_affine)
    save_nifti(os.path.join(outdir,'QIV.nii.gz'), QIV, dwi_affine)
    save_nifti(os.path.join(outdir,'MSD.nii.gz'), MSD, dwi_affine)
         
if __name__ == '__main__':
    main()