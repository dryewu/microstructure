#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:44:35 2023

@author: wuye
"""

import argparse
import os
import dipy.reconst.dki_micro as dki_micro
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
    
    dki_micro_model = dki_micro.KurtosisMicrostructureModel(gtab)
    dki_micro_fit = dki_micro_model.fit(dwi_data, mask=mask_data)

    AWF = dki_micro_fit.awf
    Tortuosity = dki_micro_fit.tortuosity
    Restricted = dki_micro_fit.restricted_evals
    Hindered = dki_micro_fit.hindered_evals
    Axonal = dki_micro_fit.axonal_diffusivity
    Hindered_AD = dki_micro_fit.hindered_ad
    Hindered_RD = dki_micro_fit.hindered_rd

    outdir = os.path.join(subjectDirectory,'WMTI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    save_nifti(os.path.join(outdir,'AWF.nii.gz'), AWF, dwi_affine)
    save_nifti(os.path.join(outdir,'Tortuosity.nii.gz'), Tortuosity, dwi_affine)
    save_nifti(os.path.join(outdir,'Restricted.nii.gz'), Restricted, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered.nii.gz'), Hindered, dwi_affine)
    save_nifti(os.path.join(outdir,'Axonal.nii.gz'), Axonal, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered_AD.nii.gz'), Hindered_AD, dwi_affine)
    save_nifti(os.path.join(outdir,'Hindered_RD.nii.gz'), Hindered_RD, dwi_affine)
          
if __name__ == '__main__':
    main()