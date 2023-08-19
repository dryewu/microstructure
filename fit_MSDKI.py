#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 14:35:37 2023

@author: wuye
"""

import argparse
import os
import dipy.reconst.msdki as msdki
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti
 
def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit MSDKI model with Dipy",
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
    
    msdki_model = msdki.MeanDiffusionKurtosisModel(gtab)
    msdki_fit = msdki_model.fit(dwi_data, mask=mask_data)

    MSD = msdki_fit.msd
    MSK = msdki_fit.msk
    F = msdki_fit.smt2f
    DI = msdki_fit.smt2di
    uFA = msdki_fit.smt2uFA

    outdir = os.path.join(subjectDirectory,'MSDKI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    save_nifti(os.path.join(outdir,'MSD.nii.gz'), MSD, dwi_affine)
    save_nifti(os.path.join(outdir,'MSK.nii.gz'), MSK, dwi_affine)
    save_nifti(os.path.join(outdir,'F.nii.gz'), F, dwi_affine)
    save_nifti(os.path.join(outdir,'DI.nii.gz'), DI, dwi_affine)
    save_nifti(os.path.join(outdir,'uFA.nii.gz'), uFA, dwi_affine)
                
if __name__ == '__main__':
    main()