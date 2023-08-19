#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:36:02 2023

@author: wuye
"""

import argparse
import os
import dipy.reconst.fwdti as fwdti
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti, save_nifti
 
def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit FreeWater model with Dipy",
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
    
    fwdtimodel = fwdti.FreeWaterTensorModel(gtab)
    fwdtifit = fwdtimodel.fit(dwi_data, mask=mask_data)

    FA = fwdtifit.fa
    MD = fwdtifit.md
    FW = fwdtifit.f
    RD = fwdtifit.rd
    AD = fwdtifit.ad
    ADC = fwdtifit.adc

    outdir = os.path.join(subjectDirectory,'FWDTI')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    save_nifti(os.path.join(outdir,'FA.nii.gz'), FA, dwi_affine)
    save_nifti(os.path.join(outdir,'MD.nii.gz'), MD, dwi_affine)
    save_nifti(os.path.join(outdir,'FW.nii.gz'), FW, dwi_affine)
    save_nifti(os.path.join(outdir,'RD.nii.gz'), RD, dwi_affine)
    save_nifti(os.path.join(outdir,'AD.nii.gz'), AD, dwi_affine)
    save_nifti(os.path.join(outdir,'ADC.nii.gz'), ADC, dwi_affine)
                
if __name__ == '__main__':
    main()
        
        
    
    