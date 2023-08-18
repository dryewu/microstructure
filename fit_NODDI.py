#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:07:33 2023

@author: wuye
"""

import argparse
import os
import amico 

#amico.setup()

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit NODDI model with AMICO",
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
        '-b0thr', action="store", dest="b0_thr", type=float, default=10,
        help='Threshold for select non-dwi image.')
    parser.add_argument(
        '-b0step', action="store", dest="b0_step", type=float, default=100,
        help='Threshold for normalize b-value.')

    args = parser.parse_args()
    
    subjectDirectory = args.subjectDirectory
    dwiFile = args.dwiFile
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)
    
    amico.util.fsl2scheme(bvalFile,bvecFile,schemeFilename=os.path.join(subjectDirectory,'NODDI.scheme'),bStep = args.b0_step)
    
    ae = amico.Evaluation(subjectDirectory, '.')
    ae.load_data(dwi_filename = dwiFile, scheme_filename = os.path.join(subjectDirectory,'NODDI.scheme'), mask_filename = maskFile, b0_thr = args.b0_thr)
    ae.set_model("NODDI")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()
    ae.fit()
    ae.save_results()
            
if __name__ == '__main__':
    main()
        
        
    
    