#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:12:54 2023

@author: wuye
"""

import argparse
import os
import amico 
import numpy as np

amico.core.setup()

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Fit SANDI model with AMICO",
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
    parser.add_argument(
        '-TE', action="store", dest="TE", type=float, default=0.030,
        help='echo time if different from delta+small_delta [s] (optional).')
    parser.add_argument(
        '-Delta', action="store", dest="Delta", type=float, default=0.020,
        help='time between pulses [s].')
    parser.add_argument(
        '-delta', action="store", dest="delta", type=float, default=0.0055,
        help='pulses duration in [s].')
    
    args = parser.parse_args()
    
    subjectDirectory = args.subjectDirectory
    dwiFile = args.dwiFile
    bvalFile = os.path.join(subjectDirectory, args.bvalFile)
    bvecFile = os.path.join(subjectDirectory, args.bvecFile)
    maskFile = os.path.join(subjectDirectory, args.maskFile)
    
    amico.util.sandi2scheme(bvalFile, bvecFile, args.Delta, args.delta, TE_data=args.TE, schemeFilename=os.path.join(subjectDirectory,'SANDI.scheme'),bStep = args.b0_step)
    
    ae = amico.Evaluation(subjectDirectory, '.')
    ae.set_config("doDebiasSignal",False)
    ae.set_config('doDirectionalAverage', True)
    
    ae.load_data(dwi_filename = dwiFile, scheme_filename = os.path.join(subjectDirectory,'SANDI.scheme'), mask_filename = maskFile, b0_thr = args.b0_thr)
    ae.set_model("SANDI")
    
    d_is = 3.0E-3        # Intra-soma diffusivity [mm^2/s]
    Rs = np.linspace(1.0,12.0,5) * 1E-6           # Radii of the soma [meters]
    d_in = np.linspace(0.25,3.0,5) * 1E-3         # Intra-neurite diffusivitie(s) [mm^2/s]
    d_isos = np.linspace(0.25,3.0,5) * 1E-3       # Extra-cellular isotropic mean diffusivitie(s) [mm^2/s]

    ae.model.set(d_is, Rs, d_in, d_isos) 
    ae.generate_kernels(regenerate=True, ndirs=1)
    ae.load_kernels()
    
    lambda1 = 0  
    lambda2 = 5.0E-3
    ae.set_solver( lambda1=lambda1, lambda2=lambda2 ) 
    ae.fit()
    ae.save_results(save_dir_avg=True)
            
if __name__ == '__main__':
    main()
        
        
    
    