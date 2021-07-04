#!/bin/tcsh -xef

for i in {102..132}
do
  cd ~/Desktop/RawMRI/Sub_${i}
  3drefit -space ORIG ${i}_resting_pre_4D.nii
  3drefit -space ORIG ${i}_T1.nii
  afni_proc.py -subj_id s${i}.REST \
    -dsets ${i}_resting_pre_4D.nii \
    -copy_anat ${i}_T1.nii \
    -blocks despike tshift align tlrc volreg blur mask regress \
    -tlrc_base MNI152_T1_2009c+tlrc \
    -volreg_align_e2a \
    -volreg_tlrc_warp \
    -regress_anaticor \
    -regress_censor_motion 0.2 \
    -regress_censor_outliers 0.1 \
    -regress_bandpass 0.01 0.1 \
    -regress_apply_mot_types demean deriv \
    -regress_run_clustsim no \
    -regress_est_blur_epits \
    -regress_est_blur_errts \
    -execute
done



