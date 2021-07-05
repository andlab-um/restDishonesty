#!/bin/tcsh -xef

for i in {103..104} {106..116} {118..132}
do
  cd ~/Desktop/MRI_dpabi/Sub_${i}/
  3dresample -master sFiltered_4DVolume.nii -inset ~/Desktop/rewarding_association-test_z_FDR_0.01.nii -prefix reward_resampled.nii
  3dROIstats -mask reward_resampled.nii  -1Dformat sFiltered_4DVolume.nii > s${i}_reward.1D
  3dTcorr1D -mask ~/Desktop/result_ply/mask/s${i}_automask.nii -pearson -prefix s${i}_reward.Tcorr1D.nii sFiltered_4DVolume.nii s${i}_reward.1D
  3dcalc -a s${i}_reward.Tcorr1D.nii -expr 'atanh(a)' -prefix s${i}_reward.z.nii
done


























