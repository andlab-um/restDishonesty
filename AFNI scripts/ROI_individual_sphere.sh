#!/bin/tcsh -xef

for i in {102..104} {106..116} {118..132}
do
  cd ~/Desktop/MRI_dpabi/Sub_${i}/
  echo "-6 50 18" > tmp_moral.txt
  3dUndump -prefix moral_sphere.nii -master sFiltered_4DVolume.nii -srad 10 -xyz tmp_moral.txt
  3dresample -master sFiltered_4DVolume.nii -inset moral_sphere.nii -prefix moral_sp_resampled.nii
  3dROIstats -mask moral_sp_resampled.nii -1Dformat sFiltered_4DVolume.nii > s${i}_moral_sp.1D
  3dTcorr1D -mask ~/Desktop/result_ply/mask/s${i}_automask.nii -pearson -prefix s${i}_moral_sp.Tcorr1D.nii sFiltered_4DVolume.nii s${i}_moral_sp.1D
  3dcalc -a s${i}_moral_sp.Tcorr1D.nii -expr 'atanh(a)' -prefix s${i}_moral_sp.z.nii
done


























