#!/bin/tcsh -xef
cd ~/Desktop/MRI_dpabi/

for i in {102..104} {106..116} {118..132}
do
  cp Sub_${i}/s${i}_moral_sp.z.nii ~/Desktop/result_ply/z-moral_sp

done

cd ~/Desktop/result_ply/z-moral_sp
3dbucket -prefix all_sub.nii *.z*.nii
cp ~/Desktop/lierate.txt ~/Desktop/result_ply/z-moral_sp
cp ~/Desktop/ch2.nii ~/Desktop/result_ply/z-moral_sp
3dTcorr1D -prefix corr_test.nii all_sub.nii lierate.txt

























