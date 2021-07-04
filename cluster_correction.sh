

for i in {102..104} {106..116} {118..132}
do
  cd ~/Desktop/FunImgS_ply/Sub_${i}
  cp s${i}_automask.nii ~/Desktop/result_ply/mask
done

cd ~/Desktop/result_ply/mask
3dMean -mask_inter -prefix group_mask.nii *automask*.nii

3dClustSim -mask group_mask.nii -acf 0.721286897 7.260865517 15.54216552
























