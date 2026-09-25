DICOMFolder="/home/borges/TESTE1/"

## BREAST
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TOTAL_MAMA -ta total -ot dicom_rtstruct -ho -bs --roi_subset spleen gallbladder liver stomach pancreas lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right esophagus trachea thyroid_gland duodenum vertebrae_L2 vertebrae_L1 vertebrae_T1 vertebrae_T11 vertebrae_T10 vertebrae_T9 vertebrae_T8 vertebrae_T7 vertebrae_T6 vertebrae_T5 vertebrae_T4 vertebrae_T3 vertebrae_T2 vertebrae_T1 vertebrae_C7 vertebrae_C6 vertebrae_C5 vertebrae_C4 vertebrae_C3 vertebrae_C2 vertebrae_C1 humerus_left humerus_right spinal_cord rib_left_1 rib_left_2 rib_left_3 rib_left_4 rib_left_5 rib_left_6 rib_left_7 rib_left_8 rib_left_9 rib_left_10 rib_left_11 rib_left_12 rib_right_1 rib_right_2 rib_right_3 rib_right_4 rib_right_5 rib_right_6 rib_right_7 rib_right_8 rib_right_9 rib_right_10 rib_right_11 rib_right_12 sternum costal_cartilages sternocleidomastoid_right sternocleidomastoid_left
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BREASTS -ta breasts -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/HEART -ta heartchambers_highres -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/CORO -ta coronary_arteries -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BODY -ta body -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TISSUE -ta tissue_4_types -ot dicom_rtstruct -ho -bs

python /home/borges/Codigos_AutoSegmentacao/BreastSplit.py $DICOMFolder $DICOMFolder/BREASTS.dcm
python /home/borges/Codigos_AutoSegmentacao/NothingBreaksLikeAHeart.py $DICOMFolder/HEART.dcm $DICOMFolder
python /home/borges/Codigos_AutoSegmentacao/IHaveABody.py $DICOMFolder/BODY.dcm $DICOMFolder

python /home/borges/Codigos_AutoSegmentacao/unir_rtstructs.py -o $DICOMFolder/GrupoMama.dcm  $DICOMFolder/Cardiac_area.dcm $DICOMFolder/BREASTS_separado.dcm $DICOMFolder/TOTAL_MAMA.dcm $DICOMFolder/CORO.dcm $DICOMFolder/CORPO.dcm #$DICOMFolder/TISSUE.dcm