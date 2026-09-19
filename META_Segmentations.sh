DICOMFolder="/home/borges/META01/"

## META
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TOTAL_META -ta total -ot dicom_rtstruct -ho -bs 
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BODY -ta body -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TISSUE -ta tissue_4_types -ot dicom_rtstruct -ho -bs

python /home/borges/Codigos_AutoSegmentacao/IHaveABody.py $DICOMFolder/BODY.dcm $DICOMFolder

python /home/borges/Codigos_AutoSegmentacao/unir_rtstructs.py -o $DICOMFolder/GrupoMETA.dcm $DICOMFolder/TOTAL_META.dcm $DICOMFolder/CORPO.dcm

# rm $DICOMFolder/Cardiac_area.dcm $DICOMFolder/BREASTS_separado.dcm $DICOMFolder/TOTAL_MAMA.dcm $DICOMFolder/CORO.dcm $DICOMFolder/CORPO.dcm $DICOMFolder/BREASTS.dcm $DICOMFolder/HEART.dcm $DICOMFolder/BODY.dcm 