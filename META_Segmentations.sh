DICOMFolder="/home/borges/META01/"

## META
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TOTAL_META -ta total -ot dicom_rtstruct -ho -bs 
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BODY -ta body -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TISSUE -ta tissue_4_types -ot dicom_rtstruct -ho -bs

python IHaveABody.py $DICOMFolder/BODY.dcm "#55ffff"

python unir_rtstructs.py -o $DICOMFolder/TOTAL_META.dcm $DICOMFolder/CORPO.dcm $DICOMFolder/TISSUE.dcm

# rm $DICOMFolder/Cardiac_area.dcm $DICOMFolder/BREASTS_separado.dcm $DICOMFolder/TOTAL_MAMA.dcm $DICOMFolder/CORO.dcm $DICOMFolder/CORPO.dcm $DICOMFolder/BREASTS.dcm $DICOMFolder/HEART.dcm $DICOMFolder/BODY.dcm 