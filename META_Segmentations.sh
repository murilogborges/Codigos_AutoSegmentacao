DICOMFolder="/home/borges/META01/"

## META
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TOTAL_META -ta total -ot dicom_rtstruct -ho -bs 
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BODY -ta body -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TISSUE -ta tissue_4_types -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/HEAD -ta head_glands_cavities -ot dicom_rtstruct -ho -bs
TotalSegmentator -i $DICOMFolder -o $DICOMFolder/LIVER -ta liver_lesions -ot dicom_rtstruct -ho -bs 

ARQUIVO="$DICOMFolder/TOTAL_META.dcm"
ESTRUTURA="brain"

python3 - <<EOF
import sys
import pydicom

arquivo = "$ARQUIVO"
estrutura = "$ESTRUTURA"

try:
    ds = pydicom.dcmread(arquivo)
    nomes = [roi.ROIName for roi in ds.StructureSetROISequence]
    if estrutura in nomes:
        sys.exit(0)  # encontrado
    else:
        sys.exit(1)  # não encontrado
except Exception as e:
    print("Erro ao ler DICOM:", e)
    sys.exit(2)
EOF

if [ $? -eq 0 ]; then
    TotalSegmentator -i $DICOMFolder -o $DICOMFolder/BRAIN -ta brain_structures -ot dicom_rtstruct -ho -bs
    TotalSegmentator -i $DICOMFolder -o $DICOMFolder/TONGUE -ta head_muscles -ot dicom_rtstruct -ho -bs
    TotalSegmentator -i $DICOMFolder -o $DICOMFolder/OPTICS -ta oculomotor_muscles -ot dicom_rtstruct -ho -bs
else
    echo "Estrutura '$ESTRUTURA' não encontrada em $ARQUIVO."
fi

python /home/borges/Codigos_AutoSegmentacao/IHaveABody.py $DICOMFolder/BODY.dcm $DICOMFolder
python /home/borges/Codigos_AutoSegmentacao/unir_rtstructs.py -o $DICOMFolder/GrupoMETA.dcm $DICOMFolder/TOTAL_META.dcm $DICOMFolder/CORPO.dcm $DICOMFolder/HEAD.dcm $DICOMFolder/BRAIN.dcm $DICOMFolder/TONGUE.dcm $DICOMFolder/OPTICS.dcm $DICOMFolder/LIVER.dcm

# rm $DICOMFolder/Cardiac_area.dcm $DICOMFolder/BREASTS_separado.dcm $DICOMFolder/TOTAL_MAMA.dcm $DICOMFolder/CORO.dcm $DICOMFolder/CORPO.dcm $DICOMFolder/BREASTS.dcm $DICOMFolder/HEART.dcm $DICOMFolder/BODY.dcm 