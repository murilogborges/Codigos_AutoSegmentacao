#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Uso: $0 <pasta_DICOM> [arquivo_RTSTRUCT_final]" >&2
}

if [[ $# -lt 1 || $# -gt 2 || ! -d "$1" ]]; then
    usage
    exit 2
fi

DICOMFolder="$(cd -- "$1" && pwd)"
OUTPUT_FILE="${2:-$DICOMFolder/GrupoMAMA.dcm}"
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="$(pwd)/$OUTPUT_FILE"
fi

TOTAL_STEPS=12
CURRENT_STEP=0
PROGRESS_LINE_ACTIVE=0
SEGMENTATION_LOG="$(mktemp "${TMPDIR:-/tmp}/totalsegmentator-mama.XXXXXX")"
trap 'clear_progress_line; rm -f -- "$SEGMENTATION_LOG"' EXIT

progress_start() {
    if [[ -t 2 ]]; then
        printf '\r\033[KProgresso geral: %3d%% (%d/%d) - Em andamento: %s' \
            "$((CURRENT_STEP * 100 / TOTAL_STEPS))" "$CURRENT_STEP" "$TOTAL_STEPS" "$1" >&2
        PROGRESS_LINE_ACTIVE=1
    fi
}

progress_step() {
    local label="$1"
    if [[ $PROGRESS_LINE_ACTIVE -eq 1 ]]; then
        printf '\r\033[K' >&2
        PROGRESS_LINE_ACTIVE=0
    fi
    CURRENT_STEP=$((CURRENT_STEP + 1))
    printf 'Progresso geral: %3d%% (%d/%d) - %s\n' \
        "$((CURRENT_STEP * 100 / TOTAL_STEPS))" "$CURRENT_STEP" "$TOTAL_STEPS" "$label" >&2
}

clear_progress_line() {
    if [[ $PROGRESS_LINE_ACTIVE -eq 1 ]]; then
        printf '\r\033[K' >&2
        PROGRESS_LINE_ACTIVE=0
    fi
}

show_failure_log() {
    if [[ -s "$SEGMENTATION_LOG" ]]; then
        cat "$SEGMENTATION_LOG" >&2
    fi
}

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Erro: arquivo esperado não foi gerado: $1" >&2
        exit 1
    fi
}

for output in TOTAL_MAMA BREASTS HEART CORO BODY TISSUE MUSC LIVER; do
    if [[ -e "$DICOMFolder/$output" ]]; then
        echo "Erro: saída já existe e não será sobrescrita: $DICOMFolder/$output" >&2
        exit 1
    fi
    if [[ -e "$DICOMFolder/$output.dcm" && ! -s "$DICOMFolder/$output.dcm" ]]; then
        echo "Erro: RTSTRUCT existente está vazio e não pode ser reutilizado: $DICOMFolder/$output.dcm" >&2
        exit 1
    fi
    if [[ ! -s "$DICOMFolder/$output.dcm" \
       && ( -e "$DICOMFolder/$output.nii" || -e "$DICOMFolder/$output.nii.gz" ) ]]; then
        echo "Erro: resíduo de execução anterior encontrado sem RTSTRUCT: $DICOMFolder/$output" >&2
        exit 1
    fi
done
if [[ -e "$DICOMFolder/CORPO.dcm" && ! -s "$DICOMFolder/CORPO.dcm" ]]; then
    echo "Erro: CORPO.dcm existente está vazio e não pode ser reutilizado." >&2
    exit 1
fi
if [[ -e "$OUTPUT_FILE" ]]; then
    echo "Erro: saída já existe e não será sobrescrita: $OUTPUT_FILE" >&2
    exit 1
fi
for input_stem in TOTAL_MAMA BREASTS_separado Cardiac_area CORO CORPO MUSC LIVER; do
    if [[ "$OUTPUT_FILE" == "$DICOMFolder/$input_stem.dcm" ]]; then
        echo "Erro: o RTSTRUCT final não pode sobrescrever uma entrada do fluxo: $OUTPUT_FILE" >&2
        exit 2
    fi
done

remove_failed_outputs() {
    local output_name="$1"
    rm -f -- "$DICOMFolder/$output_name.dcm" \
        "$DICOMFolder/$output_name.nii" \
        "$DICOMFolder/$output_name.nii.gz"
}

run_segmentation() {
    local output_name="$1"
    local task="$2"
    shift 2

    if [[ -s "$DICOMFolder/$output_name.dcm" ]]; then
        return 0
    fi

    progress_start "TotalSegmentator: $task"
    if ! TotalSegmentator -i "$DICOMFolder" -o "$DICOMFolder/$output_name" \
        -ta "$task" -ot dicom_rtstruct "$@" >"$SEGMENTATION_LOG" 2>&1; then
        clear_progress_line
        remove_failed_outputs "$output_name"
        return 1
    fi
    if [[ ! -s "$DICOMFolder/$output_name.dcm" ]]; then
        clear_progress_line
        remove_failed_outputs "$output_name"
        return 1
    fi
    return 0
}

rtstruct_inputs=()
if run_segmentation TOTAL_MAMA total -ho -bs --roi_subset spleen gallbladder liver stomach pancreas lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right esophagus trachea thyroid_gland duodenum vertebrae_L2 vertebrae_L1 vertebrae_T1 vertebrae_T11 vertebrae_T10 vertebrae_T9 vertebrae_T8 vertebrae_T7 vertebrae_T6 vertebrae_T5 vertebrae_T4 vertebrae_T3 vertebrae_T2 vertebrae_T1 vertebrae_C7 vertebrae_C6 vertebrae_C5 vertebrae_C4 vertebrae_C3 vertebrae_C2 vertebrae_C1 humerus_left humerus_right spinal_cord rib_left_1 rib_left_2 rib_left_3 rib_left_4 rib_left_5 rib_left_6 rib_left_7 rib_left_8 rib_left_9 rib_left_10 rib_left_11 rib_left_12 rib_right_1 rib_right_2 rib_right_3 rib_right_4 rib_right_5 rib_right_6 rib_right_7 rib_right_8 rib_right_9 rib_right_10 rib_right_11 rib_right_12 sternum costal_cartilages; then
    rtstruct_inputs+=("$DICOMFolder/TOTAL_MAMA.dcm")
fi
progress_step "Segmentação total"

breasts_available=0
if run_segmentation BREASTS breasts -ho -bs; then
    breasts_available=1
fi
progress_step "Segmentação breasts"
if [[ $breasts_available -eq 1 ]]; then
    progress_start "Separação das mamas"
    rm -f -- "$DICOMFolder/BREASTS_separado.dcm"
    if python3 "$SCRIPT_DIR/BreastSplit.py" "$DICOMFolder" "$DICOMFolder/BREASTS.dcm" \
        >"$SEGMENTATION_LOG" 2>&1 \
        && [[ -s "$DICOMFolder/BREASTS_separado.dcm" ]]; then
        rtstruct_inputs+=("$DICOMFolder/BREASTS_separado.dcm")
    else
        clear_progress_line
        rm -f -- "$DICOMFolder/BREASTS_separado.dcm"
    fi
fi
progress_step "Separação das mamas"

heart_available=0
if run_segmentation HEART heartchambers_highres -ho -bs; then
    heart_available=1
fi
progress_step "Segmentação heartchambers_highres"
if [[ $heart_available -eq 1 ]]; then
    progress_start "Preparação das estruturas cardíacas"
    if python3 "$SCRIPT_DIR/NothingBreaksLikeAHeart.py" "$DICOMFolder/HEART.dcm" "$DICOMFolder" \
        >"$SEGMENTATION_LOG" 2>&1 \
        && [[ -s "$DICOMFolder/Cardiac_area.dcm" ]]; then
        rtstruct_inputs+=("$DICOMFolder/Cardiac_area.dcm")
    else
        clear_progress_line
    fi
fi
progress_step "Preparação das estruturas cardíacas"

if run_segmentation CORO coronary_arteries -ho -bs; then
    rtstruct_inputs+=("$DICOMFolder/CORO.dcm")
fi
progress_step "Segmentação coronary_arteries"

body_available=0
if [[ -s "$DICOMFolder/CORPO.dcm" ]]; then
    body_available=2
elif run_segmentation BODY body -ho -bs; then
    body_available=1
fi
progress_step "Segmentação body"
if [[ $body_available -ne 1 ]]; then
    if [[ $body_available -ne 2 ]]; then
        echo "Erro: não foi possível gerar o arquivo final porque a segmentação 'body' falhou; verifique TotalSegmentator." >&2
        show_failure_log
        exit 1
    fi
else
    progress_start "Criação da estrutura corpo"
    if ! python3 "$SCRIPT_DIR/IHaveABody.py" "$DICOMFolder/BODY.dcm" "$DICOMFolder" \
        >"$SEGMENTATION_LOG" 2>&1 \
        || [[ ! -s "$DICOMFolder/CORPO.dcm" ]]; then
        clear_progress_line
        echo "Erro: não foi possível gerar o arquivo final porque IHaveABody.py não produziu CORPO.dcm." >&2
        show_failure_log
        exit 1
    fi
fi
rtstruct_inputs+=("$DICOMFolder/CORPO.dcm")
progress_step "Criação da estrutura corpo"

run_segmentation TISSUE tissue_4_types -ho -bs || true
progress_step "Segmentação tissue_4_types"

if run_segmentation MUSC headneck_muscles -ho -bs; then
    rtstruct_inputs+=("$DICOMFolder/MUSC.dcm")
fi
progress_step "Segmentação headneck_muscles"
if run_segmentation LIVER liver_lesions -ho -bs; then
    rtstruct_inputs+=("$DICOMFolder/LIVER.dcm")
fi
progress_step "Segmentação liver_lesions"

progress_start "Consolidação do RTSTRUCT"
if ! python3 "$SCRIPT_DIR/unir_rtstructs.py" \
    -o "$OUTPUT_FILE" \
    "${rtstruct_inputs[@]}" >"$SEGMENTATION_LOG" 2>&1; then
    clear_progress_line
    echo "Erro: não foi possível gerar o RTSTRUCT final em $OUTPUT_FILE." >&2
    show_failure_log
    exit 1
fi
require_file "$OUTPUT_FILE"
progress_step "Consolidação do RTSTRUCT"
