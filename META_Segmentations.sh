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
OUTPUT_FILE="${2:-$DICOMFolder/GrupoMETA.dcm}"
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="$(pwd)/$OUTPUT_FILE"
fi

TOTAL_STEPS=10
CURRENT_STEP=0
PROGRESS_LINE_ACTIVE=0
SEGMENTATION_LOG="$(mktemp "${TMPDIR:-/tmp}/totalsegmentator-meta.XXXXXX")"
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

for output in TOTAL_META BODY TISSUE HEAD LIVER BRAIN TONGUE OPTICS; do
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
if [[ -e "$OUTPUT_FILE" ]]; then
    echo "Erro: saída final já existe e não será sobrescrita: $OUTPUT_FILE" >&2
    exit 1
fi
if [[ -e "$DICOMFolder/CORPO.dcm" && ! -s "$DICOMFolder/CORPO.dcm" ]]; then
    echo "Erro: CORPO.dcm existente está vazio e não pode ser reutilizado." >&2
    exit 1
fi
for input_stem in TOTAL_META BODY TISSUE HEAD LIVER BRAIN TONGUE OPTICS CORPO; do
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
for module in \
    "TOTAL_META total" \
    "BODY body" \
    "TISSUE tissue_4_types" \
    "HEAD head_glands_cavities" \
    "LIVER liver_lesions"
do
    read -r output_name task <<< "$module"
    if [[ "$output_name" == "BODY" && -s "$DICOMFolder/CORPO.dcm" ]]; then
        progress_step "Reutilização de CORPO.dcm; segmentação body ignorada"
        continue
    fi
    if run_segmentation "$output_name" "$task" -ho -bs; then
        if [[ "$output_name" != "TISSUE" && "$output_name" != "BODY" ]]; then
            rtstruct_inputs+=("$DICOMFolder/$output_name.dcm")
        fi
    elif [[ "$output_name" == "BODY" ]]; then
        progress_step "Segmentação $task"
        echo "Erro: não foi possível gerar o arquivo final porque a segmentação 'body' falhou; verifique TotalSegmentator." >&2
        show_failure_log
        exit 1
    fi
    progress_step "Segmentação $task"
done

brain_status=1
if [[ -s "$DICOMFolder/TOTAL_META.dcm" ]]; then
    set +e
    python3 -c '
import sys
import pydicom

try:
    dataset = pydicom.dcmread(sys.argv[1], stop_before_pixels=True)
except Exception as error:
    print(f"Erro ao ler RTSTRUCT para verificar brain: {error}", file=sys.stderr)
    raise SystemExit(2)
names = [getattr(roi, "ROIName", "") for roi in getattr(dataset, "StructureSetROISequence", [])]
raise SystemExit(0 if "brain" in names else 1)
' "$DICOMFolder/TOTAL_META.dcm" >"$SEGMENTATION_LOG" 2>&1
    brain_status=$?
    set -e
fi

if [[ $brain_status -eq 0 ]]; then
    for module in \
        "BRAIN brain_structures" \
        "TONGUE head_muscles" \
        "OPTICS oculomotor_muscles"
    do
        read -r output_name task <<< "$module"
        if run_segmentation "$output_name" "$task" -ho -bs; then
            rtstruct_inputs+=("$DICOMFolder/$output_name.dcm")
        fi
        progress_step "Segmentação $task"
    done
elif [[ $brain_status -eq 1 ]]; then
    progress_step "Tarefas cerebrais ignoradas"
    progress_step "Tarefas cerebrais ignoradas"
    progress_step "Tarefas cerebrais ignoradas"
else
    progress_step "Tarefas cerebrais ignoradas"
    progress_step "Tarefas cerebrais ignoradas"
    progress_step "Tarefas cerebrais ignoradas"
fi

if [[ -s "$DICOMFolder/CORPO.dcm" ]]; then
    :
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
progress_step "Criação/reutilização da estrutura corpo"

progress_start "Consolidação do RTSTRUCT"
if ! python3 "$SCRIPT_DIR/unir_rtstructs.py" \
    -o "$OUTPUT_FILE" \
    "${rtstruct_inputs[@]}" >"$SEGMENTATION_LOG" 2>&1; then
    clear_progress_line
    echo "Erro: não foi possível gerar o RTSTRUCT final em $OUTPUT_FILE." >&2
    show_failure_log
    exit 1
fi
if [[ ! -s "$OUTPUT_FILE" ]]; then
    echo "Erro: arquivo consolidado não foi gerado: $OUTPUT_FILE" >&2
    show_failure_log
    exit 1
fi
progress_step "Consolidação do RTSTRUCT"
