#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Uso: $0 <pasta_DICOM>" >&2
    echo "O nome da pasta deve conter exatamente um grupo: META, MAMA ou PELVE." >&2
}

if [[ $# -ne 1 || ! -d "$1" ]]; then
    usage
    exit 2
fi

DICOMFolder="$(cd -- "$1" && pwd)"
folder_name="${DICOMFolder##*/}"
folder_name="${folder_name,,}"
groups=()
for group in meta mama pelve; do
    if [[ "$folder_name" == *"$group"* ]]; then
        groups+=("$group")
    fi
done

if [[ ${#groups[@]} -ne 1 ]]; then
    echo "Erro: o nome da pasta '$folder_name' deve conter exatamente um dos grupos META, MAMA ou PELVE." >&2
    exit 2
fi

mama_output="$DICOMFolder/GrupoMAMA.dcm"
if [[ "${groups[0]}" == "mama" ]]; then
    normalized_name="${folder_name//[^a-z0-9]/}"
    if [[ "$normalized_name" =~ mama(dir|ld|d|esq|le|e)?[0-9]+(dir|ld|d|esq|le|e)? ]]; then
        prefix_laterality="${BASH_REMATCH[1]:-}"
        suffix_laterality="${BASH_REMATCH[2]:-}"
        case "$prefix_laterality" in
            d|dir|ld) prefix_side=right ;;
            e|esq|le) prefix_side=left ;;
            *) prefix_side="" ;;
        esac
        case "$suffix_laterality" in
            d|dir|ld) suffix_side=right ;;
            e|esq|le) suffix_side=left ;;
            *) suffix_side="" ;;
        esac
        if [[ -n "$prefix_side" && -n "$suffix_side" && "$prefix_side" != "$suffix_side" ]]; then
            echo "Erro: lateralidade ambígua no nome da pasta '$folder_name'." >&2
            exit 2
        fi
        mama_side="${prefix_side:-$suffix_side}"
        if [[ "$mama_side" == "right" ]]; then
            mama_output="$DICOMFolder/GrupoMAMA_LD.dcm"
        elif [[ "$mama_side" == "left" ]]; then
            mama_output="$DICOMFolder/GrupoMAMA_LE.dcm"
        fi
    fi
fi

if ! command -v TotalSegmentator >/dev/null 2>&1; then
    echo "Erro: TotalSegmentator não está disponível no PATH." >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Erro: python3 não está disponível no PATH." >&2
    exit 127
fi
if ! python3 -c 'import cv2, numpy, pydicom, rt_utils, scipy, shapely' >/dev/null 2>&1; then
    echo "Erro: faltam dependências Python (cv2, numpy, pydicom, rt_utils, scipy ou shapely)." >&2
    exit 1
fi

case "${groups[0]}" in
    mama)
        bash "$SCRIPT_DIR/MAMA_Segmentations.sh" "$DICOMFolder" "$mama_output"
        ;;
    meta)
        bash "$SCRIPT_DIR/META_Segmentations.sh" "$DICOMFolder"
        ;;
    pelve)
        bash "$SCRIPT_DIR/META_Segmentations.sh" \
            "$DICOMFolder" "$DICOMFolder/GrupoPELVE.dcm"
        ;;
esac
