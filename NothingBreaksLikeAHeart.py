#!/usr/bin/env python3
import sys
import os
import tempfile
import cv2
import numpy as np
from scipy.ndimage import label, binary_closing, binary_fill_holes, gaussian_filter
from rt_utils import RTStructBuilder, image_helper


def _save_rtstruct_atomic(rtstruct, output_path):
    output_dir = os.path.dirname(output_path)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".Cardiac_area.",
        suffix=".dcm",
        dir=output_dir,
    )
    os.close(file_descriptor)
    try:
        rtstruct.save(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _get_roi_mask(rtstruct, roi_name):
    roi_number = next(
        (
            roi.ROINumber
            for roi in rtstruct.ds.StructureSetROISequence
            if roi.ROIName == roi_name
        ),
        None,
    )
    if roi_number is None:
        raise ValueError(f"ROI '{roi_name}' não encontrada.")

    contours = [
        contour
        for roi_contour in getattr(rtstruct.ds, "ROIContourSequence", [])
        if str(getattr(roi_contour, "ReferencedROINumber", "")) == str(roi_number)
        for contour in getattr(roi_contour, "ContourSequence", [])
    ]
    combined = None
    for contour in contours:
        if getattr(contour, "ContourGeometricType", "") not in {
            "CLOSED_PLANAR",
            "CLOSEDPLANAR_XOR",
        }:
            continue
        try:
            data = np.asarray(contour.ContourData, dtype=float)
            point_count = int(
                getattr(contour, "NumberOfContourPoints", data.size // 3)
            )
            if (
                data.size < 9
                or data.size % 3
                or point_count != data.size // 3
                or not np.isfinite(data).all()
                or np.unique(data.reshape(-1, 3), axis=0).shape[0] < 3
            ):
                continue
            mask = image_helper.create_series_mask_from_contour_sequence(
                rtstruct.series_data,
                [contour],
            )
        except (cv2.error, ValueError, IndexError, KeyError, AttributeError, TypeError):
            continue

        if combined is None:
            combined = np.asarray(mask, dtype=bool)
        elif getattr(contour, "ContourGeometricType", "") == "CLOSEDPLANAR_XOR":
            combined = np.logical_xor(combined, mask)
        else:
            combined = np.logical_or(combined, mask)

    if combined is None:
        return None
    return combined

def simplify_mask(mask, sigma=8.5):
    """Simplifica a máscara para um único volume contínuo e suavizado por corte."""
    simplified = np.zeros_like(mask)

    for z in range(mask.shape[2]):
        slice_mask = mask[:, :, z]
        if slice_mask.sum() == 0:
            continue

        slice_mask = binary_fill_holes(slice_mask)
        slice_mask = binary_closing(slice_mask, structure=np.ones((5,5)))

        labeled, ncomponents = label(slice_mask)
        if ncomponents > 0:
            sizes = [(labeled == comp).sum() for comp in range(1, ncomponents+1)]
            largest_comp = np.argmax(sizes) + 1
            slice_mask = (labeled == largest_comp)

        simplified[:, :, z] = slice_mask

    smoothed = gaussian_filter(simplified.astype(float), sigma=sigma)
    return smoothed > 0.5

def connect_global(mask):
    """Liga pontos em 3D para garantir um único volume contínuo e homogêneo."""
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, structure=np.ones((7,7,7)))
    labeled, ncomponents = label(mask)
    if ncomponents > 0:
        sizes = [(labeled == comp).sum() for comp in range(1, ncomponents+1)]
        largest_comp = np.argmax(sizes) + 1
        mask = (labeled == largest_comp)
    return mask

def main():
    if len(sys.argv) != 3:
        print("Uso: python NothingBreaksLikeAHeart.py <rtstruct> <pasta_das_imagens>")
        sys.exit(1)

    rtstruct_path = sys.argv[1]
    dicom_series_path = sys.argv[2]

    output_dir = os.path.dirname(rtstruct_path)
    output_path = os.path.join(output_dir, "Cardiac_area.dcm")

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path,
        rt_struct_path=rtstruct_path
    )

    all_structures = rtstruct.get_roi_names()

    # --- Pulmonary artery para definir corte inicial ---
    pa_mask = _get_roi_mask(rtstruct, "pulmonary_artery")
    if pa_mask is None or np.asarray(pa_mask).ndim != 3:
        raise RuntimeError("Máscara 'pulmonary_artery' ausente ou inválida.")
    pa_mask = np.asarray(pa_mask, dtype=bool)
    if not pa_mask.any():
        raise RuntimeError("Máscara 'pulmonary_artery' está vazia.")

    start_slice = None
    bifurcation_detected = False
    for z in range(pa_mask.shape[2]-1, -1, -1):
        slice_mask = pa_mask[:, :, z]
        labeled, ncomponents = label(slice_mask)
        if ncomponents >= 2:
            bifurcation_detected = True
        elif bifurcation_detected and ncomponents == 1:
            start_slice = max(0, z-1)
            break
    if start_slice is None:
        start_slice = pa_mask.shape[2] - 1

    merged_mask = np.zeros_like(pa_mask, dtype=bool)
    mandatory_mask = np.zeros_like(pa_mask, dtype=bool)
    found_structures = 0

    for struct in all_structures:
        mask = _get_roi_mask(rtstruct, struct)
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != pa_mask.shape:
            raise ValueError(
                f"A máscara '{struct}' tem shape {mask.shape}, "
                f"diferente de pulmonary_artery {pa_mask.shape}."
            )
        if not mask.any():
            continue
        found_structures += 1

        # Tratamento especial para aorta
        if struct.lower() == "aorta":
            refined_mask = np.zeros_like(mask)
            for z in range(mask.shape[2]):
                slice_mask = mask[:, :, z]
                labeled, ncomponents = label(slice_mask)
                if ncomponents == 0:
                    continue
                if ncomponents == 1:
                    refined_mask[:, :, z] = slice_mask
                    continue
                else:
                    best_component = None
                    best_y = mask.shape[0]
                    for comp in range(1, ncomponents+1):
                        coords = np.argwhere(labeled == comp)
                        if coords.size == 0:
                            continue
                        min_y = coords[:,0].min()
                        if min_y < best_y:
                            best_y = min_y
                            best_component = comp
                    if best_component is not None:
                        refined_mask[:, :, z] = (labeled == best_component)
            mask = refined_mask

        # Guardar atrium, ventricle e myocardium separadamente
        if ("atrium" in struct.lower() or 
            "ventricle" in struct.lower() or 
            "myocardium" in struct.lower()):
            mandatory_mask |= mask

        merged_mask |= mask

    if found_structures == 0 or not merged_mask.any():
        raise RuntimeError("Nenhuma máscara cardíaca válida foi encontrada.")

    merged_mask[:, :, start_slice+1:] = False

    # Primeira suavização
    simplified_mask = simplify_mask(merged_mask, sigma=2.5)

    # Inclusão obrigatória
    if mandatory_mask.any():
        simplified_mask = np.logical_or(simplified_mask, mandatory_mask)

    # Segunda suavização + liga pontos global
    final_mask = simplify_mask(simplified_mask, sigma=3.0)
    final_mask = connect_global(final_mask)

    # Verificação final: garantir inclusão obrigatória
    if mandatory_mask.any():
        final_mask = np.logical_or(final_mask, mandatory_mask)
    if not final_mask.any():
        raise RuntimeError("A máscara Cardiac_area ficou vazia após o processamento.")

    rtstruct.add_roi(
        mask=final_mask,
        name="Cardiac_area",
        color=[255, 0, 0]
    )

    _save_rtstruct_atomic(rtstruct, output_path)

if __name__ == "__main__":
    main()
