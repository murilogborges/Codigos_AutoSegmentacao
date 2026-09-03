#!/usr/bin/env python3
import sys
import os
import numpy as np
from scipy.ndimage import label, binary_closing, binary_fill_holes, gaussian_filter
from rt_utils import RTStructBuilder

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
    print("Estruturas encontradas:", all_structures)

    # --- Pulmonary artery para definir corte inicial ---
    try:
        pa_mask = rtstruct.get_roi_mask_by_name("pulmonary_artery")
    except Exception as e:
        print("Erro ao obter pulmonary_artery:", e)
        sys.exit(1)

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
        start_slice = 0
    print(f"Corte inicial da Cardiac_area: {start_slice}")

    merged_mask = None
    mandatory_mask = None  # atrium, ventricle, myocardium

    for struct in all_structures:
        try:
            mask = rtstruct.get_roi_mask_by_name(struct)
        except Exception as e:
            print(f"Erro ao obter {struct}: {e}")
            continue

        # Tratamento especial para aorta
        if struct.lower() == "aorta":
            refined_mask = np.zeros_like(mask)
            for z in range(mask.shape[2]):
                slice_mask = mask[:, :, z]
                labeled, ncomponents = label(slice_mask)
                if ncomponents <= 1:
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
            if mandatory_mask is None:
                mandatory_mask = mask
            else:
                mandatory_mask = np.logical_or(mandatory_mask, mask)

        if merged_mask is None:
            merged_mask = mask
        else:
            merged_mask = np.logical_or(merged_mask, mask)

    merged_mask[:, :, start_slice+1:] = False

    # Primeira suavização
    simplified_mask = simplify_mask(merged_mask, sigma=2.5)

    # Inclusão obrigatória
    if mandatory_mask is not None:
        simplified_mask = np.logical_or(simplified_mask, mandatory_mask)

    # Segunda suavização + liga pontos global
    final_mask = simplify_mask(simplified_mask, sigma=3.0)
    final_mask = connect_global(final_mask)

    # Verificação final: garantir inclusão obrigatória
    if mandatory_mask is not None:
        final_mask = np.logical_or(final_mask, mandatory_mask)
        final_mask = connect_global(final_mask)

    rtstruct.add_roi(
        mask=final_mask,
        name="Cardiac_area",
        color=[255, 0, 0]
    )

    rtstruct.save(output_path)
    print(f"Novo RTSTRUCT salvo em: {output_path}")

if __name__ == "__main__":
    main()
