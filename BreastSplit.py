#!/usr/bin/env python3
"""
Uso:
    python BreastSplit.py <dicom_series_path> <rt_struct_path>

Exemplo:
    python BreastSplit.py /home/borges/4/ScalarVolume_7/ /home/borges/4/breasts.dcm
"""
import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from scipy.ndimage import binary_closing, binary_opening
from rt_utils import RTStructBuilder

# ------------------ parâmetros ajustáveis ------------------
SMOOTH_SIGMA = 1.2         # suavização inicial para reduzir ruído
MIN_COMPONENT_VOXELS = 30  # ignora componentes muito pequenos (ruído)
MORPH_ITER = 1             # iterações de fechamento/abertura morfológica
FINAL_SMOOTH_SIGMA = 0.6   # suavização final antes do threshold
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python BreastSplitLR_forceLR.py <dicom_series_path> <rt_struct_path>")
        sys.exit(1)

    dicom_series_path = sys.argv[1]
    rt_struct_path = sys.argv[2]

    # Carregar RTSTRUCT e máscara original
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path,
        rt_struct_path=rt_struct_path
    )

    mask = rtstruct.get_roi_mask_by_name("breast")
    if mask is None:
        raise RuntimeError("Máscara 'breast' não encontrada no RTSTRUCT.")
    mask = np.asarray(mask).astype(bool)  # shape: (slices, rows, cols)

    # Pré-processamento: suavizar e binarizar para reduzir ruído (mantendo bool)
    mask_smooth = gaussian_filter(mask.astype(float), sigma=SMOOTH_SIGMA) > 0.5

    # Labeling 3D para encontrar componentes conectados
    structure = generate_binary_structure(3, 2)  # conectividade 3D (faces+arestas)
    labeled, ncomp = label(mask_smooth, structure=structure)

    # Coletar componentes válidas (tamanho e centroides em X)
    components = []
    for lab in range(1, ncomp + 1):
        coords = np.where(labeled == lab)  # tuple (z, y, x)
        count = coords[0].size
        if count < MIN_COMPONENT_VOXELS:
            continue
        centroid_x = coords[2].mean()  # média do índice X (coluna)
        components.append({
            "label": lab,
            "count": int(count),
            "centroid_x": float(centroid_x),
            "coords": coords
        })

    # Se não houver componentes após filtro, usar máscara suavizada inteira
    if len(components) == 0:
        # fallback: use mask_smooth como única componente
        coords_all = np.where(mask_smooth)
        if coords_all[0].size == 0:
            raise RuntimeError("Máscara vazia após pré-processamento.")
        # construir componente único
        components.append({
            "label": 1,
            "count": int(coords_all[0].size),
            "centroid_x": float(coords_all[2].mean()),
            "coords": coords_all
        })

    # Calcular centro geométrico X do volume (para forçar esquerda/direita)
    coords_all = np.where(mask)
    min_x, max_x = int(coords_all[2].min()), int(coords_all[2].max())
    mid_x = (min_x + max_x) / 2.0  # ponto médio (float)

    # Criar máscaras vazias (boolean)
    mask_left = np.zeros_like(mask, dtype=bool)
    mask_right = np.zeros_like(mask, dtype=bool)

    # Atribuir cada componente ao lado esquerdo ou direito com base no centroid_x vs mid_x
    for comp in components:
        if comp["centroid_x"] < mid_x:
            mask_left[comp["coords"]] = True
        else:
            mask_right[comp["coords"]] = True

    # Se algum dos lados ficou vazio, fazer split por índice X (garante sempre L e R)
    if not mask_left.any() or not mask_right.any():
        # dividir por índice inteiro (metade entre min_x e max_x)
        mid_idx = int((min_x + max_x) // 2)
        mask_left[:, :, :mid_idx+1] = mask[:, :, :mid_idx+1]
        mask_right[:, :, mid_idx+1:] = mask[:, :, mid_idx+1:]

    # Pós-processamento morfológico para limpar e preencher buracos
    for _ in range(MORPH_ITER):
        mask_left = binary_closing(mask_left, structure=structure)
        mask_left = binary_opening(mask_left, structure=structure)
        mask_right = binary_closing(mask_right, structure=structure)
        mask_right = binary_opening(mask_right, structure=structure)

    # Garantir que não haja sobreposição; se houver, atribuir pelo X mais próximo do centro do respectivo lado
    overlap = mask_left & mask_right
    if overlap.any():
        def centroid_x_of(mask_bool):
            c = np.where(mask_bool)
            if c[0].size == 0:
                return None
            return c[2].mean()
        cx_left = centroid_x_of(mask_left)
        cx_right = centroid_x_of(mask_right)
        ov_coords = np.where(overlap)
        for z, y, x in zip(*ov_coords):
            if cx_left is None:
                mask_right[z, y, x] = True
                mask_left[z, y, x] = False
                continue
            if cx_right is None:
                mask_left[z, y, x] = True
                mask_right[z, y, x] = False
                continue
            if abs(x - cx_left) <= abs(x - cx_right):
                mask_right[z, y, x] = False
            else:
                mask_left[z, y, x] = False

    # Suavização final e threshold (mantendo boolean)
    mask_left = gaussian_filter(mask_left.astype(float), sigma=FINAL_SMOOTH_SIGMA) > 0.5
    mask_right = gaussian_filter(mask_right.astype(float), sigma=FINAL_SMOOTH_SIGMA) > 0.5

    # Garantir dtype boolean antes de adicionar ao RTStruct (requisito do rt_utils)
    mask_left = mask_left.astype(bool)
    mask_right = mask_right.astype(bool)

    # Debug prints
    print("Voxels originais na máscara:", int(mask.sum()))
    print("Voxels esquerda:", int(mask_left.sum()), "Voxels direita:", int(mask_right.sum()))
    print("mid_x (float):", mid_x, "min_x:", min_x, "max_x:", max_x)

    # Adicionar novas estruturas ao RTSTRUCT (dtype bool exigido pelo rt_utils)
    # Lembrando que o nome "Breast_right" é para a máscara do lado esquerdo da imagem (que corresponde à mama direita do paciente) e vice-versa
    rtstruct.add_roi(mask=mask_left, color="#4C00FF", name="Breast_right")
    rtstruct.add_roi(mask=mask_right, color="#CC00FF", name="Breast_left")

    # Salvar novo RTSTRUCT no mesmo diretório do arquivo original
    output_dir = os.path.dirname(rt_struct_path)
    base = os.path.basename(rt_struct_path)
    name, ext = os.path.splitext(base)
    output_filename = f"{name}_separado{ext}"
    output_path = os.path.join(output_dir, output_filename)
    rtstruct.save(output_path)

    print("RTSTRUCT salvo em:", output_path)

if __name__ == "__main__":
    main()