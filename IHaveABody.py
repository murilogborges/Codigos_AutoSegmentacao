import os
import sys
import pydicom
import numpy as np
from pydicom.dataset import Dataset
from skimage import measure, draw
import scipy.ndimage as ndimage

def hex_to_rgb(hex_color: str):
    """Converte string hex (#RRGGBB) para lista [R,G,B]."""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def gerar_corpo(input_rtstruct, hex_color="#55ffff"):
    ds = pydicom.dcmread(input_rtstruct)

    # Agrupa contornos por fatia Z
    contours_by_slice = {}
    for roi_contour in ds.ROIContourSequence:
        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData).reshape(-1, 3)
            z = round(np.mean(pts[:,2]), 2)
            if z not in contours_by_slice:
                contours_by_slice[z] = []
            contours_by_slice[z].append(pts)

    # Limites espaciais
    all_points = np.vstack([pts for sl in contours_by_slice.values() for pts in sl])
    min_x, min_y, min_z = np.min(all_points, axis=0)
    max_x, max_y, max_z = np.max(all_points, axis=0)

    grid_size = 256
    zs = sorted(contours_by_slice.keys())
    mask = np.zeros((grid_size, grid_size, len(zs)), dtype=np.uint8)

    # Rasteriza cada contorno em sua fatia
    for iz, z in enumerate(zs):
        for pts in contours_by_slice[z]:
            rr = ((pts[:,1]-min_y)/(max_y-min_y)*(grid_size-1)).astype(int)
            cc = ((pts[:,0]-min_x)/(max_x-min_x)*(grid_size-1)).astype(int)
            rr, cc = draw.polygon(rr, cc, mask.shape[:2])
            mask[rr, cc, iz] = 1

    # Suavização e interpolação volumétrica
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.zoom(mask, (1,1,2), order=1)  # aumenta resolução no eixo Z

    # Suavização extra com filtro gaussiano
    mask = ndimage.gaussian_filter(mask.astype(float), sigma=1.5) > 0.4
    mask = mask.astype(np.uint8)

    # Propaga contornos para fatias vazias (primeira e última)
    for iz in range(mask.shape[2]):
        if np.sum(mask[:,:,iz]) == 0:
            if iz > 0:
                mask[:,:,iz] = mask[:,:,iz-1]
            elif iz < mask.shape[2]-1:
                mask[:,:,iz] = mask[:,:,iz+1]

    # Mantém apenas o maior volume conectado
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features+1))
        largest_label = (np.argmax(sizes) + 1)
        mask = (labeled == largest_label).astype(np.uint8)

    # Limpa sequências originais
    ds.StructureSetROISequence.clear()
    ds.ROIContourSequence.clear()

    # Converte cor hex para RGB
    rgb_color = hex_to_rgb(hex_color)

    # Cria novo ROI "corpo"
    new_roi = Dataset()
    new_roi.ROINumber = 15
    new_roi.ROIName = "corpo"
    new_roi.ROIGenerationAlgorithm = "AUTOMATIC"
    new_roi.ROIDisplayColor = rgb_color
    new_roi.RTROIInterpretedType = "EXTERNAL"
    ds.StructureSetROISequence.append(new_roi)

    # Cria ROIContourSequence para o "corpo"
    new_roi_contour = Dataset()
    new_roi_contour.ReferencedROINumber = new_roi.ROINumber
    new_roi_contour.ContourSequence = []

    # Reconverte máscara suavizada em contornos slice a slice
    for iz, z in enumerate(np.linspace(min_z, max_z, mask.shape[2])):
        slice_mask = mask[:,:,iz]
        contours = measure.find_contours(slice_mask, 0.5)
        for c in contours:
            xs = c[:,1]/mask.shape[1]*(max_x-min_x)+min_x
            ys = c[:,0]/mask.shape[0]*(max_y-min_y)+min_y
            zs = np.full_like(xs, z)
            coords = np.vstack([xs, ys, zs]).T.flatten().tolist()

            new_contour = Dataset()
            new_contour.ContourGeometricType = "CLOSED_PLANAR"
            new_contour.NumberOfContourPoints = len(xs)
            new_contour.ContourData = coords
            new_roi_contour.ContourSequence.append(new_contour)

    ds.ROIContourSequence.append(new_roi_contour)

    # Salva novo RTSTRUCT
    output_path = os.path.join(os.path.dirname(input_rtstruct), "CORPO.dcm")
    ds.save_as(output_path)
    print(f"Novo RTSTRUCT salvo em {output_path} com cor {hex_color} (RGB {rgb_color})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python IHaveABody.py <INPUT_RTSTRUCT.dcm> [#RRGGBB]")
        sys.exit(1)

    input_rtstruct = sys.argv[1]
    hex_color = sys.argv[2] if len(sys.argv) > 2 else "#55ffff"
    gerar_corpo(input_rtstruct, hex_color)
