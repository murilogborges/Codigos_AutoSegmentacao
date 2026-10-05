#!/usr/bin/env python3
"""Unifica todas as estruturas de um RTSTRUCT em uma ROI chamada ``corpo``."""

import os
import sys
from typing import Iterable

import numpy as np
import cv2
from scipy import ndimage
from rt_utils import RTStruct, RTStructBuilder, image_helper


MARGIN_MM = 2.0
SMOOTH_SIGMA_MM = 0.8


def hex_to_rgb(hex_color: str) -> list[int]:
    """Converte uma cor hexadecimal em [R, G, B]."""
    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError("A cor deve estar no formato #RRGGBB.")
    try:
        channels = [int(value[index:index + 2], 16) for index in (0, 2, 4)]
    except ValueError as error:
        raise ValueError("A cor deve estar no formato #RRGGBB.") from error
    return channels


def _slice_spacing_mm(series_data: list) -> float:
    """Obtém a distância física entre cortes ordenados."""
    if not series_data:
        raise ValueError("A série DICOM não contém imagens.")
    if len(series_data) < 2:
        return float(getattr(series_data[0], "SliceThickness", 1.0))

    positions = np.asarray(
        [float(np.dot(
            np.cross(
                np.asarray(image.ImageOrientationPatient[:3], dtype=float),
                np.asarray(image.ImageOrientationPatient[3:], dtype=float),
            ),
            np.asarray(image.ImagePositionPatient, dtype=float),
        )) for image in series_data]
    )
    differences = np.abs(np.diff(positions))
    non_zero = differences[differences > 1e-4]
    if non_zero.size == 0:
        return float(getattr(series_data[0], "SliceThickness", 1.0))
    return float(np.median(non_zero))


def _voxel_spacing_mm(series_data: list) -> tuple[float, float, float]:
    if not series_data:
        raise ValueError("A série DICOM não contém imagens.")
    pixel_spacing = np.asarray(series_data[0].PixelSpacing, dtype=float)
    if pixel_spacing.shape != (2,) or np.any(pixel_spacing <= 0):
        raise ValueError("PixelSpacing inválido na série DICOM.")
    slice_spacing = _slice_spacing_mm(series_data)
    if slice_spacing <= 0:
        raise ValueError("Não foi possível determinar o espaçamento entre cortes.")
    return float(pixel_spacing[0]), float(pixel_spacing[1]), slice_spacing


def _largest_component(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    labels, number = ndimage.label(mask, structure=structure)
    if number == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (int(np.argmax(sizes)) + 1)


def _dilate_mm(mask: np.ndarray, spacing: tuple[float, float, float],
               distance_mm: float) -> np.ndarray:
    distances = ndimage.distance_transform_edt(~mask, sampling=spacing)
    return distances <= distance_mm


def _erode_mm(mask: np.ndarray, spacing: tuple[float, float, float],
              distance_mm: float) -> np.ndarray:
    distances = ndimage.distance_transform_edt(mask, sampling=spacing)
    return distances > distance_mm


def _iter_roi_names(rtstruct) -> Iterable[str]:
    seen = set()
    for name in rtstruct.get_roi_names():
        if name not in seen:
            seen.add(name)
            yield name


def _get_roi_number(rtstruct, name: str):
    for roi in rtstruct.ds.StructureSetROISequence:
        if roi.ROIName == name:
            return roi.ROINumber
    raise KeyError(f"ROI '{name}' não encontrada na sequência de estruturas.")


def _valid_contour(contour) -> bool:
    if getattr(contour, "ContourGeometricType", "") not in {
        "CLOSED_PLANAR", "CLOSEDPLANAR_XOR"
    }:
        return False
    try:
        data = np.asarray(getattr(contour, "ContourData", []), dtype=float)
        point_count = int(
            getattr(contour, "NumberOfContourPoints", data.size // 3)
        )
    except (TypeError, ValueError):
        return False
    if data.size < 9 or data.size % 3 != 0 or not np.isfinite(data).all():
        return False
    if point_count < 3 or point_count != data.size // 3:
        return False
    if not hasattr(contour, "ContourImageSequence") or not contour.ContourImageSequence:
        return False
    points = data.reshape(-1, 3)
    return np.unique(points, axis=0).shape[0] >= 3


def _mask_for_roi(rtstruct, name: str) -> np.ndarray:
    roi_number = _get_roi_number(rtstruct, name)
    contour_sequence = [
        contour
        for roi_contour in getattr(rtstruct.ds, "ROIContourSequence", [])
        if str(getattr(roi_contour, "ReferencedROINumber", "")) == str(roi_number)
        for contour in getattr(roi_contour, "ContourSequence", [])
    ]

    closed_contours = [
        contour for contour in contour_sequence
        if getattr(contour, "ContourGeometricType", "") in {
            "CLOSED_PLANAR", "CLOSEDPLANAR_XOR"
        }
    ]
    valid_contours = [contour for contour in closed_contours if _valid_contour(contour)]
    invalid_count = len(closed_contours) - len(valid_contours)
    if invalid_count:
        print(
            f"Aviso: {invalid_count} contorno(s) inválido(s) ignorado(s) "
            f"na ROI '{name}'.",
            file=sys.stderr,
        )
    if not valid_contours:
        raise ValueError("não há contornos fechados válidos")

    geometric_types = {
        getattr(contour, "ContourGeometricType", "")
        for contour in valid_contours
    }
    if len(geometric_types) > 1:
        raise ValueError("a ROI mistura contornos CLOSED_PLANAR e CLOSEDPLANAR_XOR")
    xor_contours = geometric_types == {"CLOSEDPLANAR_XOR"}
    combined = None
    failed = []
    for contour in valid_contours:
        try:
            contour_mask = image_helper.create_series_mask_from_contour_sequence(
                rtstruct.series_data, [contour]
            )
        except (cv2.error, ValueError, IndexError, KeyError, AttributeError,
                TypeError) as error:
            failed.append(str(error))
            continue
        if combined is not None and contour_mask.shape != combined.shape:
            raise ValueError("contornos da ROI produziram máscaras com dimensões diferentes")
        if combined is None:
            combined = contour_mask
        elif xor_contours:
            combined = np.logical_xor(combined, contour_mask)
        else:
            combined = np.logical_or(combined, contour_mask)

    if failed:
        print(
            f"Aviso: {len(failed)} contorno(s) não puderam ser rasterizados "
            f"na ROI '{name}' e foram ignorados: " + "; ".join(failed),
            file=sys.stderr,
        )
    if combined is None or not combined.any():
        raise ValueError("todos os contornos falharam na rasterização")
    return np.asarray(combined, dtype=bool)


def _build_body_mask(rtstruct, spacing: tuple[float, float, float]) -> np.ndarray:
    masks = []
    for name in _iter_roi_names(rtstruct):
        roi_number = _get_roi_number(rtstruct, name)
        has_contours = any(
            str(getattr(roi_contour, "ReferencedROINumber", "")) == str(roi_number)
            and any(
                getattr(contour, "ContourGeometricType", "") in {
                    "CLOSED_PLANAR", "CLOSEDPLANAR_XOR"
                }
                for contour in getattr(roi_contour, "ContourSequence", [])
            )
            for roi_contour in getattr(rtstruct.ds, "ROIContourSequence", [])
        )
        if not has_contours:
            continue
        try:
            roi_mask = _mask_for_roi(rtstruct, name)
        except (RTStruct.ROIException, ValueError, IndexError, KeyError,
                AttributeError, TypeError) as error:
            print(
                f"Aviso: não foi possível rasterizar a ROI '{name}'; "
                f"ela será ignorada: {error}",
                file=sys.stderr,
            )
            continue
        if roi_mask.any():
            masks.append(roi_mask)

    if not masks:
        raise RuntimeError("Nenhuma ROI com contornos rasterizáveis foi encontrada.")

    merged = np.logical_or.reduce(masks)
    merged = ndimage.binary_fill_holes(merged)
    expanded = _dilate_mm(merged, spacing, MARGIN_MM)
    expanded = ndimage.binary_fill_holes(expanded)
    expanded = _largest_component(expanded)

    body = _erode_mm(expanded, spacing, MARGIN_MM)
    body = ndimage.binary_fill_holes(body)
    body = _largest_component(body)
    if not body.any():
        raise RuntimeError(
            "A erosão de 2 mm eliminou a estrutura; verifique a série e os contornos."
        )

    smooth_sigma = tuple(SMOOTH_SIGMA_MM / value for value in spacing)
    smoothed = ndimage.gaussian_filter(body.astype(float), sigma=smooth_sigma)
    body = smoothed >= 0.5
    body = _largest_component(body)
    if not body.any():
        raise RuntimeError("A suavização produziu uma máscara vazia.")
    return body.astype(bool)


def gerar_corpo(input_rtstruct: str, dicom_series_path: str,
                hex_color: str = "#55ffff") -> str:
    """Gera ``CORPO.dcm`` usando a geometria física da série DICOM."""
    color = hex_to_rgb(hex_color)
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path,
        rt_struct_path=input_rtstruct,
    )
    spacing = _voxel_spacing_mm(rtstruct.series_data)
    print(
        "Espaçamento dos voxels (mm): "
        f"linhas={spacing[0]:.4g}, colunas={spacing[1]:.4g}, cortes={spacing[2]:.4g}"
    )

    body_mask = _build_body_mask(rtstruct, spacing)
    print(f"Voxels finais na máscara corpo: {int(body_mask.sum())}")

    # O objeto carregado preserva os metadados e as referências do RTSTRUCT original.
    rtstruct.ds.StructureSetROISequence.clear()
    rtstruct.ds.ROIContourSequence.clear()
    rtstruct.ds.RTROIObservationsSequence.clear()
    rtstruct.add_roi(
        mask=body_mask,
        color=color,
        name="corpo",
        description="Unificação das estruturas com margem física de 2 mm",
        approximate_contours=True,
        roi_generation_algorithm="AUTOMATIC",
    )

    output_path = os.path.join(os.path.dirname(input_rtstruct), "CORPO.dcm")
    rtstruct.save(output_path)
    print(f"Novo RTSTRUCT salvo em {output_path}")
    return output_path


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(
            "Uso: python IHaveABody.py "
            "<INPUT_RTSTRUCT.dcm> <DICOM_SERIES_DIR> [#RRGGBB]"
        )
        raise SystemExit(1)

    input_rtstruct = sys.argv[1]
    dicom_series_path = sys.argv[2]
    hex_color = sys.argv[3] if len(sys.argv) == 4 else "#55ffff"
    gerar_corpo(input_rtstruct, dicom_series_path, hex_color)


if __name__ == "__main__":
    main()
