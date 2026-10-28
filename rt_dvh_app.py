"""Desktop viewer for comparing RTSTRUCT contours and dose-volume histograms."""

from __future__ import annotations

import copy
import csv
import logging
import os
import queue
import struct
import threading
import tkinter as tk
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pydicom
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.filereader import read_dataset
from pydicom.dataset import FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RTDoseStorage,
    RTStructureSetStorage,
)
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

_VR_MISMATCH_WARNINGS = (
    "Expected explicit VR, but found implicit VR",
    "Expected implicit VR, but found explicit VR",
)
_VR_REPAIR_LOGGED = False
_RTSTRUCT_REFERENCE_WARNED: set[Path] = set()
_DOSE_FRAME_MISMATCH_WARNED: set[tuple[str, str]] = set()
_VALID_VRS = {
    b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FD", b"FL", b"IS",
    b"LO", b"LT", b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"PN", b"SH",
    b"SL", b"SQ", b"SS", b"ST", b"SV", b"TM", b"UC", b"UI", b"UL", b"UN",
    b"UR", b"US", b"UT", b"UV",
}
_LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN"}


def _dataset_start_and_transfer_syntax(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        initial = stream.read(132)
        offset = 132 if len(initial) >= 132 and initial[128:132] == b"DICM" else 0
        stream.seek(offset)
        while True:
            element_start = stream.tell()
            header = stream.read(8)
            if len(header) < 8:
                raise ValueError(f"{path} não contém um cabeçalho DICOM válido.")
            group, _element = struct.unpack("<HH", header[:4])
            if group != 0x0002:
                stream.seek(element_start)
                break
            vr = header[4:6]
            if vr not in _VALID_VRS:
                raise ValueError(f"{path} possui metadados DICOM inválidos.")
            if vr in _LONG_VRS:
                length_data = stream.read(4)
                if len(length_data) != 4:
                    raise ValueError(f"{path} possui metadados DICOM truncados.")
                length = struct.unpack("<I", length_data)[0]
            else:
                length = struct.unpack("<H", header[6:8])[0]
            if length == 0xFFFFFFFF:
                raise ValueError(f"{path} possui um elemento de metadados com tamanho indefinido.")
            stream.seek(length, 1)

        dataset_offset = stream.tell()
        stream.seek(dataset_offset)
        header = stream.read(8)
    if len(header) < 8:
        raise ValueError(f"{path} não contém elementos de dataset DICOM.")

    little_group = struct.unpack("<H", header[:2])[0]
    big_group = struct.unpack(">H", header[:2])[0]
    common_groups = {0x0008, 0x0010, 0x0018, 0x0020, 0x0028, 0x3004, 0x7FE0}
    if little_group in common_groups:
        little_endian = True
    elif big_group in common_groups:
        little_endian = False
    else:
        little_endian = True
    is_explicit = header[4:6] in _VALID_VRS
    if is_explicit and not little_endian:
        return dataset_offset, str(ExplicitVRBigEndian)
    if is_explicit:
        return dataset_offset, str(ExplicitVRLittleEndian)
    return dataset_offset, str(ImplicitVRLittleEndian)


def _read_dataset_without_transfer_syntax(path: Path, kwargs: dict):
    offset, transfer_syntax = _dataset_start_and_transfer_syntax(path)
    is_little_endian = transfer_syntax != str(ExplicitVRBigEndian)
    is_implicit_vr = transfer_syntax == str(ImplicitVRLittleEndian)
    stop_before_pixels = bool(kwargs.get("stop_before_pixels", False))
    with path.open("rb") as stream:
        stream.seek(offset)
        dataset = read_dataset(
            stream,
            is_implicit_VR=is_implicit_vr,
            is_little_endian=is_little_endian,
            defer_size=kwargs.get("defer_size"),
            specific_tags=kwargs.get("specific_tags"),
            stop_when=(
                (lambda tag, _vr, _length: tag == 0x7FE00010)
                if stop_before_pixels
                else None
            ),
        )
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = transfer_syntax
    return dataset


def read_dicom(path: str | Path, **kwargs):
    """Read standard DICOM files and retry without a preamble when necessary."""
    dicom_path = Path(path)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            dataset = pydicom.dcmread(dicom_path, **kwargs)
        except InvalidDicomError:
            retry_kwargs = dict(kwargs)
            retry_kwargs["force"] = True
            dataset = pydicom.dcmread(dicom_path, **retry_kwargs)

    vr_mismatch = False
    for warning in captured:
        if any(message in str(warning.message) for message in _VR_MISMATCH_WARNINGS):
            vr_mismatch = True
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)
    if vr_mismatch:
        _offset, actual_transfer_syntax = _dataset_start_and_transfer_syntax(dicom_path)
        if getattr(dataset, "file_meta", None) is None:
            raise ValueError(f"{dicom_path} não possui metadados DICOM para validar a codificação.")
        dataset.file_meta.TransferSyntaxUID = actual_transfer_syntax
        global _VR_REPAIR_LOGGED
        if not _VR_REPAIR_LOGGED:
            logging.getLogger(__name__).info(
                "Discrepância de VR detectada; a codificação real foi inferida do arquivo. "
                "Avisos adicionais equivalentes serão agrupados."
            )
            _VR_REPAIR_LOGGED = True
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if not transfer_syntax:
        dataset = _read_dataset_without_transfer_syntax(dicom_path, kwargs)
    return dataset


@dataclass(frozen=True)
class DicomFile:
    path: Path
    series_uid: str
    description: str
    patient_name: str = ""
    instance_paths: tuple[Path, ...] = ()

    @property
    def label(self) -> str:
        series_id = self.series_uid[-8:] if self.series_uid else "DICOM"
        patient = f" — Paciente: {self.patient_name}" if self.patient_name else ""
        return (
            f"{self.path.name} ({self.path.parent.name}) — "
            f"{self.description or 'DICOM'} [{series_id}]{patient}"
        )


@dataclass
class CTSeries:
    datasets: list
    paths: list[Path]
    row_spacing: float
    column_spacing: float
    slice_spacing: float
    row_direction: np.ndarray
    column_direction: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        first = self.datasets[0]
        return int(first.Rows), int(first.Columns), len(self.datasets)


@dataclass
class DVHCurve:
    name: str
    dose_gy: np.ndarray
    volume_cc: np.ndarray
    volume_percent: np.ndarray


def _sop_class(dataset) -> str:
    return str(getattr(dataset, "SOPClassUID", ""))


def _iter_dicom_candidates(root: Path):
    excluded_directories = {
        ".git", ".venv", "venv", "env", "site-packages", "__pycache__",
        "node_modules", ".cache",
    }
    for current, directories, filenames in os.walk(root):
        directories[:] = [
            name for name in directories if name.casefold() not in excluded_directories
        ]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            suffix = path.suffix.casefold()
            try:
                with path.open("rb") as stream:
                    prefix = stream.read(132)
                if len(prefix) >= 132 and prefix[128:132] == b"DICM":
                    yield path
                    continue
                if suffix not in {".dcm", ".dicom", ".ima", ".dic"} or len(prefix) < 8:
                    continue
                little_group = int.from_bytes(prefix[:2], "little")
                big_group = int.from_bytes(prefix[:2], "big")
                plausible_groups = (little_group, big_group)
                if any(0x0002 <= group <= 0x7FE0 and group % 2 == 0 for group in plausible_groups):
                    yield path
            except OSError:
                continue


def discover_dicom_files(
    folder: str | Path, progress=None
) -> tuple[list[DicomFile], list[DicomFile], list[DicomFile]]:
    """Find CT series, RT dose files, and RT structure sets beneath a folder."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"A pasta não existe: {root}")

    ct_groups: dict[str, list[tuple[Path, str, str]]] = {}
    doses: list[DicomFile] = []
    structures: list[DicomFile] = []
    if progress:
        progress(0, "Localizando arquivos DICOM na pasta")
    candidates = list(_iter_dicom_candidates(root))
    total = max(len(candidates), 1)
    if progress:
        progress(0, f"Verificando cabeçalhos DICOM (0/{len(candidates)})")
    for index, path in enumerate(candidates, 1):
        if progress and (index == total or index % 20 == 0):
            progress(index * 100 / total, f"Verificando DICOM ({index}/{len(candidates)})")
        try:
            dataset = read_dicom(path, stop_before_pixels=True, force=False)
        except (InvalidDicomError, OSError, ValueError):
            continue

        sop = _sop_class(dataset)
        modality = str(getattr(dataset, "Modality", ""))
        description = str(getattr(dataset, "SeriesDescription", ""))
        if sop == str(CTImageStorage) and modality == "CT":
            uid = str(getattr(dataset, "SeriesInstanceUID", ""))
            if uid:
                patient_name = str(getattr(dataset, "PatientName", "")).strip()
                ct_groups.setdefault(uid, []).append((path, description, patient_name))
        elif sop == str(RTDoseStorage) or modality == "RTDOSE":
            doses.append(
                DicomFile(path, str(getattr(dataset, "SeriesInstanceUID", "")), description)
            )
        elif sop == str(RTStructureSetStorage) or modality == "RTSTRUCT":
            structures.append(
                DicomFile(path, str(getattr(dataset, "SeriesInstanceUID", "")), description)
            )

    ct_series = [
        DicomFile(
            paths[0][0],
            uid,
            paths[0][1] or f"CT ({len(paths)} cortes)",
            paths[0][2],
            tuple(item[0] for item in paths),
        )
        for uid, paths in sorted(ct_groups.items())
    ]
    structures.sort(
        key=lambda item: (
            0 if item.path.name.upper().startswith("TOTAL_") else 1,
            item.path.name.casefold(),
        )
    )
    return ct_series, sorted(doses, key=lambda item: item.path.name.casefold()), structures


def load_ct_series(
    folder: str | Path,
    series_uid: str,
    instance_paths: tuple[Path, ...] | list[Path] | None = None,
    progress=None,
) -> CTSeries:
    """Load one CT image series in physical slice order."""
    paths: list[Path] = []
    datasets = []
    indexed_series = instance_paths is not None
    paths_to_check = list(instance_paths) if indexed_series else list(
        _iter_dicom_candidates(Path(folder))
    )
    total = max(len(paths_to_check), 1)
    if progress:
        progress(0, f"Lendo imagens da CT (0/{len(paths_to_check)})")
    for index, path in enumerate(paths_to_check, 1):
        if indexed_series:
            try:
                datasets.append(read_dicom(path))
                paths.append(path)
            except (InvalidDicomError, OSError, ValueError):
                continue
            if progress and (index == total or index % 10 == 0):
                progress(
                    index * 100 / total,
                    f"Lendo imagens da CT ({index}/{len(paths_to_check)})",
                )
            continue
        try:
            header = read_dicom(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError):
            continue
        if (
            str(getattr(header, "SeriesInstanceUID", "")) == series_uid
            and _sop_class(header) == str(CTImageStorage)
        ):
            datasets.append(read_dicom(path))
            paths.append(path)
        if progress and (index == total or index % 10 == 0):
            progress(index * 100 / total, f"Lendo imagens da CT ({index}/{len(paths_to_check)})")

    if not datasets:
        raise ValueError("A série CT selecionada não contém imagens legíveis.")
    orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    datasets_paths = sorted(
        zip(datasets, paths),
        key=lambda pair: float(
            np.dot(normal, np.asarray(pair[0].ImagePositionPatient, dtype=float))
        ),
    )
    datasets = [pair[0] for pair in datasets_paths]
    paths = [pair[1] for pair in datasets_paths]

    first = datasets[0]
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    rows, columns = int(first.Rows), int(first.Columns)
    first_position = np.asarray(first.ImagePositionPatient, dtype=float)
    projections = []
    for dataset in datasets:
        if (int(dataset.Rows), int(dataset.Columns)) != (rows, columns):
            raise ValueError("A série CT contém imagens com dimensões inconsistentes.")
        if not np.allclose(dataset.ImageOrientationPatient, orientation, atol=1e-4):
            raise ValueError("A série CT contém orientações diferentes.")
        if not np.allclose(dataset.PixelSpacing, pixel_spacing, atol=1e-4):
            raise ValueError("A série CT contém espaçamentos de pixel diferentes.")
        projections.append(float(np.dot(normal, np.asarray(dataset.ImagePositionPatient, dtype=float))))
    differences = np.diff(projections)
    if len(differences):
        slice_spacing = float(np.median(differences))
        if slice_spacing <= 0 or not np.allclose(
            differences, slice_spacing, rtol=0.05, atol=0.05
        ):
            raise ValueError(
                "A série CT tem espaçamento entre cortes irregular; "
                "não é possível calcular volumes corretamente."
            )
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
    if not np.all(np.isfinite(first_position)):
        raise ValueError("A série CT não possui geometria espacial válida.")

    return CTSeries(
        datasets=datasets,
        paths=paths,
        row_spacing=float(pixel_spacing[0]),
        column_spacing=float(pixel_spacing[1]),
        slice_spacing=slice_spacing,
        row_direction=row_direction,
        column_direction=column_direction,
    )


def read_roi_names(rtstruct_path: str | Path, progress=None) -> list[str]:
    if progress:
        progress(5, f"Lendo cabeçalho RTSTRUCT: {Path(rtstruct_path).name}")
    dataset = read_dicom(rtstruct_path, stop_before_pixels=True)
    if (
        _sop_class(dataset) != str(RTStructureSetStorage)
        and str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT"
    ):
        raise ValueError(f"O arquivo não é um RTSTRUCT: {rtstruct_path}")
    names = [
        str(getattr(roi, "ROIName", "")).strip()
        for roi in getattr(dataset, "StructureSetROISequence", [])
    ]
    names = filter_selectable_roi_names([name for name in names if name])
    if progress:
        progress(100, f"{len(names)} estrutura(s) lida(s) de {Path(rtstruct_path).name}")
    return names


def load_structure_masks(
    ct_series: CTSeries,
    rtstruct_path: str | Path,
    roi_names: list[str],
    allow_empty: bool = False,
    progress=None,
) -> dict[str, np.ndarray]:
    """Rasterize selected RTSTRUCT ROIs on the selected CT grid with rt-utils."""
    from rt_utils.rtstruct import RTStruct

    if progress:
        progress(0, f"Lendo arquivo RTSTRUCT: {Path(rtstruct_path).name}")
    dataset = read_dicom(rtstruct_path)
    if (
        _sop_class(dataset) != str(RTStructureSetStorage)
        and str(getattr(dataset, "Modality", "")).upper() != "RTSTRUCT"
    ):
        raise ValueError(f"O arquivo não é um RTSTRUCT: {rtstruct_path}")
    ct_frame_uid = str(getattr(ct_series.datasets[0], "FrameOfReferenceUID", ""))
    rt_frame_uids = {
        str(frame.FrameOfReferenceUID)
        for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", [])
        if getattr(frame, "FrameOfReferenceUID", "")
    }
    rtstruct_path = Path(rtstruct_path)
    if (
        (not rt_frame_uids or ct_frame_uid not in rt_frame_uids)
        and rtstruct_path not in _RTSTRUCT_REFERENCE_WARNED
    ):
        logging.getLogger(__name__).warning(
            "O RTSTRUCT %s não declara o mesmo Frame of Reference do CT. "
            "A leitura continuará e os contornos serão associados às imagens "
            "pelas coordenadas físicas; confirme que estão no mesmo espaço.",
            rtstruct_path,
        )
        _RTSTRUCT_REFERENCE_WARNED.add(rtstruct_path)
    # rt-utils requires a reference frame even though mask rasterization uses contour
    # coordinates. Supply a temporary frame when metadata is absent; never rewrite input.
    raster_dataset = copy.deepcopy(dataset)
    frames = getattr(raster_dataset, "ReferencedFrameOfReferenceSequence", None)
    if not frames:
        raster_dataset.ReferencedFrameOfReferenceSequence = Sequence([Dataset()])
        frames = raster_dataset.ReferencedFrameOfReferenceSequence
    if not getattr(frames[-1], "FrameOfReferenceUID", ""):
        frames[-1].FrameOfReferenceUID = str(
            getattr(ct_series.datasets[0], "FrameOfReferenceUID", "")
        )
    empty_rois = _sanitize_contours_for_rasterization(
        raster_dataset, set(roi_names), ct_series
    )
    empty_rois.update(
        _repair_contour_image_references(raster_dataset, ct_series, set(roi_names))
    )
    rtstruct = RTStruct(ct_series.datasets, raster_dataset)
    available = set(rtstruct.get_roi_names())
    masks: dict[str, np.ndarray] = {}
    expected_shape = ct_series.shape
    if progress:
        progress(10, f"Rasterizando {len(roi_names)} ROI(s) de {Path(rtstruct_path).name}")
    for index, name in enumerate(roi_names, 1):
        if name not in available:
            raise ValueError(f"A estrutura '{name}' não foi encontrada no RTSTRUCT.")
        if name in empty_rois:
            if not allow_empty:
                raise ValueError(f"A estrutura '{name}' não possui contornos rasterizáveis.")
            masks[name] = np.zeros(expected_shape, dtype=bool)
            continue
        # rt-utils returns (columns, rows, slices); use the CT (rows, columns, slices) convention.
        mask = np.asarray(rtstruct.get_roi_mask_by_name(name), dtype=bool).transpose(1, 0, 2)
        if mask.shape != expected_shape:
            raise ValueError(
                f"A máscara '{name}' tem dimensão inesperada {mask.shape}; CT: {expected_shape}."
            )
        if not np.any(mask) and not allow_empty:
            raise ValueError(f"A estrutura '{name}' não possui contornos rasterizáveis.")
        masks[name] = mask
        if progress:
            progress(
                10 + index * 90 / max(1, len(roi_names)),
                f"Rasterizando ROI {index}/{len(roi_names)}: {name}",
            )
    return masks


def _repair_contour_image_references(
    dataset, ct_series: CTSeries, roi_names: set[str]
) -> set[str]:
    """Map contours to CT slices by physical coordinates, not possibly broken UIDs."""
    first = ct_series.datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    slice_projections = np.asarray(
        [
            np.dot(normal, np.asarray(item.ImagePositionPatient, dtype=float))
            for item in ct_series.datasets
        ]
    )
    tolerance = max(0.5, ct_series.slice_spacing * 0.51)
    if len(slice_projections) == 1:
        tolerance = max(
            tolerance,
            float(getattr(first, "SliceThickness", ct_series.slice_spacing)) * 0.51,
        )

    roi_numbers = {
        int(roi.ROINumber): str(getattr(roi, "ROIName", ""))
        for roi in getattr(dataset, "StructureSetROISequence", [])
    }
    valid_rois = set()
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        roi_name = roi_numbers.get(int(roi_contour.ReferencedROINumber), "")
        if roi_name not in roi_names:
            continue
        valid_contours = []
        for contour in getattr(roi_contour, "ContourSequence", []):
            contour_points = np.asarray(
                getattr(contour, "ContourData", []), dtype=float
            )
            if contour_points.size < 3 or contour_points.size % 3:
                continue
            projection = float(
                np.mean(contour_points.reshape(-1, 3) @ normal)
            )
            distances = np.abs(slice_projections - projection)
            slice_index = int(np.argmin(distances))
            if distances[slice_index] <= tolerance:
                image_reference = Dataset()
                image_reference.ReferencedSOPClassUID = ct_series.datasets[
                    slice_index
                ].SOPClassUID
                image_reference.ReferencedSOPInstanceUID = ct_series.datasets[
                    slice_index
                ].SOPInstanceUID
                contour.ContourImageSequence = Sequence([image_reference])
                valid_contours.append(contour)
            else:
                logging.getLogger(__name__).warning(
                    "Contorno da ROI '%s' fora da série CT selecionada; ignorado.",
                    roi_name,
                )
        roi_contour.ContourSequence = Sequence(valid_contours)
        if valid_contours:
            valid_rois.add(roi_name)
    return roi_names - valid_rois


def _sanitize_contours_for_rasterization(
    dataset, roi_names: set[str], ct_series: CTSeries
) -> set[str]:
    """Remove malformed or degenerate polygons before rt-utils calls OpenCV fillPoly."""
    from rt_utils import image_helper

    roi_numbers = {
        int(roi.ROINumber): str(getattr(roi, "ROIName", ""))
        for roi in getattr(dataset, "StructureSetROISequence", [])
    }
    transformation = image_helper.get_patient_to_pixel_transformation_matrix(
        ct_series.datasets
    )
    valid_rois = set()
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        roi_name = roi_numbers.get(int(roi_contour.ReferencedROINumber), "")
        if roi_name not in roi_names:
            continue
        valid_contours = []
        for contour in getattr(roi_contour, "ContourSequence", []):
            try:
                coordinates = np.asarray(contour.ContourData, dtype=float)
                if coordinates.size < 9 or coordinates.size % 3:
                    raise ValueError("contorno precisa de pelo menos três pontos 3D")
                points = coordinates.reshape(-1, 3)
                if not np.all(np.isfinite(points)):
                    raise ValueError("coordenadas não finitas")
                transformed = image_helper.apply_transformation_to_3d_points(
                    points, transformation
                )
                polygon = np.rint(transformed[:, :2])
                if (
                    not np.all(np.isfinite(polygon))
                    or np.any(np.abs(polygon) > np.iinfo(np.int32).max)
                    or len(np.unique(polygon, axis=0)) < 3
                ):
                    raise ValueError("polígono vazio ou degenerado após projeção na CT")
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                logging.getLogger(__name__).warning(
                    "Contorno inválido ignorado na ROI '%s': %s", roi_name, error
                )
                continue
            valid_contours.append(contour)
        roi_contour.ContourSequence = Sequence(valid_contours)
        if valid_contours:
            valid_rois.add(roi_name)
    return roi_names - valid_rois


def choose_default_rtstructs(files: list[DicomFile]) -> tuple[int, int]:
    auto_index = next(
        (index for index, item in enumerate(files) if item.path.name.upper().startswith("TOTAL_")),
        0,
    )
    ptv_index = next(
        (index for index, item in enumerate(files) if item.path.name.upper().startswith("SS")),
        next((index for index in range(len(files)) if index != auto_index), auto_index),
    )
    return auto_index, ptv_index


def preferred_ptv_indices(roi_names: list[str]) -> list[int]:
    return [index for index, name in enumerate(roi_names) if "ptv" in name.casefold()]


def filter_selectable_roi_names(roi_names: list[str]) -> list[str]:
    return [name for name in roi_names if "isocenter" not in name.casefold()]


def contained_structure_candidates(
    left_masks: dict[str, np.ndarray],
    right_mask: np.ndarray,
    threshold: float = 0.6,
) -> list[tuple[str, float]]:
    if not 0 <= threshold <= 1:
        raise ValueError("O percentual mínimo de inclusão deve estar entre 0 e 1.")
    percentages = structure_containment_percentages(left_masks, right_mask)
    return sorted(
        [
            (name, percentage)
            for name, percentage in percentages.items()
            if percentage >= threshold
        ],
        key=lambda item: (-item[1], item[0].casefold()),
    )


def structure_containment_percentages(
    left_masks: dict[str, np.ndarray], right_mask: np.ndarray
) -> dict[str, float]:
    percentages = {}
    for name, mask in left_masks.items():
        volume = int(np.count_nonzero(mask))
        if volume == 0:
            percentages[name] = 0.0
            continue
        percentages[name] = float(np.count_nonzero(mask & right_mask) / volume)
    return percentages


def structure_comparison_metrics(
    auto_masks: dict[str, np.ndarray],
    ptv_mask: np.ndarray,
    ct_series: CTSeries,
    progress=None,
) -> list[dict[str, float | str]]:
    """Measure auto-ROI inclusion in the PTV and PTV coverage by auto-ROI margins."""
    ptv_mask = np.asarray(ptv_mask, dtype=bool)
    if any(np.asarray(mask).shape != ptv_mask.shape for mask in auto_masks.values()):
        raise ValueError("As máscaras das estruturas precisam ter a mesma grade.")

    ptv_volume = int(np.count_nonzero(ptv_mask))
    if ptv_volume == 0:
        raise ValueError("A união dos PTVs selecionados não contém voxels.")
    rows = []
    for index, name in enumerate(auto_masks):
        auto_mask = np.asarray(auto_masks[name], dtype=bool)
        auto_volume = int(np.count_nonzero(auto_mask))
        if auto_volume == 0:
            within_ptv = ptv_within_margin_05 = ptv_within_margin_10 = dice = 0.0
        else:
            overlap = int(np.count_nonzero(auto_mask & ptv_mask))
            within_ptv = overlap / auto_volume * 100.0
            margin_05 = margin_mask(auto_mask, ct_series, 5.0)
            ptv_within_margin_05 = (
                int(np.count_nonzero(ptv_mask & margin_05)) / ptv_volume * 100.0
            )
            del margin_05
            if progress:
                progress(
                    (index + 0.5) * 100 / max(1, len(auto_masks)),
                    f"Calculando margem de 0,5 cm: {name}",
                )
            margin_10 = margin_mask(auto_mask, ct_series, 10.0)
            ptv_within_margin_10 = (
                int(np.count_nonzero(ptv_mask & margin_10)) / ptv_volume * 100.0
            )
            del margin_10
            dice_denominator = auto_volume + ptv_volume
            dice = 2.0 * overlap / dice_denominator * 100.0 if dice_denominator else 0.0
        rows.append(
            {
                "structure": name,
                "inside_ptv_percent": within_ptv,
                "ptv_inside_auto_margin_05_percent": ptv_within_margin_05,
                "ptv_inside_auto_margin_10_percent": ptv_within_margin_10,
                "dice_percent": dice,
            }
        )
        if progress:
            progress(
                (index + 1) * 100 / max(1, len(auto_masks)),
                f"Métricas concluídas: {name}",
            )
    return rows


def containment_failure_message(
    percentages: dict[str, float], threshold: float = 0.6
) -> str:
    ranked = sorted(
        percentages.items(), key=lambda item: (-item[1], item[0].casefold())
    )
    message = (
        f"Foram verificadas {len(percentages)} estrutura(s) da auto-segmentação. "
        f"Nenhuma alcançou o limite de {threshold:.0%}. "
        "O percentual é a fração do volume de cada estrutura à esquerda que "
        "está dentro da união dos PTVs selecionados."
    )
    if not ranked:
        return message + "\nNenhuma estrutura candidata foi encontrada."
    best_name, best_percentage = ranked[0]
    if best_percentage == 0:
        message += (
            "\nNão houve sobreposição entre as máscaras na CT selecionada. "
            "Verifique se os dois RTSTRUCTs pertencem ao mesmo paciente e "
            "estão no mesmo espaço físico/Frame of Reference."
        )
    else:
        message += f"\nMaior inclusão: {best_name} ({best_percentage:.1%})."
    details = "\n".join(
        f"• {name}: {percentage:.1%}" for name, percentage in ranked[:5]
    )
    return f"{message}\n\nMaiores resultados:\n{details}"


def load_dose_grid(
    dose_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, np.ndarray], str]:
    """Return absolute dose in Gy and its DICOM grid geometry."""
    dataset = read_dicom(dose_path)
    if _sop_class(dataset) != str(RTDoseStorage):
        raise ValueError(f"O arquivo não é um RTDOSE: {dose_path}")
    units = str(getattr(dataset, "DoseUnits", "")).upper()
    if units not in {"GY", "CGY"}:
        raise ValueError(f"Unidade de dose '{units or 'ausente'}' não é absoluta (GY/CGY).")
    pixels = np.asarray(dataset.pixel_array, dtype=np.float64)
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    scale = float(getattr(dataset, "DoseGridScaling", 0))
    if scale <= 0:
        raise ValueError("O RTDOSE não possui DoseGridScaling válido.")
    dose_gy = pixels * scale * (0.01 if units == "CGY" else 1.0)

    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    row_direction, column_direction = orientation[:3], orientation[3:]
    normal = np.cross(row_direction, column_direction)
    position = np.asarray(dataset.ImagePositionPatient, dtype=float)
    pixel_spacing = np.asarray(dataset.PixelSpacing, dtype=float)
    offsets = np.atleast_1d(
        np.asarray(getattr(dataset, "GridFrameOffsetVector", [0.0]), dtype=float)
    )
    if len(offsets) != dose_gy.shape[0]:
        raise ValueError("GridFrameOffsetVector não corresponde às dimensões do RTDOSE.")
    if (
        len(offsets) > 1
        and np.allclose(row_direction, [1, 0, 0], atol=1e-5)
        and np.allclose(column_direction, [0, 1, 0], atol=1e-5)
        and np.isclose(offsets[0], position[2], atol=1e-5)
    ):
        offsets = offsets - float(np.dot(position, normal))
    order = np.argsort(offsets)
    offsets = offsets[order]
    dose_gy = dose_gy[order, :, :]
    if len(offsets) > 1 and np.any(np.diff(offsets) <= 0):
        raise ValueError("O RTDOSE possui posições de quadros inválidas.")
    return (
        dose_gy,
        position,
        row_direction,
        column_direction,
        normal,
        (float(pixel_spacing[0]), float(pixel_spacing[1]), offsets),
        str(getattr(dataset, "FrameOfReferenceUID", "")),
    )


def inspect_dose_file(path: str | Path) -> DicomFile:
    dataset = read_dicom(path, stop_before_pixels=True)
    if _sop_class(dataset) != str(RTDoseStorage) and str(
        getattr(dataset, "Modality", "")
    ).upper() != "RTDOSE":
        raise ValueError(f"O arquivo selecionado não é um RTDOSE DICOM: {path}")
    return DicomFile(
        Path(path),
        str(getattr(dataset, "SeriesInstanceUID", "")),
        str(getattr(dataset, "SeriesDescription", "")),
    )


def sample_dose_on_mask(
    mask: np.ndarray,
    ct_series: CTSeries,
    dose_path: str | Path | None = None,
    dose_grid=None,
) -> tuple[np.ndarray, float]:
    """Interpolate RTDOSE values (Gy) at each included CT voxel center."""
    if dose_grid is None:
        if dose_path is None:
            raise ValueError("Informe o arquivo RTDOSE ou uma grade de dose já carregada.")
        dose_grid = load_dose_grid(dose_path)
    (
        dose,
        dose_origin,
        dose_row,
        dose_column,
        dose_normal,
        geometry,
        dose_frame_uid,
    ) = dose_grid
    row_spacing, column_spacing, offsets = geometry
    ct_frame_uid = str(getattr(ct_series.datasets[0], "FrameOfReferenceUID", ""))
    if not dose_frame_uid or not ct_frame_uid:
        raise ValueError("O RTDOSE e a série CT precisam informar o Frame of Reference.")
    if dose_frame_uid != ct_frame_uid:
        frame_pair = (dose_frame_uid, ct_frame_uid)
        if frame_pair not in _DOSE_FRAME_MISMATCH_WARNED:
            logging.getLogger(__name__).warning(
                "O RTDOSE e a CT têm Frame of Reference diferentes. O DVH será "
                "calculado assumindo que suas coordenadas físicas são diretamente "
                "comparáveis; nenhum registro espacial será aplicado."
            )
            _DOSE_FRAME_MISMATCH_WARNED.add(frame_pair)
    if mask.shape != ct_series.shape:
        raise ValueError("A máscara não coincide com a grade CT selecionada.")
    rows, columns, slices = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Não há voxels na máscara para calcular o DVH.")

    ct_orientation = ct_series.datasets[0].ImageOrientationPatient
    ct_column_direction = np.asarray(ct_orientation[:3], dtype=float)
    ct_row_direction = np.asarray(ct_orientation[3:], dtype=float)
    positions = np.asarray(
        [dataset.ImagePositionPatient for dataset in ct_series.datasets], dtype=float
    )
    patient_points = (
        positions[slices]
        + (columns * ct_series.column_spacing)[:, None] * ct_column_direction
        + (rows * ct_series.row_spacing)[:, None] * ct_row_direction
    )
    delta = patient_points - dose_origin
    dose_columns = delta @ dose_row / column_spacing
    dose_rows = delta @ dose_column / row_spacing
    frame_offsets = delta @ dose_normal
    if len(offsets) == 1:
        dose_frames = np.zeros_like(frame_offsets)
    else:
        dose_frames = np.interp(frame_offsets, offsets, np.arange(len(offsets)))

    coordinates = np.vstack((dose_frames, dose_rows, dose_columns))
    sampled = map_coordinates(dose, coordinates, order=1, mode="constant", cval=np.nan)
    valid = (
        np.isfinite(sampled)
        & (dose_rows >= 0)
        & (dose_rows <= dose.shape[1] - 1)
        & (dose_columns >= 0)
        & (dose_columns <= dose.shape[2] - 1)
    )
    if len(offsets) > 1:
        valid &= (frame_offsets >= offsets[0]) & (frame_offsets <= offsets[-1])
    if not np.any(valid):
        raise ValueError(
            "Nenhum voxel da estrutura está dentro da área coberta pelo RTDOSE."
        )
    sampled = sampled[valid]
    voxel_volume_cc = (
        ct_series.row_spacing * ct_series.column_spacing * ct_series.slice_spacing / 1000.0
    )
    return sampled, voxel_volume_cc


def create_dvh_curve(
    name: str, doses_gy: np.ndarray, voxel_volume_cc: float
) -> DVHCurve:
    doses = np.asarray(doses_gy, dtype=float)
    doses = doses[np.isfinite(doses)]
    if not len(doses) or voxel_volume_cc <= 0:
        raise ValueError("Não há amostras de dose válidas para gerar o DVH.")
    max_dose = max(float(np.max(doses)), 0.0)
    bin_count = max(1, int(np.floor(np.nextafter(max_dose, np.inf) / 0.1)) + 1)
    edges = np.arange(bin_count + 1, dtype=float) * 0.1
    histogram, edges = np.histogram(doses, bins=edges)
    volume_cc = np.cumsum(histogram[::-1])[::-1].astype(float) * voxel_volume_cc
    total_volume = float(len(doses) * voxel_volume_cc)
    percent = volume_cc / total_volume * 100.0
    return DVHCurve(name, edges[:-1], volume_cc, percent)


def margin_mask(mask: np.ndarray, ct_series: CTSeries, margin_mm: float) -> np.ndarray:
    if margin_mm <= 0:
        raise ValueError("A margem deve ser maior que zero.")
    sampling = (
        ct_series.row_spacing,
        ct_series.column_spacing,
        ct_series.slice_spacing,
    )
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    bounds = []
    for axis, spacing in enumerate(sampling):
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        occupied = np.flatnonzero(np.any(mask, axis=other_axes))
        padding = int(np.ceil(margin_mm / spacing)) + 1
        bounds.append(
            (
                max(0, int(occupied[0]) - padding),
                min(mask.shape[axis], int(occupied[-1]) + padding + 1),
            )
        )
    slices = tuple(slice(start, stop) for start, stop in bounds)
    cropped = mask[slices]
    expanded = distance_transform_edt(~cropped, sampling=sampling) <= margin_mm
    result = np.zeros(mask.shape, dtype=bool)
    result[slices] = expanded
    return result


def erode_mask(mask: np.ndarray, ct_series: CTSeries, erosion_mm: float) -> np.ndarray:
    if erosion_mm <= 0:
        raise ValueError("A erosão deve ser maior que zero.")
    sampling = (
        ct_series.row_spacing,
        ct_series.column_spacing,
        ct_series.slice_spacing,
    )
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    bounds = []
    padding = []
    for axis, spacing in enumerate(sampling):
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        occupied = np.flatnonzero(np.any(mask, axis=other_axes))
        pad = int(np.ceil(erosion_mm / spacing)) + 1
        bounds.append(
            (
                max(0, int(occupied[0]) - pad),
                min(mask.shape[axis], int(occupied[-1]) + pad + 1),
            )
        )
        padding.append(pad)
    slices = tuple(slice(start, stop) for start, stop in bounds)
    cropped = mask[slices]
    padded = np.pad(cropped, [(pad, pad) for pad in padding], mode="constant")
    eroded = distance_transform_edt(padded, sampling=sampling) > erosion_mm
    unpadded = tuple(
        slice(pad, pad + size) for pad, size in zip(padding, cropped.shape)
    )
    result = np.zeros(mask.shape, dtype=bool)
    result[slices] = eroded[unpadded]
    return result


def coarsen_mask(mask: np.ndarray, maximum_axis_size: int = 55) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Downsample a display mask by occupied blocks, preserving small ROIs."""
    strides = tuple(max(1, int(np.ceil(size / maximum_axis_size))) for size in mask.shape)
    padding = tuple((0, (-size) % stride) for size, stride in zip(mask.shape, strides))
    padded = np.pad(mask, padding, mode="constant")
    output_shape = tuple(padded.shape[index] // strides[index] for index in range(3))
    blocks = padded.reshape(
        output_shape[0], strides[0],
        output_shape[1], strides[1],
        output_shape[2], strides[2],
    )
    return blocks.any(axis=(1, 3, 5)), strides


def draw_structure_3d(
    left_axes,
    right_axes,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    ct_series: CTSeries | None,
    prepared_surfaces: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None,
    comparison_axes=None,
    prepared_comparison_surfaces: tuple[
        tuple[np.ndarray, np.ndarray] | None, ...
    ] | None = None,
) -> None:
    masks = (left_mask, right_mask)
    if prepared_surfaces is None:
        prepared_surfaces = tuple(
            _structure_surface(mask, ct_series) if mask is not None else None
            for mask in masks
        )
    panels = [
        (left_axes, "Auto-segmentação", [(prepared_surfaces[0], "royalblue")]),
        (right_axes, "PTV", [(prepared_surfaces[1], "orangered")]),
    ]
    if comparison_axes is not None:
        if prepared_comparison_surfaces is None:
            prepared_comparison_surfaces = prepare_comparison_surfaces(
                left_mask, right_mask, ct_series
            )
        comparison_colors = ("#176b3a", "#a8e6a3", "#e79a45")
        panels.append(
            (
                comparison_axes,
                "Sobreposição e diferenças",
                list(zip(prepared_comparison_surfaces, comparison_colors)),
            )
        )

    surfaces = []
    for axes, title, groups in panels:
        axes.clear()
        axes.set_title(title)
        axes.set_axis_off()
        for prepared, color in groups:
            if prepared is None:
                continue
            vertices, faces = prepared
            if len(vertices) and len(faces):
                surfaces.append((axes, vertices, faces, color))

    if not surfaces:
        return
    bounds = np.concatenate([vertices for _axes, vertices, _faces, _color in surfaces])
    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) * 0.55, 1.0)
    for axes, vertices, faces, color in surfaces:
        base_color = np.asarray(to_rgba(color))
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        normals /= lengths[:, None]
        light = np.asarray([0.3, -0.4, 0.85])
        light /= np.linalg.norm(light)
        illumination = 0.35 + 0.65 * np.maximum(normals @ light, 0.0)
        face_colors = np.column_stack(
            (base_color[:3][None, :] * illumination[:, None], np.full(len(faces), 0.96))
        )
        collection = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors="none",
            antialiased=True,
            zsort="average",
        )
        axes.add_collection3d(collection)
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.set_zlim(center[2] - radius, center[2] + radius)
        axes.set_box_aspect((1, 1, 1))
def prepare_structure_surfaces(
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    ct_series: CTSeries | None,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, ...]:
    return tuple(
        _structure_surface(mask, ct_series) if mask is not None and np.any(mask) else None
        for mask in (left_mask, right_mask)
    )


def prepare_comparison_surfaces(
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    ct_series: CTSeries | None,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, ...]:
    if left_mask is None or right_mask is None:
        return (None, None, None)
    if left_mask.shape != right_mask.shape:
        raise ValueError("As máscaras 3D precisam ter a mesma grade para comparação.")
    region = np.empty(left_mask.shape, dtype=bool)
    surfaces = []
    for index, (exclusive_mask, included_mask) in enumerate(
        (
            (left_mask, right_mask),
            (left_mask, right_mask),
            (right_mask, left_mask),
        )
    ):
        if index == 1:
            np.logical_and(left_mask, right_mask, out=region)
        else:
            np.logical_not(included_mask, out=region)
            np.logical_and(exclusive_mask, region, out=region)
        surfaces.append(
            _structure_surface(region, ct_series) if np.any(region) else None
        )
    return tuple(surfaces)


def _structure_surface(
    mask: np.ndarray, ct_series: CTSeries | None, smoothing_sigma: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Create a display-resolution surface in patient coordinates."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    start = np.maximum(coordinates.min(axis=0) - 2, 0)
    stop = np.minimum(coordinates.max(axis=0) + 3, mask.shape)
    cropped = mask[
        start[0] : stop[0],
        start[1] : stop[1],
        start[2] : stop[2],
    ]
    display_mask, strides = coarsen_mask(cropped, maximum_axis_size=88)
    padded = np.pad(display_mask.astype(np.float32), 2)
    smoothed = gaussian_filter(padded, sigma=smoothing_sigma)
    level = float(smoothed.max()) * 0.5
    vertices, faces, _normals, _values = marching_cubes(
        smoothed, level=level, step_size=1, allow_degenerate=False
    )
    vertices = (vertices - 2.0) * np.asarray(strides)
    vertices += start

    if ct_series is None:
        physical_vertices = vertices
    else:
        first = ct_series.datasets[0]
        orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
        column_direction, row_direction = orientation[:3], orientation[3:]
        row_offsets = (
            vertices[:, 0, None] * ct_series.row_spacing * row_direction[None, :]
        )
        column_offsets = (
            vertices[:, 1, None] * ct_series.column_spacing * column_direction[None, :]
        )
        slice_positions = np.asarray(
            [dataset.ImagePositionPatient for dataset in ct_series.datasets], dtype=float
        )
        slice_coordinates = np.clip(vertices[:, 2], 0, len(slice_positions) - 1)
        slice_origins = np.column_stack(
            [
                np.interp(
                    slice_coordinates,
                    np.arange(len(slice_positions)),
                    slice_positions[:, axis],
                )
                for axis in range(3)
            ]
        )
        physical_vertices = slice_origins + row_offsets + column_offsets
    return physical_vertices, faces


def sync_3d_view(source_axes, target_axes) -> None:
    target_axes.view_init(
        elev=source_axes.elev,
        azim=source_axes.azim,
        roll=getattr(source_axes, "roll", 0),
    )


def compute_dvh_curves(
    ct_series: CTSeries,
    dose_path: str | Path,
    left_masks: dict[str, np.ndarray],
    right_mask: np.ndarray,
    patient_mask: np.ndarray,
    progress=None,
) -> tuple[list[DVHCurve], np.ndarray, np.ndarray]:
    """Calculate DVHs for the union, expansions, and single physician ROI."""
    if not left_masks:
        raise ValueError("Selecione ao menos uma estrutura de auto-segmentação.")
    combined = np.logical_or.reduce(list(left_masks.values()))
    if progress:
        progress(0, "Unindo estruturas, erodindo patient e calculando margens")
    eroded_patient = erode_mask(patient_mask, ct_series, 5.0)
    if not np.any(eroded_patient):
        raise ValueError(
            'A estrutura usada como "patient" ficou vazia após erosão de 0,5 cm. '
            "Escolha uma estrutura substituta mais ampla."
        )
    margin_05 = margin_mask(combined, ct_series, 5.0) & eroded_patient
    if progress:
        progress(8, "Margem de 0,5 cm concluída")
    margin_10 = margin_mask(combined, ct_series, 10.0) & eroded_patient
    if progress:
        progress(16, "Margem de 1 cm concluída; carregando RTDOSE")
    masks = [
        (f"Auto-segmentação — {name}", mask)
        for name, mask in left_masks.items()
    ] + [
        ("Auto-segmentação — união", combined),
        ("Auto-segmentação — margem 0,5 cm", margin_05),
        ("Auto-segmentação — margem 1 cm", margin_10),
        ("PTV Original", right_mask),
    ]
    dose_grid = load_dose_grid(dose_path)
    if progress:
        progress(20, "RTDOSE carregado; validando geometria e unidades")
    curves = []
    total_curves = len(masks)
    for index, (name, mask) in enumerate(masks, 1):
        if progress:
            progress(
                20 + (index - 1) * 80 / total_curves,
                f"Interpolando dose e calculando DVH ({index}/{total_curves}): {name}",
            )
        doses, voxel_volume = sample_dose_on_mask(mask, ct_series, dose_grid=dose_grid)
        curves.append(create_dvh_curve(name, doses, voxel_volume))
        if progress:
            progress(
                20 + index * 80 / total_curves,
                f"DVH concluído ({index}/{total_curves}): {name}",
            )
    return curves, combined, right_mask


def export_dvh_csv(
    curves: list[DVHCurve], output_path: str | Path, progress=None
) -> None:
    """Write a Brazilian-locale CSV with dose, absolute volume, and percent."""
    with Path(output_path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";", lineterminator="\n")
        writer.writerow(("Estrutura", "Dose (Gy)", "Volume (cm3)", "Volume (%)"))
        for index, curve in enumerate(curves, 1):
            rows = []
            for dose, volume, percent in zip(
                curve.dose_gy, curve.volume_cc, curve.volume_percent
            ):
                rows.append(
                    (
                        curve.name,
                        f"{dose:.4f}".replace(".", ","),
                        f"{volume:.6f}".replace(".", ","),
                        f"{percent:.4f}".replace(".", ","),
                    )
                )
            writer.writerows(rows)
            if progress:
                progress(
                    index * 100 / max(1, len(curves)),
                    f"Exportando tabela DVH: {curve.name}",
                )


def export_dvh_plots(
    curves: list[DVHCurve], output_base: str | Path, progress=None
) -> tuple[Path, Path]:
    """Save the percent and absolute-volume DVHs as separate PNG files."""
    base = Path(output_base).with_suffix("")
    outputs = (
        base.with_name(f"{base.name}_volume_percentual.png"),
        base.with_name(f"{base.name}_volume_cm3.png"),
    )
    plot_specs = (
        (outputs[0], "volume_percent", "DVH — volume percentual", "Volume (%)"),
        (outputs[1], "volume_cc", "DVH — volume absoluto", "Volume (cm³)"),
    )
    for index, (output, volume_attribute, title, ylabel) in enumerate(plot_specs, 1):
        figure = Figure(figsize=(8, 6), constrained_layout=True)
        axes = figure.add_subplot(1, 1, 1)
        for curve in curves:
            axes.plot(curve.dose_gy, getattr(curve, volume_attribute), label=curve.name)
        axes.set(title=title, xlabel="Dose (Gy)", ylabel=ylabel)
        if volume_attribute == "volume_percent":
            axes.set_ylim(0, 100)
        axes.grid(True, alpha=0.3)
        axes.legend(
            fontsize="small",
            loc="upper left" if volume_attribute == "volume_cc" else "best",
        )
        figure.savefig(output, dpi=150)
        if progress:
            progress(
                index * 50,
                f"Salvando gráfico {index}/{len(plot_specs)}: {title}",
            )
    return outputs


class ROISelectionPanel(ttk.Frame):
    def __init__(self, master, on_change=None, height=190):
        super().__init__(master)
        self._on_change = on_change
        self._names: list[str] = []
        self._variables: list[tk.BooleanVar] = []
        self._updating = False
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.rows = ttk.Frame(self.canvas)
        self.rows.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self._window = self.canvas.create_window((0, 0), window=self.rows, anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._window, width=event.width),
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.scrollbar = scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.rows.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units"))
        self.rows.bind("<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units"))
        self.rows.bind("<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units"))

    def _mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(direction, "units")
        return "break"

    def set_items(self, names: list[str], selected: list[str] | None = None) -> None:
        selected_names = set(selected or [])
        self._updating = True
        try:
            for child in self.rows.winfo_children():
                child.destroy()
            self._names = list(names)
            self._variables = []
            for name in self._names:
                variable = tk.BooleanVar(value=name in selected_names)
                self._variables.append(variable)
                ttk.Checkbutton(
                    self.rows,
                    text=name,
                    variable=variable,
                    command=self._selection_changed,
                ).pack(anchor="w", fill=tk.X, padx=4, pady=1)
                checkbox = self.rows.winfo_children()[-1]
                checkbox.bind("<MouseWheel>", self._mousewheel)
                checkbox.bind(
                    "<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units")
                )
                checkbox.bind(
                    "<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units")
                )
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        finally:
            self._updating = False

    def get_selected(self) -> list[str]:
        return [
            name
            for name, variable in zip(self._names, self._variables)
            if variable.get()
        ]

    def get_items(self) -> list[str]:
        return list(self._names)

    def selection_set_names(self, names: list[str], notify: bool = True) -> None:
        selected_names = set(names)
        self._updating = True
        try:
            for name, variable in zip(self._names, self._variables):
                variable.set(name in selected_names)
        finally:
            self._updating = False
        if notify:
            self._selection_changed()

    def set_state(self, state: str) -> None:
        for child in self.rows.winfo_children():
            child.configure(state=state)

    def _selection_changed(self) -> None:
        if not self._updating and self._on_change:
            self._on_change()


class DvhApplication:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Comparador RTSTRUCT e DVH")
        self.root.geometry("1320x700")
        self.folder: Path | None = None
        self.ct_files: list[DicomFile] = []
        self.dose_files: list[DicomFile] = []
        self.rtstruct_files: list[DicomFile] = []
        self.ct_series: CTSeries | None = None
        self.curves: list[DVHCurve] = []
        self.three_d_window: tk.Toplevel | None = None
        self.three_d_figure: Figure | None = None
        self.three_d_canvas: FigureCanvasTkAgg | None = None
        self.ax_left_3d = None
        self.ax_right_3d = None
        self.ax_comparison_3d = None
        self.dvh_window: tk.Toplevel | None = None
        self.dvh_figure: Figure | None = None
        self.dvh_canvas: FigureCanvasTkAgg | None = None
        self.ax_percent = None
        self.ax_cc = None
        self._dvh_plot_lines: dict[object, list[tuple[DVHCurve, object]]] = {}
        self._dvh_cursor_lines: dict[object, object] = {}
        self._dvh_cursor_labels: dict[object, object] = {}
        self._task_queue: queue.Queue = queue.Queue()
        self._progress_queue: queue.Queue = queue.Queue()
        self._task_active = False
        self._busy_widgets: list[tk.Widget] = []
        self._mask_cache: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self._mask_nonempty: dict[tuple[str, str, str], bool] = {}
        self._mask_cache_bytes = 0
        self._surface_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._containment_cache: OrderedDict[
            tuple[str, str, str, tuple[str, ...]], dict[str, float]
        ] = OrderedDict()
        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Selecionar pasta DICOM", command=self.select_folder).grid(
            row=0, column=0, padx=4, pady=3
        )
        self.folder_label = ttk.Label(controls, text="Nenhuma pasta selecionada")
        self.folder_label.grid(row=0, column=1, columnspan=5, sticky="w")

        ttk.Label(controls, text="Imagem CT:").grid(row=1, column=0, sticky="e", padx=4)
        self.ct_combo = ttk.Combobox(controls, state="readonly", width=52)
        self.ct_combo.grid(row=1, column=1, sticky="ew", padx=4)
        self.ct_combo.bind("<<ComboboxSelected>>", lambda _event: self._ct_selection_changed())
        ttk.Label(controls, text="Dose RTDOSE:").grid(row=1, column=2, sticky="e", padx=4)
        self.dose_combo = ttk.Combobox(controls, state="readonly", width=52)
        self.dose_combo.grid(row=1, column=3, sticky="ew", padx=4)
        ttk.Button(
            controls, text="Procurar RTDOSE...", command=self.select_external_dose
        ).grid(row=1, column=4, padx=4)

        ttk.Label(controls, text="Auto-segmentação (esquerda):").grid(
            row=2, column=0, sticky="e", padx=4
        )
        self.left_file = ttk.Combobox(controls, state="readonly", width=52)
        self.left_file.grid(row=2, column=1, sticky="ew", padx=4)
        self.left_file.bind("<<ComboboxSelected>>", lambda _event: self.load_roi_choices())
        ttk.Label(controls, text="PTV (direita):").grid(row=2, column=2, sticky="e", padx=4)
        self.right_file = ttk.Combobox(controls, state="readonly", width=52)
        self.right_file.grid(row=2, column=3, sticky="ew", padx=4)
        self.right_file.bind("<<ComboboxSelected>>", lambda _event: self.load_roi_choices())
        ttk.Button(
            controls, text="Abrir RTSTRUCT...", command=lambda: self.select_rtstruct_file("left")
        ).grid(row=2, column=4, padx=4)
        ttk.Button(
            controls, text="Abrir RTSTRUCT...", command=lambda: self.select_rtstruct_file("right")
        ).grid(row=2, column=5, padx=4)

        selection_controls = ttk.Frame(self.root, padding=(8, 0))
        selection_controls.pack(fill=tk.BOTH, expand=True)
        left_selection = ttk.LabelFrame(
            selection_controls, text="Auto-segmentação: marque as estruturas desejadas"
        )
        left_selection.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        self.left_rois = ROISelectionPanel(left_selection, on_change=None, height=320)
        self.left_rois.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        right_selection = ttk.LabelFrame(
            selection_controls, text="PTV: marque as estruturas de referência desejadas"
        )
        right_selection.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        self.right_rois = ROISelectionPanel(right_selection, on_change=None, height=320)
        self.right_rois.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        selection_controls.columnconfigure(0, weight=1, uniform="roi")
        selection_controls.columnconfigure(1, weight=1, uniform="roi")
        selection_controls.rowconfigure(0, weight=1)

        actions = ttk.Frame(self.root, padding=(8, 0))
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Iniciar análise 3D", command=self.start_analysis).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(actions, text="Calcular DVH", command=self.calculate).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(actions, text="Tabela de métricas", command=self.calculate_metrics).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(actions, text="Exportar DVH (.csv)", command=self.export).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(actions, text="Salvar gráficos DVH (2 PNG)", command=self.export_plots).pack(
            side=tk.LEFT, padx=4, pady=4
        )

        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 8))
        status_line = ttk.Frame(status_bar)
        status_line.pack(fill=tk.X)
        self.status = ttk.Label(
            status_line, text="Selecione a pasta com os arquivos DICOM.", anchor="w"
        )
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.percent_label = ttk.Label(status_line, text="0%", width=5, anchor="e")
        self.percent_label.pack(side=tk.RIGHT, padx=(8, 0))
        self.spinner = ttk.Label(status_line, text="●", width=2, anchor="center")
        self.spinner.pack(side=tk.RIGHT, padx=(8, 0))
        self.progress = ttk.Progressbar(
            status_bar, mode="determinate", maximum=100, value=0
        )
        self.progress.pack(fill=tk.X, pady=(4, 0))
        self._spinner_frames = ("◐", "◓", "◑", "◒")
        self._spinner_after: str | None = None
        self._busy_widgets = [
            widget
            for widget in self._descendants(self.root)
            if widget.winfo_class() in {"TButton", "TCombobox"}
            or isinstance(widget, ROISelectionPanel)
        ]

    @staticmethod
    def _descendants(widget: tk.Widget):
        for child in widget.winfo_children():
            yield child
            yield from DvhApplication._descendants(child)

    def _start_task(self, label: str, operation, on_success) -> None:
        if self._task_active:
            return
        self._task_active = True
        self.status.configure(text=label)
        self.progress.configure(value=0)
        self.percent_label.configure(text="0%")
        while not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except queue.Empty:
                break
        for widget in self._busy_widgets:
            if isinstance(widget, ROISelectionPanel):
                widget.set_state(tk.DISABLED)
            else:
                widget.state(["disabled"])
        self._animate_spinner(0)
        self.root.update_idletasks()

        def report_progress(percent: float, detail: str) -> None:
            self._progress_queue.put((max(0.0, min(100.0, percent)), detail))

        def execute():
            try:
                result = operation(report_progress)
            except Exception as error:
                self._task_queue.put((None, error, on_success))
            else:
                self._task_queue.put((result, None, on_success))

        threading.Thread(target=execute, daemon=True).start()
        self.root.after(100, self._poll_task)

    def _animate_spinner(self, frame: int) -> None:
        if not self._task_active:
            return
        self.spinner.configure(text=self._spinner_frames[frame])
        self._spinner_after = self.root.after(
            120, self._animate_spinner, (frame + 1) % len(self._spinner_frames)
        )

    def _poll_task(self) -> None:
        while True:
            try:
                percent, detail = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self.progress.configure(value=percent)
            self.percent_label.configure(text=f"{percent:.0f}%")
            self.status.configure(text=detail)
        try:
            result, error, on_success = self._task_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_task)
            return
        self.progress.configure(value=100 if error is None else self.progress["value"])
        if error is None:
            self.percent_label.configure(text="100%")
        if self._spinner_after is not None:
            self.root.after_cancel(self._spinner_after)
            self._spinner_after = None
        self.spinner.configure(text="●")
        self._task_active = False
        for widget in self._busy_widgets:
            if isinstance(widget, ROISelectionPanel):
                widget.set_state(tk.NORMAL)
            else:
                widget.state(["!disabled"])
        if error is not None:
            self.status.configure(text=f"Falha: {error}")
            messagebox.showerror("Erro durante o processamento", str(error))
            return
        try:
            on_success(result)
        except Exception as callback_error:
            self.status.configure(text=f"Falha ao concluir: {callback_error}")
            messagebox.showerror("Erro ao concluir o processamento", str(callback_error))

    def _selected_path(self, combo: ttk.Combobox, items: list[DicomFile]) -> Path:
        index = combo.current()
        if index < 0 or index >= len(items):
            raise ValueError("Selecione um arquivo DICOM na lista.")
        return items[index].path

    def _choose_patient_structure(
        self, left_path: Path, right_path: Path
    ) -> tuple[Path, str] | None:
        options = [
            (right_path, name, "RTSTRUCT do PTV")
            for name in self.right_rois.get_items()
        ] + [
            (left_path, name, "RTSTRUCT auto-segmentação")
            for name in self.left_rois.get_items()
        ]
        patient_options = [
            option for option in options if option[1].strip().casefold() == "patient"
        ]
        if patient_options:
            path, name, _source = patient_options[0]
            return path, name
        if not options:
            messagebox.showerror(
                "Estrutura de limite necessária",
                'Não há estruturas disponíveis para substituir "patient".',
            )
            return None

        dialog = tk.Toplevel(self.root)
        dialog.title('Escolher substituta para "patient"')
        dialog.transient(self.root)
        dialog.resizable(True, True)
        ttk.Label(
            dialog,
            text=(
                'A estrutura "patient" não foi encontrada nos RTSTRUCTs selecionados.\n'
                "Escolha a estrutura que delimita o corpo. Ela será erodida em 0,5 cm "
                "e usada para limitar as margens do DVH."
            ),
            wraplength=560,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=12, pady=(12, 8))
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        choices = tk.Listbox(list_frame, height=min(16, max(5, len(options))))
        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=choices.yview
        )
        choices.configure(yscrollcommand=scrollbar.set)
        choices.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for _path, name, source in options:
            choices.insert(tk.END, f"{source}: {name}")
        result: list[tuple[Path, str] | None] = [None]

        def accept():
            selected = choices.curselection()
            if not selected:
                messagebox.showwarning(
                    "Seleção necessária",
                    "Selecione uma estrutura para continuar.",
                    parent=dialog,
                )
                return
            path, name, _source = options[selected[0]]
            result[0] = (path, name)
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=(4, 12))
        ttk.Button(buttons, text="Usar estrutura", command=accept).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(buttons, text="Cancelar", command=dialog.destroy).pack(
            side=tk.RIGHT
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set()
        choices.focus_set()
        self.root.wait_window(dialog)
        return result[0]

    def select_folder(self) -> None:
        selected = filedialog.askdirectory(title="Selecione a pasta com os arquivos DICOM")
        if not selected:
            return

        def loaded(result):
            ct, doses, structures = result
            self.folder = Path(selected)
            self.ct_files, self.dose_files, self.rtstruct_files = ct, doses, structures
            self._populate_folder_selection(ct, doses, structures)

        self._start_task(
            "Lendo arquivos DICOM da pasta selecionada...",
            lambda report: discover_dicom_files(selected, progress=report),
            loaded,
        )

    def _populate_folder_selection(
        self,
        ct: list[DicomFile],
        doses: list[DicomFile],
        structures: list[DicomFile],
    ) -> None:
        self.ct_series = None
        self.folder_label.configure(text=str(self.folder))
        self._set_combo(self.ct_combo, ct)
        self._set_combo(self.dose_combo, doses)
        left_index, right_index = choose_default_rtstructs(structures)
        self._set_combo(self.left_file, structures, left_index)
        self._set_combo(self.right_file, structures, right_index)
        self.left_rois.set_items([])
        self.right_rois.set_items([])
        if doses:
            dose_status = f"{len(doses)} RTDOSE"
        else:
            dose_status = "nenhum RTDOSE (use Procurar RTDOSE...)"
        self.status.configure(
            text=f"Encontrados: {len(ct)} série(s) CT, {dose_status} e {len(structures)} RTSTRUCT."
        )
        if not ct or not structures:
            messagebox.showwarning(
                "Arquivos insuficientes",
                "É necessário encontrar ao menos uma série CT e um RTSTRUCT. "
                "Use Abrir RTSTRUCT... para carregar outro arquivo se necessário. "
                "Se a dose não estiver nesta pasta, use Procurar RTDOSE...",
            )
        self.load_roi_choices()

    def select_external_dose(self) -> None:
        selected = filedialog.askopenfilename(
            title="Selecione um arquivo de dose RTDOSE",
            filetypes=[("Arquivos DICOM", "*.dcm *.dicom"), ("Todos os arquivos", "*.*")],
        )
        if not selected:
            return

        def loaded(dose):
            self._add_dose_file(dose)

        self._start_task(
            "Verificando arquivo RTDOSE...",
            lambda report: self._inspect_dose_with_progress(selected, report),
            loaded,
        )

    @staticmethod
    def _inspect_dose_with_progress(path: str, report):
        report(10, "Lendo cabeçalho do RTDOSE")
        dose = inspect_dose_file(path)
        report(100, "RTDOSE validado")
        return dose

    def _add_dose_file(self, dose: DicomFile) -> None:
        existing = next(
            (index for index, item in enumerate(self.dose_files) if item.path == dose.path),
            None,
        )
        if existing is None:
            self.dose_files.append(dose)
            self.dose_combo["values"] = [
                item.label for item in self.dose_files
            ]
            existing = len(self.dose_files) - 1
        self.dose_combo.current(existing)
        self.status.configure(text=f"Dose RTDOSE selecionada: {dose.path}")

    def select_rtstruct_file(self, side: str) -> None:
        selected = filedialog.askopenfilename(
            title="Selecione o arquivo RTSTRUCT",
            filetypes=[("Arquivos DICOM", "*.dcm *.dicom"), ("Todos os arquivos", "*.*")],
            initialdir=str(self.folder) if self.folder else None,
        )
        if not selected:
            return
        path = Path(selected)

        def read_structure_file(report):
            names = read_roi_names(path, progress=report)
            header = read_dicom(path, stop_before_pixels=True)
            item = DicomFile(
                path,
                str(getattr(header, "SeriesInstanceUID", "")),
                str(getattr(header, "SeriesDescription", "")),
            )
            return item, names

        self._start_task(
            "Lendo RTSTRUCT e carregando estruturas...",
            read_structure_file,
            lambda result: self._add_rtstruct_file(side, *result),
        )

    def _add_rtstruct_file(self, side: str, item: DicomFile, names: list[str]) -> None:
        index = next(
            (
                position
                for position, existing in enumerate(self.rtstruct_files)
                if existing.path == item.path
            ),
            None,
        )
        if index is None:
            self.rtstruct_files.append(item)
            index = len(self.rtstruct_files) - 1
        combo = self.left_file if side == "left" else self.right_file
        combo["values"] = [struct.label for struct in self.rtstruct_files]
        combo.current(index)
        listbox = self.left_rois if side == "left" else self.right_rois
        filtered_names = filter_selectable_roi_names(names)
        selected = (
            [filtered_names[index] for index in preferred_ptv_indices(filtered_names)]
            if side == "right"
            else []
        )
        listbox.set_items(filtered_names, selected)

    @staticmethod
    def _set_combo(
        combo: ttk.Combobox, items: list[DicomFile], selected_index: int = 0
    ) -> None:
        combo["values"] = [item.label for item in items]
        if items:
            combo.current(selected_index)

    def load_roi_choices(self) -> None:
        indexes = (self.left_file.current(), self.right_file.current())
        paths = [
            self.rtstruct_files[index].path if 0 <= index < len(self.rtstruct_files) else None
            for index in indexes
        ]

        def read_roi_lists(report):
            names = []
            for index, path in enumerate(paths):
                if path:
                    names.append(
                        read_roi_names(
                            path,
                            progress=lambda percent, message, index=index: report(
                                (index * 50) + percent * 0.5, message
                            ),
                        )
                    )
                else:
                    names.append([])
            return [filter_selectable_roi_names(side_names) for side_names in names]

        self._start_task(
            "Lendo nomes das estruturas RTSTRUCT...",
            read_roi_lists,
            self._populate_roi_choices,
        )

    def _populate_roi_choices(self, roi_lists: list[list[str]]) -> None:
        left_names, right_names = roi_lists
        self.left_rois.set_items(left_names)
        defaults = [
            right_names[index] for index in preferred_ptv_indices(right_names)
        ]
        self.right_rois.set_items(right_names, defaults)

    def _load_inputs(self) -> tuple[CTSeries, Path, Path, Path, list[str], list[str]]:
        if self.folder is None:
            raise ValueError("Selecione primeiro uma pasta DICOM.")
        if self.ct_combo.current() < 0:
            raise ValueError("Selecione uma série CT.")
        ct_file = self.ct_files[self.ct_combo.current()]
        left_path = self._selected_path(self.left_file, self.rtstruct_files)
        right_path = self._selected_path(self.right_file, self.rtstruct_files)
        if left_path == right_path:
            raise ValueError("Selecione dois arquivos RTSTRUCT diferentes para a comparação.")
        return (
            load_ct_series(
                self.folder, ct_file.series_uid, ct_file.instance_paths or None
            ),
            self._selected_path(self.dose_combo, self.dose_files),
            left_path,
            right_path,
            self.left_rois.get_selected(),
            self.right_rois.get_selected(),
        )

    def show_3d(self) -> None:
        self.start_analysis()

    def _ct_selection_changed(self) -> None:
        self.ct_series = None

    def start_analysis(self) -> None:
        if self.folder is None or self.ct_combo.current() < 0:
            messagebox.showwarning("Seleção necessária", "Selecione uma pasta e uma série CT.")
            return
        right_names = self.right_rois.get_selected()
        if not right_names:
            no_ptv = not preferred_ptv_indices(self.right_rois.get_items())
            messagebox.showwarning(
                "Seleção necessária",
                (
                    "Nenhuma estrutura PTV foi encontrada. Marque manualmente "
                    "um ou mais PTVs antes de iniciar."
                    if no_ptv
                    else "Marque um ou mais PTVs antes de iniciar."
                ),
            )
            return
        left_index, right_index = self.left_file.current(), self.right_file.current()
        if not (0 <= left_index < len(self.rtstruct_files)) or not (
            0 <= right_index < len(self.rtstruct_files)
        ):
            messagebox.showwarning(
                "Seleção necessária", "Selecione os arquivos RTSTRUCT dos dois conjuntos."
            )
            return
        series_file = self.ct_files[self.ct_combo.current()]
        left_path = self.rtstruct_files[left_index].path
        right_path = self.rtstruct_files[right_index].path
        if left_path == right_path:
            messagebox.showwarning(
                "Seleção necessária", "Escolha arquivos RTSTRUCT diferentes para comparar."
            )
            return
        all_left_names = self.left_rois.get_items()
        if not all_left_names:
            messagebox.showwarning(
                "Sem estruturas",
                "O RTSTRUCT da auto-segmentação não contém estruturas selecionáveis.",
            )
            return
        containment_key = (
            str(left_path),
            str(right_path),
            series_file.series_uid,
            tuple(sorted(right_names)),
        )

        def find_contained(report):
            ct_series = self.ct_series or load_ct_series(
                self.folder,
                series_file.series_uid,
                series_file.instance_paths or None,
                progress=lambda percent, message: report(percent * 0.2, message),
            )
            if containment_key in self._containment_cache:
                report(95, "Usando resultados de inclusão já calculados")
                return ct_series, self._containment_cache[containment_key], {}
            report(20, "Rasterizando estruturas candidatas para calcular a inclusão")
            left_masks = load_structure_masks(
                ct_series,
                left_path,
                all_left_names,
                allow_empty=True,
                progress=lambda percent, message: report(20 + percent * 0.5, message),
            )
            right_masks = self._cached_structure_masks(
                ct_series,
                right_path,
                series_file.series_uid,
                right_names,
                progress=lambda percent, message: report(70 + percent * 0.2, message),
            )
            right_union = np.logical_or.reduce(list(right_masks.values()))
            percentages = structure_containment_percentages(left_masks, right_union)
            self._containment_cache[containment_key] = percentages
            self._containment_cache.move_to_end(containment_key)
            while len(self._containment_cache) > 16:
                self._containment_cache.popitem(last=False)
            report(100, "Análise de inclusão concluída")
            return ct_series, percentages, left_masks

        def show_suggestions(result):
            self.ct_series, percentages, candidate_masks = result
            matches = [
                (name, percentage)
                for name, percentage in percentages.items()
                if percentage >= 0.6
            ]
            if matches:
                description = "\n".join(
                    f"• {name}: {percentage:.1%}" for name, percentage in matches[:20]
                )
                if len(matches) > 20:
                    description += f"\n… e mais {len(matches) - 20} estrutura(s)"
                accepted = messagebox.askyesno(
                    "Confirmar estruturas auto-segmentadas",
                    "Estas estruturas estão pelo menos 60% dentro da união das "
                    "PTVs selecionados:\n\n"
                    f"{description}\n\n"
                    "Deseja selecioná-las para a análise?",
                )
                if accepted:
                    self.left_rois.selection_set_names(
                        [name for name, _percentage in matches], notify=False
                    )
            else:
                details = containment_failure_message(percentages)
                messagebox.showinfo(
                    "Nenhuma estrutura correspondente",
                    f"{details}\n\nAs caixas selecionadas manualmente serão mantidas.",
                )
            selected_left = self.left_rois.get_selected()
            self._cache_selected_masks(
                left_path,
                series_file.series_uid,
                {
                    name: candidate_masks[name]
                    for name in selected_left
                    if name in candidate_masks
                },
            )
            self._render_selected_3d()

        if self.three_d_window is not None and self.three_d_window.winfo_exists():
            self.three_d_window.withdraw()
        self._start_task(
            "Iniciando análise: verificando estruturas selecionadas...",
            find_contained,
            show_suggestions,
        )

    def _cache_selected_masks(
        self, path: Path, series_uid: str, masks: dict[str, np.ndarray]
    ) -> None:
        for name, mask in masks.items():
            key = (str(path), series_uid, name)
            previous = self._mask_cache.pop(key, None)
            if previous is not None:
                self._mask_cache_bytes -= previous.nbytes
            self._mask_cache[key] = mask
            self._mask_nonempty[key] = bool(np.any(mask))
            self._mask_cache_bytes += mask.nbytes
        while self._mask_cache_bytes > 512 * 1024 * 1024 and self._mask_cache:
            old_key, old_mask = self._mask_cache.popitem(last=False)
            self._mask_nonempty.pop(old_key, None)
            self._mask_cache_bytes -= old_mask.nbytes

    def _render_selected_3d(self) -> None:
        if self.folder is None or self.ct_combo.current() < 0:
            return
        series_file = self.ct_files[self.ct_combo.current()]
        left_index, right_index = self.left_file.current(), self.right_file.current()
        left_names = self.left_rois.get_selected()
        right_names = self.right_rois.get_selected()
        left_path = (
            self.rtstruct_files[left_index].path
            if 0 <= left_index < len(self.rtstruct_files)
            else None
        )
        right_path = (
            self.rtstruct_files[right_index].path
            if 0 <= right_index < len(self.rtstruct_files)
            else None
        )
        existing_series = self.ct_series

        def create_masks(report):
            ct_series = existing_series or load_ct_series(
                self.folder,
                series_file.series_uid,
                series_file.instance_paths or None,
                progress=lambda percent, message: report(percent * 0.18, message),
            )
            all_left_masks = (
                self._cached_structure_masks(
                    ct_series,
                    left_path,
                    series_file.series_uid,
                    left_names,
                    progress=lambda percent, message: report(
                        18 + percent * 0.36, message
                    ),
                )
                if left_path is not None and left_names
                else {}
            )
            left_mask = (
                np.logical_or.reduce(
                    [all_left_masks[name] for name in left_names if name in all_left_masks]
                )
                if left_names and all_left_masks
                else None
            )
            right_masks = (
                self._cached_structure_masks(
                    ct_series,
                    right_path,
                    series_file.series_uid,
                    right_names,
                    progress=lambda percent, message: report(
                        54 + percent * 0.26, message
                    ),
                )
                if right_path is not None and right_names
                else {}
            )
            right_mask = (
                np.logical_or.reduce(list(right_masks.values())) if right_masks else None
            )
            surfaces = (
                self._cached_surface(
                    left_mask, left_path, series_file.series_uid, left_names, ct_series
                ),
                self._cached_surface(
                    right_mask, right_path, series_file.series_uid, right_names, ct_series
                ),
            )
            comparison_surfaces = None
            if left_mask is not None and right_mask is not None:
                report(78, "Preparando superfícies da sobreposição 3D")
                comparison_surfaces = prepare_comparison_surfaces(
                    left_mask, right_mask, ct_series
                )
            report(95, "Finalizando superfícies 3D")
            report(100, "Atualizando visualização")
            return ct_series, left_mask, right_mask, surfaces, comparison_surfaces

        def rendered(result):
            (
                self.ct_series,
                left_mask,
                right_mask,
                surfaces,
                comparison_surfaces,
            ) = result
            self._draw_3d(left_mask, right_mask, surfaces, comparison_surfaces)
            self.status.configure(
                text="Visualização atualizada; conjuntos em coordenadas físicas comuns."
            )

        self._start_task(
            "Carregando e renderizando as estruturas 3D...",
            create_masks,
            rendered,
        )

    def _cached_structure_masks(
        self,
        ct_series: CTSeries,
        path: Path,
        series_uid: str,
        names: list[str],
        allow_empty: bool = False,
        cache_results: bool = True,
        progress=None,
    ) -> dict[str, np.ndarray]:
        keys = {name: (str(path), series_uid, name) for name in names}
        missing = [name for name, key in keys.items() if key not in self._mask_cache]
        loaded = {}
        if missing:
            loaded = load_structure_masks(
                ct_series,
                path,
                missing,
                allow_empty=allow_empty,
                progress=progress,
            )
            if cache_results:
                self._cache_selected_masks(path, series_uid, loaded)
                for name in loaded:
                    self._mask_cache.move_to_end(keys[name])
        masks = {}
        for name, key in keys.items():
            if name in loaded:
                masks[name] = loaded[name]
            else:
                self._mask_cache.move_to_end(key)
                masks[name] = self._mask_cache[key]
            if not allow_empty:
                nonempty = self._mask_nonempty.get(key)
                if name in loaded and not cache_results:
                    nonempty = bool(np.any(masks[name]))
                if nonempty is None:
                    nonempty = bool(np.any(masks[name]))
                    self._mask_nonempty[key] = nonempty
                if not nonempty:
                    raise ValueError(
                        f"A estrutura '{name}' não possui contornos rasterizáveis."
                    )
        return masks

    def _cached_surface(
        self,
        mask: np.ndarray | None,
        path: Path | None,
        series_uid: str,
        names: list[str],
        ct_series: CTSeries,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if mask is None or not np.any(mask):
            return None
        key = (str(path), series_uid, tuple(sorted(names)))
        if key in self._surface_cache:
            self._surface_cache.move_to_end(key)
            return self._surface_cache[key]
        surface = _structure_surface(mask, ct_series)
        self._surface_cache[key] = surface
        while len(self._surface_cache) > 4:
            self._surface_cache.popitem(last=False)
        return surface

    @staticmethod
    def _load_selected_mask(
        ct_series: CTSeries,
        rtstruct_path: Path | None,
        names: list[str],
        multiple: bool,
    ) -> np.ndarray | None:
        if rtstruct_path is None or not names:
            return None
        masks = load_structure_masks(
            ct_series, rtstruct_path, names if multiple else names[:1]
        )
        return np.logical_or.reduce(list(masks.values()))

    def _draw_3d(
        self,
        left_mask: np.ndarray | None,
        right_mask: np.ndarray | None,
        prepared_surfaces: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None,
        prepared_comparison_surfaces: tuple[
            tuple[np.ndarray, np.ndarray] | None, ...
        ] | None = None,
    ) -> None:
        self._ensure_3d_window()
        draw_structure_3d(
            self.ax_left_3d,
            self.ax_right_3d,
            left_mask,
            right_mask,
            self.ct_series,
            prepared_surfaces,
            self.ax_comparison_3d,
            prepared_comparison_surfaces,
        )
        self.three_d_canvas.draw_idle()

    def _ensure_3d_window(self) -> None:
        if self.three_d_window is not None and self.three_d_window.winfo_exists():
            self.three_d_window.deiconify()
            self.three_d_window.lift()
            return
        self.three_d_window = tk.Toplevel(self.root)
        self.three_d_window.title("Visualização 3D das estruturas")
        self.three_d_window.geometry("1500x760")
        self.three_d_window.protocol("WM_DELETE_WINDOW", self._close_3d_window)
        self.three_d_figure = Figure(figsize=(16, 7), dpi=110, constrained_layout=True)
        self.ax_left_3d = self.three_d_figure.add_subplot(1, 3, 1, projection="3d")
        self.ax_comparison_3d = self.three_d_figure.add_subplot(1, 3, 2, projection="3d")
        self.ax_right_3d = self.three_d_figure.add_subplot(1, 3, 3, projection="3d")
        self.three_d_canvas = FigureCanvasTkAgg(
            self.three_d_figure, master=self.three_d_window
        )
        self.three_d_canvas.mpl_connect("motion_notify_event", self._sync_3d_views)
        self.three_d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _close_3d_window(self) -> None:
        if self.three_d_window is not None and self.three_d_window.winfo_exists():
            self.three_d_window.destroy()
        self.three_d_window = None
        self.three_d_figure = None
        self.three_d_canvas = None
        self.ax_left_3d = None
        self.ax_right_3d = None
        self.ax_comparison_3d = None

    def _ensure_dvh_window(self) -> None:
        if self.dvh_window is not None and self.dvh_window.winfo_exists():
            self.dvh_window.deiconify()
            self.dvh_window.lift()
            return
        self.dvh_window = tk.Toplevel(self.root)
        self.dvh_window.title("Histogramas dose-volume (DVH)")
        self.dvh_window.geometry("1100x800")
        self.dvh_window.protocol("WM_DELETE_WINDOW", self._close_dvh_window)
        self.dvh_figure = Figure(figsize=(11, 8), dpi=110, constrained_layout=True)
        self.ax_percent = self.dvh_figure.add_subplot(2, 1, 1)
        self.ax_cc = self.dvh_figure.add_subplot(2, 1, 2)
        self.dvh_canvas = FigureCanvasTkAgg(self.dvh_figure, master=self.dvh_window)
        self.dvh_canvas.mpl_connect("motion_notify_event", self._update_dvh_cursor)
        self.dvh_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _close_dvh_window(self) -> None:
        if self.dvh_window is not None and self.dvh_window.winfo_exists():
            self.dvh_window.destroy()
        self.dvh_window = None
        self.dvh_figure = None
        self.dvh_canvas = None
        self.ax_percent = None
        self.ax_cc = None

    def _sync_3d_views(self, event) -> None:
        axes = (self.ax_left_3d, self.ax_right_3d, self.ax_comparison_3d)
        source_axes = event.inaxes
        if source_axes not in axes:
            return
        source_view = (
            source_axes.elev,
            source_axes.azim,
            getattr(source_axes, "roll", 0),
        )
        changed = False
        for target_axes in axes:
            if target_axes is None or target_axes is source_axes:
                continue
            target_view = (
                target_axes.elev,
                target_axes.azim,
                getattr(target_axes, "roll", 0),
            )
            if not np.allclose(source_view, target_view, rtol=0, atol=1e-6):
                sync_3d_view(source_axes, target_axes)
                changed = True
        if not changed:
            return
        self.three_d_canvas.draw_idle()

    def calculate(self) -> None:
        try:
            if self.folder is None or self.ct_combo.current() < 0:
                raise ValueError("Selecione uma série CT.")
            if self.dose_combo.current() < 0:
                raise ValueError("Selecione um arquivo RTDOSE.")
            left_index, right_index = self.left_file.current(), self.right_file.current()
            left_names = self.left_rois.get_selected()
            right_names = self.right_rois.get_selected()
            if not left_names or not right_names:
                raise ValueError(
                    "Selecione uma ou mais estruturas de auto-segmentação e um ou mais PTVs."
                )
            if not (0 <= left_index < len(self.rtstruct_files)) or not (
                0 <= right_index < len(self.rtstruct_files)
            ):
                raise ValueError("Selecione os arquivos RTSTRUCT dos dois conjuntos.")
            series_file = self.ct_files[self.ct_combo.current()]
            dose_path = self.dose_files[self.dose_combo.current()].path
            left_path = self.rtstruct_files[left_index].path
            right_path = self.rtstruct_files[right_index].path
            patient_reference = self._choose_patient_structure(left_path, right_path)
            if patient_reference is None:
                return
            patient_path, patient_name = patient_reference
        except (ValueError, IndexError) as error:
            messagebox.showerror("Erro ao calcular DVH", str(error))
            return

        def calculate_dvh(report):
            ct_series = self.ct_series or load_ct_series(
                self.folder,
                series_file.series_uid,
                series_file.instance_paths or None,
                progress=lambda percent, message: report(percent * 0.18, message),
            )
            left_masks = self._cached_structure_masks(
                ct_series,
                left_path,
                series_file.series_uid,
                left_names,
                progress=lambda percent, message: report(18 + percent * 0.18, message),
            )
            right_masks = self._cached_structure_masks(
                ct_series,
                right_path,
                series_file.series_uid,
                right_names,
                progress=lambda percent, message: report(36 + percent * 0.18, message),
            )
            right_union = np.logical_or.reduce(list(right_masks.values()))
            patient_masks = self._cached_structure_masks(
                ct_series,
                patient_path,
                series_file.series_uid,
                [patient_name],
                progress=lambda percent, message: report(
                    54 + percent * 0.1, message
                ),
            )
            curves, _left_union, _right_mask = compute_dvh_curves(
                ct_series,
                dose_path,
                left_masks,
                right_union,
                patient_masks[patient_name],
                progress=lambda percent, message: report(64 + percent * 0.36, message),
            )
            report(100, "Atualizando gráficos DVH")
            return ct_series, curves

        def calculated(result):
            self.ct_series, self.curves = result
            self._draw_dvhs(self.curves)
            self.status.configure(
                text="DVH concluído: ROIs individuais, união, margens e PTV Original."
            )

        self._ensure_dvh_window()
        self._start_task("Calculando dose e histogramas DVH...", calculate_dvh, calculated)

    def _draw_dvhs(self, curves: list[DVHCurve]) -> None:
        self._ensure_dvh_window()
        self.ax_percent.clear()
        self.ax_cc.clear()
        self.ax_percent.set(title="DVH — volume percentual", xlabel="Dose (Gy)", ylabel="Volume (%)")
        self.ax_percent.set_ylim(0, 100)
        self.ax_cc.set(title="DVH — volume absoluto", xlabel="Dose (Gy)", ylabel="Volume (cm³)")
        self.ax_percent.grid(True, alpha=0.3)
        self.ax_cc.grid(True, alpha=0.3)
        self._dvh_plot_lines = {}
        self._dvh_cursor_lines = {}
        self._dvh_cursor_labels = {}
        for axes, volume_attribute in (
            (self.ax_percent, "volume_percent"),
            (self.ax_cc, "volume_cc"),
        ):
            plotted = []
            for curve in curves:
                line, = axes.plot(
                    curve.dose_gy,
                    getattr(curve, volume_attribute),
                    label=curve.name,
                )
                plotted.append((curve, line))
            axes.legend(fontsize="small")
            cursor = axes.axvline(
                0.0, color="black", linewidth=1.0, linestyle="--", visible=False
            )
            readout = axes.text(
                0.98,
                0.04,
                "",
                transform=axes.transAxes,
                ha="right",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
            )
            readout.set_visible(False)
            self._dvh_plot_lines[axes] = plotted
            self._dvh_cursor_lines[axes] = cursor
            self._dvh_cursor_labels[axes] = readout
        self.dvh_canvas.draw_idle()

    def _update_dvh_cursor(self, event) -> None:
        if not self._dvh_plot_lines:
            return
        axes = event.inaxes
        if axes not in self._dvh_plot_lines or event.xdata is None:
            changed = False
            for cursor in self._dvh_cursor_lines.values():
                if cursor.get_visible():
                    cursor.set_visible(False)
                    changed = True
            for readout in self._dvh_cursor_labels.values():
                if readout.get_visible():
                    readout.set_visible(False)
                    changed = True
            if changed:
                self.dvh_canvas.draw_idle()
            return
        for other_axes, other_cursor in self._dvh_cursor_lines.items():
            if other_axes is not axes:
                other_cursor.set_visible(False)
                self._dvh_cursor_labels[other_axes].set_visible(False)
        cursor = self._dvh_cursor_lines[axes]
        cursor.set_xdata([event.xdata, event.xdata])
        cursor.set_visible(True)
        readout = self._dvh_cursor_labels[axes]
        readout.set_text(f"Dose: {event.xdata:.2f} Gy")
        readout.set_visible(True)
        value_attribute = (
            "volume_percent" if axes is self.ax_percent else "volume_cc"
        )
        unit = "%" if value_attribute == "volume_percent" else "cm³"
        plotted = self._dvh_plot_lines[axes]
        for curve, line in plotted:
            value = float(
                np.interp(
                    event.xdata,
                    curve.dose_gy,
                    getattr(curve, value_attribute),
                )
            )
            line.set_label(f"{curve.name} — {value:.1f} {unit}")
        axes.legend(
            handles=[line for _curve, line in plotted],
            fontsize="small",
            loc="upper left" if axes is self.ax_cc else "best",
        )
        self.dvh_canvas.draw_idle()

    def calculate_metrics(self) -> None:
        try:
            if self.folder is None or self.ct_combo.current() < 0:
                raise ValueError("Selecione uma pasta e uma série CT.")
            left_names = self.left_rois.get_selected()
            right_names = self.right_rois.get_selected()
            if not left_names or not right_names:
                raise ValueError(
                    "Selecione ao menos uma estrutura auto-segmentada e um ou mais PTVs."
                )
            left_index, right_index = self.left_file.current(), self.right_file.current()
            if not (0 <= left_index < len(self.rtstruct_files)) or not (
                0 <= right_index < len(self.rtstruct_files)
            ):
                raise ValueError("Selecione os arquivos RTSTRUCT dos dois conjuntos.")
            series_file = self.ct_files[self.ct_combo.current()]
            left_path = self.rtstruct_files[left_index].path
            right_path = self.rtstruct_files[right_index].path
        except (ValueError, IndexError) as error:
            messagebox.showerror("Erro ao calcular métricas", str(error))
            return

        def calculate_comparison(report):
            ct_series = self.ct_series or load_ct_series(
                self.folder,
                series_file.series_uid,
                series_file.instance_paths or None,
                progress=lambda percent, message: report(percent * 0.2, message),
            )
            left_masks = self._cached_structure_masks(
                ct_series,
                left_path,
                series_file.series_uid,
                left_names,
                progress=lambda percent, message: report(20 + percent * 0.34, message),
            )
            right_masks = self._cached_structure_masks(
                ct_series,
                right_path,
                series_file.series_uid,
                right_names,
                progress=lambda percent, message: report(54 + percent * 0.34, message),
            )
            ptv_union = np.logical_or.reduce(list(right_masks.values()))
            report(88, "Calculando Dice e inclusão no PTV; margens das estruturas automáticas")
            rows = structure_comparison_metrics(
                left_masks,
                ptv_union,
                ct_series,
                progress=lambda percent, message: report(
                    88 + percent * 0.12, message
                ),
            )
            report(100, "Tabela de métricas pronta")
            return rows

        def show_metrics(rows):
            window = tk.Toplevel(self.root)
            window.title("Métricas das estruturas auto-segmentadas")
            window.geometry("1050x560")
            ttk.Label(
                window,
                text=(
                    "Dentro do PTV = percentual do volume auto-segmentado contido na união "
                    "dos PTVs selecionados. Nas colunas de margem, a expansão de 0,5/1,0 cm "
                    "é aplicada individualmente à estrutura auto-segmentada; o valor indica "
                    "o percentual do PTV coberto por essa expansão. Dice compara as máscaras "
                    "originais; 100% indica máscaras idênticas."
                ),
                wraplength=1000,
                justify=tk.LEFT,
            ).pack(fill=tk.X, padx=10, pady=10)
            columns = (
                "structure",
                "inside_ptv",
                "ptv_inside_auto_margin_05",
                "ptv_inside_auto_margin_10",
                "dice",
            )
            table_frame = ttk.Frame(window)
            table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            table = ttk.Treeview(
                table_frame, columns=columns, show="headings", selectmode="browse"
            )
            headings = (
                ("structure", "Estrutura auto-segmentada", 300),
                ("inside_ptv", "Dentro do PTV (%)", 160),
                ("ptv_inside_auto_margin_05", "PTV na margem auto 0,5 cm (%)", 200),
                ("ptv_inside_auto_margin_10", "PTV na margem auto 1,0 cm (%)", 200),
                ("dice", "Dice original (%)", 140),
            )
            for column, title, width in headings:
                table.heading(column, text=title)
                table.column(column, width=width, anchor="w" if column == "structure" else "center")
            scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
            table.configure(yscrollcommand=scrollbar.set)
            table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            for row in rows:
                table.insert(
                    "",
                    tk.END,
                    values=(
                        row["structure"],
                        f'{row["inside_ptv_percent"]:.1f}',
                        f'{row["ptv_inside_auto_margin_05_percent"]:.1f}',
                        f'{row["ptv_inside_auto_margin_10_percent"]:.1f}',
                        f'{row["dice_percent"]:.1f}',
                    ),
                )
            self.status.configure(text="Tabela de métricas exibida.")

        self._start_task(
            "Calculando métricas das estruturas...",
            calculate_comparison,
            show_metrics,
        )

    def export(self) -> None:
        if not self.curves:
            messagebox.showwarning("DVH não calculado", "Calcule o DVH antes de exportar.")
            return
        output = filedialog.asksaveasfilename(
            title="Exportar tabela DVH",
            defaultextension=".csv",
            filetypes=[("CSV (separador ;)", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if not output:
            return

        def exported(_result):
            self.status.configure(text=f"Tabela DVH exportada para {output}")
            messagebox.showinfo("Exportação concluída", f"Tabela DVH salva em:\n{output}")

        def write_table(report):
            export_dvh_csv(
                self.curves,
                output,
                progress=lambda percent, message: report(percent, message),
            )
            report(100, "Tabela DVH salva")

        self._start_task(
            "Exportando tabela DVH...",
            write_table,
            exported,
        )

    def export_plots(self) -> None:
        if not self.curves:
            messagebox.showwarning("DVH não calculado", "Calcule o DVH antes de exportar os gráficos.")
            return
        output = filedialog.asksaveasfilename(
            title="Definir nomes dos gráficos DVH",
            defaultextension=".png",
            filetypes=[("Imagem PNG", "*.png")],
        )
        if not output:
            return

        def exported(paths):
            self.status.configure(text=f"Gráficos DVH salvos em {paths[0]} e {paths[1]}")
            messagebox.showinfo(
                "Exportação concluída",
                f"Gráficos salvos em:\n{paths[0]}\n{paths[1]}",
            )

        def write_plots(report):
            paths = export_dvh_plots(
                self.curves,
                output,
                progress=lambda percent, message: report(percent, message),
            )
            report(100, "Gráficos DVH salvos")
            return paths

        self._start_task(
            "Exportando gráficos DVH...",
            write_plots,
            exported,
        )


def main() -> None:
    root = tk.Tk()
    DvhApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
