from collections import defaultdict
from dataclasses import dataclass
import csv
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
import struct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import unicodedata
import webbrowser

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.filereader import read_dataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RTDoseStorage,
    RTStructureSetStorage,
)
from skimage import draw, measure


CT_IMAGE_UID = str(CTImageStorage)
RTSTRUCT_UID = str(RTStructureSetStorage)
RTDOSE_UID = str(RTDoseStorage)
AGATSTON_SCORE_COLUMN = "Agatston_score"
DOSE_MEAN_COLUMN = "Dose_mean_Gy"
DOSE_D0_5CC_COLUMN = "Dose_D0.5cc_Gy"
DOSE_V8_COLUMN = "Dose_V8Gy_percent"
DOSE_V16_COLUMN = "Dose_V16Gy_percent"
DOSE_V20_COLUMN = "Dose_V20Gy_percent"
ARTICLE_REFERENCE_ABNT = (
    "BELARDO, Alfonso et al. Density and volume of cardiac calcifications detected "
    "on planning CT predicts cardiotoxicity after hypo-fractionated whole breast "
    "Radiotherapy. <i>Radiotherapy and Oncology</i>, v. 211, art. 111098, 2025. "
    "DOI: https://doi.org/10.1016/j.radonc.2025.111098."
)
ROI_TRANSLATIONS = {
    "cardiac_area": "Área cardíaca",
    "area_cardiaca": "Área cardíaca",
    "heart": "Coração",
    "aorta": "Aorta",
    "heartchambers_highres": "Câmaras cardíacas",
    "coronary_artery": "Artérias coronárias",
    "coronary_arteries": "Artérias coronárias",
    "art_coron": "Artérias coronárias",
    "myocardium": "Miocárdio",
    "heart_myocardium": "Miocárdio",
    "miocardio": "Miocárdio",
    "atrium_left": "Átrio esquerdo",
    "heart_atrium_left": "Átrio esquerdo",
    "atri_e": "Átrio esquerdo",
    "atrium_right": "Átrio direito",
    "heart_atrium_right": "Átrio direito",
    "atri_d": "Átrio direito",
    "ventricle_left": "Ventrículo esquerdo",
    "heart_ventricle_left": "Ventrículo esquerdo",
    "ventri_e": "Ventrículo esquerdo",
    "ventricle_right": "Ventrículo direito",
    "heart_ventricle_right": "Ventrículo direito",
    "ventri_d": "Ventrículo direito",
    "pulmonary_artery": "Artéria pulmonar",
    "art_pulm": "Artéria pulmonar",
    "pulmonary_vein": "Veia pulmonar",
    "veia_pulm": "Veia pulmonar",
    "artéria pulmonar": "Artéria pulmonar",
}
CARDIAC_ROI_NAMES = {
    "area cardiaca",
    "cardiac area",
    "heart",
    "coracao",
    "aorta",
    "heartchambers highres",
    "heart chambers highres",
    "camaras cardiacas",
    "coronary",
    "coronary artery",
    "coronary arteries",
    "arteria coronaria",
    "arterias coronarias",
    "art coron",
    "myocardium",
    "heart myocardium",
    "miocardio",
    "atrium left",
    "heart atrium left",
    "left atrium",
    "atrio esquerdo",
    "atri e",
    "atrium right",
    "heart atrium right",
    "right atrium",
    "atrio direito",
    "atri d",
    "ventricle left",
    "heart ventricle left",
    "left ventricle",
    "ventriculo esquerdo",
    "ventri e",
    "ventricle right",
    "heart ventricle right",
    "right ventricle",
    "ventriculo direito",
    "ventri d",
    "pulmonary artery",
    "arteria pulmonar",
    "art pulm",
    "pulmonary vein",
    "veia pulmonar",
    "veia pulm",
}
CARDIAC_AREA_ROI_NAMES = {"area cardiaca", "cardiac area"}
LEGACY_AGATSTON_SCORE_COLUMN = "Agatson_score"
READ_ERRORS = (InvalidDicomError, OSError, EOFError, ValueError)
PIXEL_ERRORS = (AttributeError, OSError, RuntimeError, ValueError, NotImplementedError)
_VALID_VRS = {
    b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FD", b"FL", b"IS",
    b"LO", b"LT", b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"PN", b"SH",
    b"SL", b"SQ", b"SS", b"ST", b"SV", b"TM", b"UC", b"UI", b"UL", b"UN",
    b"UR", b"US", b"UT", b"UV",
}
_LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"SQ", b"UC", b"UR", b"UT", b"UN"}


@dataclass
class CTVolume:
    """CT 3D e geometria física necessária para localizar cada voxel."""

    series_uid: str
    frame_uid: str
    slices: list[Dataset]
    image: np.ndarray
    pixel_spacing: tuple[float, float]
    slice_spacing: float
    slice_spacings: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    normal: np.ndarray
    origins: np.ndarray
    slice_positions: np.ndarray


@dataclass
class RTDoseVolume:
    """Grade 3D de dose em Gy, com sua posição e orientação DICOM."""

    path: Path
    dose_gy: np.ndarray
    origin: np.ndarray
    row_direction: np.ndarray
    column_direction: np.ndarray
    normal: np.ndarray
    pixel_spacing: tuple[float, float]
    frame_offsets: np.ndarray
    frame_uid: str
    summation_type: str
    frame_mismatch: bool = False


def interpretar_agatson(score):
    if score == 0:
        return "🟢 Nenhum risco detectável", "green"
    if 1 <= score <= 100:
        return "🟡 Risco leve", "yellow"
    if 101 <= score <= 400:
        return "🟠 Risco moderado", "orange"
    if score > 400:
        return "🔴 Risco alto", "red"
    return "N/A", "white"


def _sop_class_uid(dataset):
    uid = getattr(dataset, "SOPClassUID", None)
    if not uid:
        uid = getattr(getattr(dataset, "file_meta", None), "MediaStorageSOPClassUID", "")
    return str(uid)


def _vector(dataset, name, length):
    try:
        values = np.asarray(getattr(dataset, name), dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Metadado DICOM {name} ausente ou inválido.") from error
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"Metadado DICOM {name} ausente ou inválido.")
    return values


def _prepare_slice_geometry(slices):
    # O produto vetorial dá a normal aos cortes. Projetar cada origem nessa
    # direção permite ordená-los no espaço do paciente, não pelo nome do arquivo.
    if not slices:
        raise ValueError("A série não contém imagens CT.")

    orientation = _vector(slices[0], "ImageOrientationPatient", 6)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    if (not np.isclose(np.linalg.norm(row_direction), 1, atol=1e-4)
            or not np.isclose(np.linalg.norm(column_direction), 1, atol=1e-4)
            or not np.isclose(np.dot(row_direction, column_direction), 0, atol=1e-4)):
        raise ValueError("ImageOrientationPatient inválido na série CT.")

    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)
    pixel_spacing = _vector(slices[0], "PixelSpacing", 2)
    if np.any(pixel_spacing <= 0):
        raise ValueError("PixelSpacing deve conter valores positivos.")
    rows = int(getattr(slices[0], "Rows", 0))
    columns = int(getattr(slices[0], "Columns", 0))
    if rows <= 0 or columns <= 0:
        raise ValueError("Dimensões Rows/Columns inválidas na série CT.")

    frame_uids = {
        str(getattr(image, "FrameOfReferenceUID", ""))
        for image in slices
        if getattr(image, "FrameOfReferenceUID", "")
    }
    if len(frame_uids) > 1:
        raise ValueError("A série contém imagens de Frames of Reference diferentes.")
    positions = []
    origins = []
    for image in slices:
        current_orientation = _vector(image, "ImageOrientationPatient", 6)
        current_spacing = _vector(image, "PixelSpacing", 2)
        origin = _vector(image, "ImagePositionPatient", 3)
        if not np.allclose(current_orientation, orientation, rtol=0, atol=1e-4):
            raise ValueError("A série contém orientações de imagem diferentes.")
        if not np.allclose(current_spacing, pixel_spacing, rtol=0, atol=1e-5):
            raise ValueError("A série contém valores PixelSpacing diferentes.")
        if int(getattr(image, "Rows", 0)) != rows or int(getattr(image, "Columns", 0)) != columns:
            raise ValueError("A série contém imagens com dimensões diferentes.")
        origins.append(origin)
        positions.append(float(np.dot(normal, origin)))

    order = np.argsort(positions)
    sorted_positions = np.asarray(positions, dtype=float)[order]
    if len(sorted_positions) > 1:
        differences = np.diff(sorted_positions)
        if np.any(differences <= 1e-4):
            raise ValueError("A série contém cortes duplicados ou sem espaçamento.")
        slice_spacing = float(np.median(differences))
        slice_spacings = np.empty(len(sorted_positions), dtype=float)
        slice_spacings[0] = differences[0]
        slice_spacings[-1] = differences[-1]
        if len(sorted_positions) > 2:
            slice_spacings[1:-1] = (differences[:-1] + differences[1:]) / 2
    else:
        slice_spacing = 0.0
        try:
            spacing_value = getattr(slices[0], "SpacingBetweenSlices", None)
            if spacing_value is not None:
                slice_spacing = abs(float(spacing_value))
        except (TypeError, ValueError):
            pass
        if not np.isfinite(slice_spacing) or slice_spacing <= 0:
            try:
                slice_spacing = float(getattr(slices[0], "SliceThickness"))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("Não foi possível determinar o espaçamento entre cortes.") from error
        if not np.isfinite(slice_spacing) or slice_spacing <= 0:
            raise ValueError("O espaçamento entre cortes deve ser positivo.")
        slice_spacings = np.asarray([slice_spacing], dtype=float)

    sorted_slices = [slices[index] for index in order]
    sorted_origins = np.asarray(origins, dtype=float)[order]
    return (
        sorted_slices,
        row_direction,
        column_direction,
        normal,
        pixel_spacing,
        sorted_origins,
        sorted_positions,
        slice_spacing,
        slice_spacings,
        (rows, columns),
    )


def scan_ct_series(folder):
    # Primeiro lemos só os cabeçalhos: assim identificamos séries sem carregar
    # os pixels de todas as imagens na memória.
    groups = defaultdict(list)
    notices = []
    ignored_count = 0
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file():
            continue
        # Windows marks downloaded files with an NTFS alternate data stream.
        # On WSL mounts, Path.iterdir() can expose it as a sibling filename.
        if path.name.casefold().endswith(":zone.identifier"):
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except READ_ERRORS as error:
            ignored_count += 1
            notices.append(f"{path.name}: não é um DICOM legível ({error}).")
            continue

        sop_uid = _sop_class_uid(dataset)
        modality = str(getattr(dataset, "Modality", "")).upper()
        if sop_uid == RTSTRUCT_UID or modality == "RTSTRUCT":
            ignored_count += 1
            continue
        has_classic_geometry = all(
            hasattr(dataset, attribute)
            for attribute in (
                "Rows",
                "Columns",
                "PixelSpacing",
                "ImageOrientationPatient",
                "ImagePositionPatient",
            )
        )
        if not ((modality == "CT" and has_classic_geometry)
                or (sop_uid == CT_IMAGE_UID and has_classic_geometry)):
            ignored_count += 1
            if modality == "CT":
                notices.append(
                    f"{path.name}: arquivo marcado como CT, mas sem metadados "
                    "de imagem/posição necessários."
                )
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not series_uid:
            ignored_count += 1
            notices.append(f"{path.name}: imagem CT sem SeriesInstanceUID.")
            continue
        dataset.filename = str(path)
        groups[series_uid].append(dataset)

    valid_groups = {}
    for series_uid, datasets in groups.items():
        try:
            ordered, *_ = _prepare_slice_geometry(datasets)
        except ValueError as error:
            notices.append(f"Série {series_uid}: {error}")
            continue
        valid_groups[series_uid] = ordered
    return valid_groups, ignored_count, notices


def load_ct_volume(series_uid, header_datasets):
    # CT armazenada pode exigir a transformação HU = pixel * slope + intercept.
    decoded = []
    failures = []
    for header in header_datasets:
        path = Path(header.filename)
        try:
            dataset = pydicom.dcmread(path)
            pixels = np.asarray(dataset.pixel_array)
            if pixels.ndim != 2:
                raise ValueError("a imagem não contém um único plano 2D.")
            rows, columns = int(dataset.Rows), int(dataset.Columns)
            if pixels.shape != (rows, columns):
                raise ValueError("as dimensões dos pixels não correspondem ao cabeçalho.")
            slope = float(getattr(dataset, "RescaleSlope", 1))
            intercept = float(getattr(dataset, "RescaleIntercept", 0))
            if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
                raise ValueError("RescaleSlope/RescaleIntercept inválidos.")
            decoded.append((dataset, pixels.astype(np.float32) * slope + intercept))
        except (*READ_ERRORS, *PIXEL_ERRORS) as error:
            failures.append(f"{path.name}: {error}")

    if not decoded:
        details = "\n".join(failures[:5])
        raise ValueError("Não foi possível decodificar nenhuma imagem da série CT."
                         + (f"\n{details}" if details else ""))
    if failures:
        remaining = "\n".join(failures[:5])
        raise ValueError("Há imagens CT inválidas na série selecionada; a série foi recusada "
                         "para evitar lacunas na geometria.\n" + remaining)

    datasets = [item[0] for item in decoded]
    geometry = _prepare_slice_geometry(datasets)
    (ordered, row_direction, column_direction, normal, pixel_spacing,
     origins, slice_positions, slice_spacing, slice_spacings, _) = geometry
    image_by_path = {str(dataset.filename): pixels for dataset, pixels in decoded}
    image = np.stack([image_by_path[str(dataset.filename)] for dataset in ordered], axis=-1)
    frame_uid = str(getattr(ordered[0], "FrameOfReferenceUID", ""))
    if not frame_uid:
        frame_uid = next(
            (str(getattr(dataset, "FrameOfReferenceUID"))
             for dataset in ordered
             if getattr(dataset, "FrameOfReferenceUID", "")),
            "",
        )
    return CTVolume(
        series_uid=series_uid,
        frame_uid=frame_uid,
        slices=ordered,
        image=image,
        pixel_spacing=(float(pixel_spacing[0]), float(pixel_spacing[1])),
        slice_spacing=slice_spacing,
        slice_spacings=slice_spacings,
        row_direction=row_direction,
        column_direction=column_direction,
        normal=normal,
        origins=origins,
        slice_positions=slice_positions,
    )


def _referenced_series_by_frame(dataset):
    references = defaultdict(set)
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        frame_uid = str(getattr(frame, "FrameOfReferenceUID", ""))
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                uid = getattr(series, "SeriesInstanceUID", "")
                if uid:
                    references[frame_uid].add(str(uid))
    return references


def _rtstruct_compatibility_error(dataset, volume):
    roi_sequence = getattr(dataset, "StructureSetROISequence", None)
    if not roi_sequence:
        return "não contém StructureSetROISequence."
    roi_numbers = {
        str(getattr(roi, "ROINumber", ""))
        for roi in roi_sequence
        if getattr(roi, "ROINumber", "")
    }
    has_contours = any(
        str(getattr(roi_contour, "ReferencedROINumber", "")) in roi_numbers
        and bool(getattr(roi_contour, "ContourSequence", None))
        for roi_contour in getattr(dataset, "ROIContourSequence", [])
    )
    if not has_contours:
        return "não contém contornos de ROI utilizáveis."
    references = _referenced_series_by_frame(dataset)
    if volume.frame_uid:
        if volume.frame_uid not in references:
            return "não referencia o Frame of Reference da série CT selecionada."
        if volume.series_uid not in references[volume.frame_uid]:
            return "não referencia a série CT selecionada nesse Frame of Reference."
    elif not any(volume.series_uid in series_uids for series_uids in references.values()):
        return "não referencia a série CT selecionada."
    return None


def scan_rtstruct_paths(paths, volume):
    valid = []
    notices = []
    ignored_count = 0
    for path in sorted((Path(path) for path in paths), key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        if path.name.casefold().endswith(":zone.identifier"):
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except READ_ERRORS as error:
            ignored_count += 1
            notices.append(f"{path.name}: arquivo DICOM inválido ({error}).")
            continue
        if _sop_class_uid(dataset) != RTSTRUCT_UID:
            ignored_count += 1
            continue
        error = _rtstruct_compatibility_error(dataset, volume)
        if error:
            notices.append(f"{path.name}: {error}")
            continue
        dataset.filename = str(path)
        valid.append(dataset)
    return valid, ignored_count, notices


def scan_rtstructs(folder, volume):
    return scan_rtstruct_paths(Path(folder).iterdir(), volume)


def _read_dicom_without_transfer_syntax(path):
    with path.open("rb") as stream:
        initial = stream.read(132)
        offset = 132 if len(initial) >= 132 and initial[128:132] == b"DICM" else 0
        stream.seek(offset)
        while True:
            element_start = stream.tell()
            header = stream.read(8)
            if len(header) < 8:
                raise ValueError(f"{path} não contém um cabeçalho DICOM válido.")
            group = struct.unpack("<H", header[:2])[0]
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
        header = stream.read(8)
        if len(header) < 8:
            raise ValueError(f"{path} não contém elementos de dataset DICOM.")

        little_group = struct.unpack("<H", header[:2])[0]
        big_group = struct.unpack(">H", header[:2])[0]
        common_groups = {0x0008, 0x0010, 0x0018, 0x0020, 0x0028, 0x3004, 0x7FE0}
        if little_group in common_groups:
            is_little_endian = True
        elif big_group in common_groups:
            is_little_endian = False
        else:
            is_little_endian = True
        is_explicit_vr = header[4:6] in _VALID_VRS
        if is_explicit_vr and not is_little_endian:
            transfer_syntax = ExplicitVRBigEndian
        elif is_explicit_vr:
            transfer_syntax = ExplicitVRLittleEndian
        else:
            transfer_syntax = ImplicitVRLittleEndian

        stream.seek(dataset_offset)
        dataset = read_dataset(
            stream,
            is_implicit_VR=transfer_syntax == ImplicitVRLittleEndian,
            is_little_endian=transfer_syntax != ExplicitVRBigEndian,
        )
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = transfer_syntax
    return dataset


def _read_rt_dose_dataset(path):
    try:
        dataset = pydicom.dcmread(path)
    except InvalidDicomError:
        dataset = pydicom.dcmread(path, force=True)
    file_meta = getattr(dataset, "file_meta", None)
    if not getattr(file_meta, "TransferSyntaxUID", None):
        dataset = _read_dicom_without_transfer_syntax(path)
    return dataset


def load_rt_dose(path, volume):
    path = Path(path)
    dataset = _read_rt_dose_dataset(path)
    if _sop_class_uid(dataset) != RTDOSE_UID and str(
        getattr(dataset, "Modality", "")
    ).upper() != "RTDOSE":
        raise ValueError(f"O arquivo selecionado não é um RTDOSE DICOM: {path}")

    frame_uid = str(getattr(dataset, "FrameOfReferenceUID", ""))
    if not frame_uid or not volume.frame_uid:
        raise ValueError("O RTDOSE e a série CT precisam informar o Frame of Reference.")
    frame_mismatch = frame_uid != volume.frame_uid

    units = str(getattr(dataset, "DoseUnits", "")).upper()
    if units not in {"GY", "CGY"}:
        raise ValueError(
            f"A unidade de dose '{units or 'ausente'}' não é absoluta (GY/CGY)."
        )
    dose_type = str(getattr(dataset, "DoseType", "")).upper()
    if dose_type and dose_type != "PHYSICAL":
        raise ValueError(
            f"O RTDOSE contém DoseType '{dose_type}', não dose física."
        )

    pixels = np.asarray(dataset.pixel_array, dtype=np.float64)
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    if pixels.ndim != 3:
        raise ValueError("O RTDOSE precisa conter uma grade 3D de dose.")
    try:
        scale = float(getattr(dataset, "DoseGridScaling", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("O RTDOSE não possui DoseGridScaling válido.") from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("O RTDOSE não possui DoseGridScaling válido.")
    # DoseGridScaling recupera a dose armazenada; cGy é convertido para Gy.
    dose_gy = pixels * scale * (0.01 if units == "CGY" else 1.0)
    if not np.isfinite(dose_gy).all():
        raise ValueError("A grade RTDOSE contém valores de dose inválidos.")

    orientation = _vector(dataset, "ImageOrientationPatient", 6)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    if (not np.isclose(np.linalg.norm(row_direction), 1, atol=1e-4)
            or not np.isclose(np.linalg.norm(column_direction), 1, atol=1e-4)
            or not np.isclose(np.dot(row_direction, column_direction), 0, atol=1e-4)):
        raise ValueError("ImageOrientationPatient inválido no RTDOSE.")
    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)
    origin = _vector(dataset, "ImagePositionPatient", 3)
    pixel_spacing = _vector(dataset, "PixelSpacing", 2)
    if np.any(pixel_spacing <= 0):
        raise ValueError("PixelSpacing deve conter valores positivos no RTDOSE.")
    try:
        offsets = np.asarray(
            getattr(dataset, "GridFrameOffsetVector", [0.0]), dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError("GridFrameOffsetVector do RTDOSE é inválido.") from error
    if offsets.shape != (dose_gy.shape[0],) or not np.isfinite(offsets).all():
        raise ValueError("GridFrameOffsetVector não corresponde às dimensões do RTDOSE.")
    if (
        np.allclose(row_direction, [1, 0, 0], atol=1e-5)
        and np.allclose(column_direction, [0, 1, 0], atol=1e-5)
        and np.isclose(offsets[0], origin[2], atol=1e-5)
    ):
        offsets = offsets - float(np.dot(origin, normal))
    order = np.argsort(offsets)
    offsets = offsets[order]
    dose_gy = dose_gy[order, :, :]
    if len(offsets) > 1 and np.any(np.diff(offsets) <= 0):
        raise ValueError("O RTDOSE possui posições de quadros inválidas.")

    return RTDoseVolume(
        path=path,
        dose_gy=dose_gy,
        origin=origin,
        row_direction=row_direction,
        column_direction=column_direction,
        normal=normal,
        pixel_spacing=(float(pixel_spacing[0]), float(pixel_spacing[1])),
        frame_offsets=offsets,
        frame_uid=frame_uid,
        summation_type=str(getattr(dataset, "DoseSummationType", "Não informado")),
        frame_mismatch=frame_mismatch,
    )


def _sample_rt_dose_on_mask(mask, volume, dose_grid, roi_name):
    mean_dose, d0_5cc, _ = _sample_rt_dose_metrics(
        mask, volume, dose_grid, roi_name
    )
    return mean_dose, d0_5cc


def _sample_dose_at_patient_points(points, dose_grid):
    # Converte coordenadas físicas (mm) para índices da grade RTDOSE.
    # A interpolação trilinear combina os 8 voxels vizinhos.
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    relative = points - dose_grid.origin
    dose_columns = relative @ dose_grid.row_direction / dose_grid.pixel_spacing[1]
    dose_rows = relative @ dose_grid.column_direction / dose_grid.pixel_spacing[0]
    frame_positions = relative @ dose_grid.normal

    frame_count, dose_row_count, dose_column_count = dose_grid.dose_gy.shape
    tolerance = 1e-5
    valid = (
        (dose_columns >= -tolerance)
        & (dose_columns <= dose_column_count - 1 + tolerance)
        & (dose_rows >= -tolerance)
        & (dose_rows <= dose_row_count - 1 + tolerance)
    )
    offsets = dose_grid.frame_offsets
    if frame_count == 1:
        valid &= np.isclose(frame_positions, offsets[0], atol=tolerance)
        frame_coordinates = np.zeros(len(points), dtype=float)
    else:
        valid &= (
            (frame_positions >= offsets[0] - tolerance)
            & (frame_positions <= offsets[-1] + tolerance)
        )
        frame_coordinates = np.interp(
            frame_positions, offsets, np.arange(frame_count, dtype=float)
        )

    samples = np.full(len(points), np.nan, dtype=float)
    if not np.any(valid):
        return samples, valid

    dose_columns = np.clip(dose_columns, 0, dose_column_count - 1)
    dose_rows = np.clip(dose_rows, 0, dose_row_count - 1)
    frame_coordinates = np.clip(frame_coordinates, 0, frame_count - 1)
    lower_columns = np.floor(dose_columns).astype(int)
    lower_rows = np.floor(dose_rows).astype(int)
    lower_frames = np.floor(frame_coordinates).astype(int)
    upper_columns = np.minimum(lower_columns + 1, dose_column_count - 1)
    upper_rows = np.minimum(lower_rows + 1, dose_row_count - 1)
    upper_frames = np.minimum(lower_frames + 1, frame_count - 1)
    column_fraction = dose_columns - lower_columns
    row_fraction = dose_rows - lower_rows
    frame_fraction = frame_coordinates - lower_frames

    interpolated = np.zeros(len(points), dtype=float)
    for frame_index, frame_weight in (
        (lower_frames, 1 - frame_fraction),
        (upper_frames, frame_fraction),
    ):
        for row_index, row_weight in (
            (lower_rows, 1 - row_fraction),
            (upper_rows, row_fraction),
        ):
            for column_index, column_weight in (
                (lower_columns, 1 - column_fraction),
                (upper_columns, column_fraction),
            ):
                interpolated += (
                    dose_grid.dose_gy[frame_index, row_index, column_index]
                    * frame_weight
                    * row_weight
                    * column_weight
                )
    samples[valid] = interpolated[valid]
    return samples, valid


def _sample_rt_dose_metrics(mask, volume, dose_grid, roi_name):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != volume.image.shape:
        raise ValueError("A máscara da estrutura não coincide com a grade CT.")
    rows, columns, slices = np.nonzero(mask)
    if not len(rows):
        return np.nan, np.nan, {8: np.nan, 16: np.nan, 20: np.nan}

    ct_points = (
        volume.origins[slices]
        + (columns * volume.pixel_spacing[1])[:, None] * volume.row_direction
        + (rows * volume.pixel_spacing[0])[:, None] * volume.column_direction
    )
    samples, valid = _sample_dose_at_patient_points(ct_points, dose_grid)
    coverage = float(np.count_nonzero(valid)) / len(rows)
    if not np.all(valid):
        raise ValueError(
            f"O RTDOSE cobre {coverage:.1%} dos voxels da estrutura "
            f"'{roi_name}'; é necessária cobertura integral para calcular "
            "a dose média e D0,5cc."
        )

    voxel_volumes_cc = (
        volume.pixel_spacing[0]
        * volume.pixel_spacing[1]
        * volume.slice_spacings[slices]
        / 1000.0
    )
    total_volume_cc = float(np.sum(voxel_volumes_cc))
    mean_dose = float(np.average(samples, weights=voxel_volumes_cc))
    if total_volume_cc < 0.5:
        d0_5cc = np.nan
    else:
        hottest_order = np.argsort(samples)[::-1]
        hottest_doses = samples[hottest_order]
        cumulative_volume = np.cumsum(voxel_volumes_cc[hottest_order])
        threshold_index = min(
            int(np.searchsorted(cumulative_volume, 0.5, side="left")),
            len(hottest_doses) - 1,
        )
        d0_5cc = float(hottest_doses[threshold_index])
    volume_percentages = {
        threshold: float(
            np.sum(voxel_volumes_cc[samples >= threshold]) / total_volume_cc * 100
        )
        for threshold in (8, 16, 20)
    }
    return mean_dose, d0_5cc, volume_percentages


def _normalize_roi_name(name):
    normalized = unicodedata.normalize("NFKD", str(name).casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _breast_laterality(name):
    normalized = _normalized_roi_key(name)
    if "mama" not in normalized and "breast" not in normalized:
        return None
    tokens = normalized.split()
    if any(token in {"e", "esq", "esquerda", "esquerdo", "left", "l"} for token in tokens):
        return "Esquerda"
    if any(token in {"d", "dir", "direita", "direito", "right", "r"} for token in tokens):
        return "Direita"
    return None


def _treatment_laterality(dataframe):
    if "treatment_laterality" in dataframe.attrs:
        return dataframe.attrs["treatment_laterality"]
    if DOSE_MEAN_COLUMN not in dataframe:
        return None
    candidates = []
    for _, row in dataframe.iterrows():
        side = _breast_laterality(row["ROI"])
        dose = pd.to_numeric(row[DOSE_MEAN_COLUMN], errors="coerce")
        if side and pd.notna(dose) and np.isfinite(dose):
            candidates.append((float(dose), side))
    if not candidates:
        return None
    highest_dose = max(dose for dose, _ in candidates)
    sides = {
        side for dose, side in candidates
        if np.isclose(dose, highest_dose, rtol=0, atol=1e-6)
    }
    return next(iter(sides)) if len(sides) == 1 else None


def _heart_constraint_results(dataframe):
    cardiac_rows = dataframe[
        dataframe["ROI"].map(_is_cardiac_area_roi)
    ]
    cardiac_row = cardiac_rows.iloc[0] if not cardiac_rows.empty else None
    laterality = _treatment_laterality(dataframe)
    rows = []
    constraint_specs = (
        (1, "Ideal", 16, {"Esquerda": 5.0, "Direita": 0.0}),
        (1, "Aceitável", 20, {"Esquerda": 5.0, "Direita": 0.0}),
        (2, "Ideal", 8, {"Esquerda": 30.0, "Direita": 10.0}),
        (2, "Aceitável", 8, {"Esquerda": 35.0, "Direita": 15.0}),
    )
    dose_columns = {
        8: DOSE_V8_COLUMN,
        16: DOSE_V16_COLUMN,
        20: DOSE_V20_COLUMN,
    }
    for constraint, classification, dose_threshold, limits in constraint_specs:
        column = dose_columns[dose_threshold]
        limit = limits.get(laterality, np.nan)
        measured = (
            float(cardiac_row[column])
            if cardiac_row is not None
            and column in cardiac_row.index
            and pd.notna(cardiac_row[column])
            else np.nan
        )
        if np.isnan(measured) or np.isnan(limit):
            status = "NÃO AVALIADO"
        else:
            status = "PASSA" if measured <= limit else "NÃO PASSA"
        rows.append({
            "Constraint": f"Heart dose constraint {constraint}",
            "Classificação": classification,
            "Lateralidade": laterality or "Não identificada",
            "Métrica": f"V{dose_threshold} Gy",
            "Limite (%)": limit,
            "Medido (%)": measured,
            "Status": status,
        })
    return rows


def _choose_rt_dose_file(root, initial_folder, volume):
    while True:
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="Selecione o arquivo DICOM RTDOSE",
            initialdir=str(initial_folder),
            filetypes=[
                ("Arquivos DICOM", "*.dcm"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not selected_path:
            return None
        try:
            return load_rt_dose(selected_path, volume)
        except (*READ_ERRORS, *PIXEL_ERRORS) as error:
            messagebox.showerror(
                "Erro ao ler o RTDOSE",
                str(error),
                parent=root,
            )


def _center_window(window, width, height):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.minsize(width, height)


def _choose_ct_folder(root, initial_folder):
    return filedialog.askdirectory(
        parent=root,
        title="Selecione a pasta com as imagens CT DICOM",
        initialdir=str(initial_folder),
        mustexist=True,
    )


def _choose_ct_series(root, series):
    if len(series) == 1:
        return next(iter(series))
    window = tk.Toplevel(root)
    window.title("Escolha a série CT")
    window.transient(root)
    window.grab_set()
    tk.Label(
        window,
        text="A pasta contém mais de uma série CT. Escolha uma; séries não serão combinadas.",
        wraplength=540,
        justify="left",
    ).pack(fill="x", padx=12, pady=(12, 6))
    listbox = tk.Listbox(window, height=min(12, len(series)), exportselection=False)
    series_uids = list(series)
    for uid in series_uids:
        datasets = series[uid]
        first = datasets[0]
        description = str(getattr(first, "SeriesDescription", "Sem descrição"))
        date = str(getattr(first, "SeriesDate", ""))
        listbox.insert("end", f"{description} | {date} | {len(datasets)} cortes | {uid}")
    listbox.selection_set(0)
    listbox.pack(fill="both", expand=True, padx=12, pady=6)
    result = {"uid": None}

    def accept(_event=None):
        selection = listbox.curselection()
        if selection:
            result["uid"] = series_uids[selection[0]]
            window.destroy()

    buttons = tk.Frame(window)
    buttons.pack(fill="x", padx=12, pady=(4, 12))
    ttk.Button(buttons, text="Selecionar", command=accept).pack(side="right", padx=(6, 0))
    ttk.Button(buttons, text="Cancelar", command=window.destroy).pack(side="right")
    listbox.bind("<Double-Button-1>", accept)
    window.protocol("WM_DELETE_WINDOW", window.destroy)
    _center_window(window, 600, 360)
    root.wait_window(window)
    return result["uid"]


def _choose_rtstruct_files(root, initial_folder, volume):
    while True:
        selected_paths = filedialog.askopenfilenames(
            parent=root,
            title="Selecione um ou mais RTSTRUCTs (Ctrl/Shift para múltiplos)",
            initialdir=str(initial_folder),
            filetypes=[
                ("Arquivos DICOM", "*.dcm"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not selected_paths:
            return None
        try:
            rtstructs, ignored_count, notices = scan_rtstruct_paths(selected_paths, volume)
        except OSError as error:
            messagebox.showerror(
                "Erro ao ler os arquivos RTSTRUCT",
                str(error),
                parent=root,
            )
            continue
        if rtstructs and not notices and ignored_count == 0:
            return rtstructs
        details = "\n".join(notices[:10]) or "A seleção não contém RTSTRUCT válido."
        if ignored_count:
            details += f"\n{ignored_count} arquivo(s) selecionado(s) não é(são) RTSTRUCT."
        messagebox.showerror(
            "Seleção de RTSTRUCT inválida",
            details + ("\n..." if len(notices) > 10 else ""),
            parent=root,
        )


def merge_rtstructs(rtstructs):
    # Junta contornos de ROIs com o mesmo nome em diferentes RTSTRUCTs.
    structures = {}
    notices = []
    for dataset in rtstructs:
        path = Path(dataset.filename)
        roi_names = {
            str(roi.ROINumber): str(roi.ROIName)
            for roi in getattr(dataset, "StructureSetROISequence", [])
        }
        for roi_contour in getattr(dataset, "ROIContourSequence", []):
            roi_name = roi_names.get(str(getattr(roi_contour, "ReferencedROINumber", "")))
            if not roi_name:
                notices.append(f"{path.name}: sequência de contorno referencia uma ROI inexistente.")
                continue
            contours = list(getattr(roi_contour, "ContourSequence", []))
            if contours:
                roi_name = ROI_TRANSLATIONS.get(roi_name.casefold().strip(), roi_name)
                structures.setdefault(roi_name, []).extend(contours)
    return structures, notices


def get_mask_from_contours(contours, volume):
    # Converte vértices dos contornos, em mm, para pixels da CT e preenche
    # o polígono no corte correspondente, formando uma máscara booleana 3D.
    rows, columns, slices = volume.image.shape
    mask = np.zeros((rows, columns, slices), dtype=bool)
    invalid_count = 0
    outside_count = 0
    tolerance = max(volume.slice_spacing * 0.5, 0.5)
    for contour in contours:
        geometric_type = str(getattr(contour, "ContourGeometricType", "")).upper()
        if geometric_type not in {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"}:
            invalid_count += 1
            continue
        try:
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        except (AttributeError, TypeError, ValueError):
            invalid_count += 1
            continue
        if len(points) < 3 or not np.isfinite(points).all():
            invalid_count += 1
            continue
        declared_count = getattr(contour, "NumberOfContourPoints", len(points))
        try:
            if int(declared_count) != len(points):
                invalid_count += 1
                continue
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        projected = points @ volume.normal
        if float(np.ptp(projected)) > 0.5:
            invalid_count += 1
            continue
        slice_position = float(np.mean(projected))
        slice_index = int(np.argmin(np.abs(volume.slice_positions - slice_position)))
        if abs(float(volume.slice_positions[slice_index]) - slice_position) > tolerance:
            outside_count += 1
            continue

        origin = volume.origins[slice_index]
        relative = points - origin
        column_coordinates = (relative @ volume.row_direction) / volume.pixel_spacing[1]
        row_coordinates = (relative @ volume.column_direction) / volume.pixel_spacing[0]
        rr, cc = draw.polygon(row_coordinates, column_coordinates, shape=(rows, columns))
        mask[rr, cc, slice_index] = True
    return mask, invalid_count, outside_count


def find_clusters(matrix, threshold=129, min_size=4):
    # Rotula componentes 2D: conectividade 8 inclui diagonais; rótulo zero é
    # o fundo. min_size elimina agrupamentos menores que o limite configurado.
    active = np.isfinite(matrix) & (matrix > threshold)
    labels = measure.label(active, connectivity=2, background=0)
    component_sizes = np.bincount(labels.ravel())
    valid_labels = np.flatnonzero(component_sizes >= min_size)
    valid_labels = valid_labels[valid_labels != 0]
    if not len(valid_labels):
        return []

    coordinates = np.argwhere(labels)
    component_labels = labels[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(component_labels, kind="stable")
    coordinates = coordinates[order]
    component_labels = component_labels[order]
    group_starts = np.searchsorted(component_labels, valid_labels)
    return [
        [
            (int(row), int(column))
            for row, column in coordinates[
                start:start + component_sizes[label]
            ]
        ]
        for label, start in zip(valid_labels, group_starts)
    ]


def agatson_score(filtered_array, cluster, volume, slice_index):
    # O escore usa área (mm²) vezes um fator definido pela HU máxima.
    # O volume (mm³) é calculado separadamente para ser exibido no resultado.
    pixel_area = volume.pixel_spacing[0] * volume.pixel_spacing[1]
    coordinates = np.asarray(cluster, dtype=int)
    values = filtered_array[coordinates[:, 0], coordinates[:, 1]]
    maximum = float(np.max(values))
    area_score = len(coordinates) * pixel_area
    volume_score = area_score * volume.slice_spacings[slice_index]
    if 130 <= maximum <= 199:
        weighted_score = area_score
    elif 200 <= maximum <= 299:
        weighted_score = 2 * area_score
    elif 300 <= maximum <= 399:
        weighted_score = 3 * area_score
    elif maximum >= 400:
        weighted_score = 4 * area_score
    else:
        weighted_score = 0
    return round(weighted_score, 1), round(volume_score, 1)


def _normalized_roi_key(name):
    normalized = _normalize_roi_name(name)
    return " ".join(normalized.replace("_", " ").replace("-", " ").split())


def _is_cardiac_roi(name):
    return _normalized_roi_key(name) in CARDIAC_ROI_NAMES


def _is_cardiac_area_roi(name):
    return _normalized_roi_key(name) in CARDIAC_AREA_ROI_NAMES


def _infer_treatment_laterality(structures, volume, dose_grid):
    if dose_grid is None:
        return None
    highest_dose_by_side = {}
    for roi_name, contours in structures.items():
        side = _breast_laterality(roi_name)
        if side is None:
            continue
        mask, _, _ = get_mask_from_contours(contours, volume)
        rows, columns, slices = np.nonzero(mask)
        if not len(rows):
            continue
        patient_points = (
            volume.origins[slices]
            + (columns * volume.pixel_spacing[1])[:, None]
            * volume.row_direction
            + (rows * volume.pixel_spacing[0])[:, None]
            * volume.column_direction
        )
        samples, valid = _sample_dose_at_patient_points(patient_points, dose_grid)
        if not np.all(valid):
            continue
        voxel_volumes_cc = (
            volume.pixel_spacing[0]
            * volume.pixel_spacing[1]
            * volume.slice_spacings[slices]
            / 1000.0
        )
        mean_dose = float(np.average(samples, weights=voxel_volumes_cc))
        highest_dose_by_side[side] = max(
            mean_dose,
            highest_dose_by_side.get(side, -np.inf),
        )
    if not highest_dose_by_side:
        return None
    highest_dose = max(highest_dose_by_side.values())
    highest_sides = [
        side
        for side, dose in highest_dose_by_side.items()
        if np.isclose(dose, highest_dose, rtol=0, atol=1e-6)
    ]
    return highest_sides[0] if len(highest_sides) == 1 else None


def calculate_results(structures, volume, dose_grid=None):
    # Só ROIs cardíacas entram no Agatston; a dose, se fornecida, é calculada
    # na mesma máscara e comparada com os centros dos voxels CT.
    results = []
    invalid_contours = 0
    outside_contours = 0
    cardiac_structures = {
        name: contours
        for name, contours in structures.items()
        if _is_cardiac_roi(name)
    }
    if not cardiac_structures:
        raise ValueError(
            "Nenhuma estrutura cardíaca reconhecida foi encontrada no RTSTRUCT. "
            "Verifique os nomes das ROIs."
        )
    treatment_laterality = _infer_treatment_laterality(
        structures, volume, dose_grid
    )
    for roi_name, contours in cardiac_structures.items():
        mask, invalid, outside = get_mask_from_contours(contours, volume)
        invalid_contours += invalid
        outside_contours += outside
        if dose_grid is not None:
            mean_dose, d0_5cc, dose_volume_percentages = _sample_rt_dose_metrics(
                mask, volume, dose_grid, roi_name
            )
        total_score = 0
        total_volume = 0
        for slice_index in range(volume.image.shape[2]):
            masked_slice = np.where(mask[:, :, slice_index],
                                    volume.image[:, :, slice_index], np.nan)
            for cluster in find_clusters(masked_slice):
                score, cluster_volume = agatson_score(
                    masked_slice, cluster, volume, slice_index
                )
                total_score += score
                total_volume += cluster_volume
        risk, color = interpretar_agatson(total_score)
        result = {
            "ROI": roi_name,
            AGATSTON_SCORE_COLUMN: round(total_score, 1),
            "Volume_score": round(total_volume, 1),
            "Interpretação": risk,
            "Cor": color,
        }
        if dose_grid is not None:
            result[DOSE_MEAN_COLUMN] = mean_dose
            result[DOSE_D0_5CC_COLUMN] = d0_5cc
            result[DOSE_V8_COLUMN] = dose_volume_percentages[8]
            result[DOSE_V16_COLUMN] = dose_volume_percentages[16]
            result[DOSE_V20_COLUMN] = dose_volume_percentages[20]
        results.append(result)
    columns = ["ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação", "Cor"]
    if dose_grid is not None:
        columns.extend([
            DOSE_MEAN_COLUMN,
            DOSE_D0_5CC_COLUMN,
            DOSE_V8_COLUMN,
            DOSE_V16_COLUMN,
            DOSE_V20_COLUMN,
        ])
    dataframe = pd.DataFrame(results, columns=columns)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=AGATSTON_SCORE_COLUMN, ascending=False)
    dataframe.attrs["treatment_laterality"] = treatment_laterality
    return dataframe, invalid_contours, outside_contours


def normalize_results_dataframe(dataframe):
    normalized = dataframe.copy()
    if AGATSTON_SCORE_COLUMN not in normalized.columns:
        if LEGACY_AGATSTON_SCORE_COLUMN not in normalized.columns:
            raise ValueError(
                "Os resultados não contêm a coluna do escore Agatston."
            )
        normalized = normalized.rename(
            columns={LEGACY_AGATSTON_SCORE_COLUMN: AGATSTON_SCORE_COLUMN}
        )
    elif LEGACY_AGATSTON_SCORE_COLUMN in normalized.columns:
        normalized = normalized.drop(columns=LEGACY_AGATSTON_SCORE_COLUMN)
    return normalized


def _format_dicom_date(value):
    if not value:
        return "Não informada"
    value = str(value).strip()
    try:
        return datetime.strptime(value[:8], "%Y%m%d").strftime("%d/%m/%Y")
    except ValueError:
        return value


def _tomography_date(dataset):
    for attribute in ("StudyDate", "AcquisitionDate", "SeriesDate"):
        value = getattr(dataset, attribute, None)
        if value:
            return _format_dicom_date(value)
    return "Não informada"


def _pdf_initial_directory(volume):
    return Path(volume.slices[0].filename).parent.parent


def _select_distinct_calcifications(
    candidates,
    pixel_spacing,
    limit=3,
    minimum_distance_mm=5,
):
    row_spacing, column_spacing = pixel_spacing
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            -item["maximum_hu"],
            -len(item["cluster"]),
            item["slice_index"],
            item["center"],
        ),
    )
    selected = []
    for candidate in ordered:
        too_close = False
        for existing in selected:
            if candidate["slice_index"] != existing["slice_index"]:
                continue
            row_delta = (
                candidate["center"][0] - existing["center"][0]
            ) * row_spacing
            column_delta = (
                candidate["center"][1] - existing["center"][1]
            ) * column_spacing
            if np.hypot(row_delta, column_delta) <= minimum_distance_mm:
                too_close = True
                break
        if not too_close:
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def _top_calcifications(contours, volume, limit=3, minimum_distance_mm=5):
    structure_mask, _, _ = get_mask_from_contours(contours, volume)
    candidates = []
    for slice_index in range(volume.image.shape[2]):
        masked_slice = np.where(
            structure_mask[:, :, slice_index],
            volume.image[:, :, slice_index],
            np.nan,
        )
        for cluster in find_clusters(masked_slice):
            score, _ = agatson_score(masked_slice, cluster, volume, slice_index)
            if score <= 0:
                continue
            coordinates = np.asarray(cluster, dtype=float)
            cluster_values = masked_slice[
                coordinates[:, 0].astype(int),
                coordinates[:, 1].astype(int),
            ]
            maximum_hu = float(np.max(cluster_values))
            center_row, center_column = np.mean(coordinates, axis=0)
            candidates.append({
                "score": score,
                "maximum_hu": maximum_hu,
                "slice_index": slice_index,
                "center": (center_row, center_column),
                "cluster": cluster,
                "structure_mask": structure_mask[:, :, slice_index],
            })
    return _select_distinct_calcifications(
        candidates,
        volume.pixel_spacing,
        limit=limit,
        minimum_distance_mm=minimum_distance_mm,
    )


def _mask_boundary(mask):
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    inner = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return inner & ~interior


def _rt_dose_isodose_levels(dose_grid):
    maximum_dose = float(np.max(dose_grid.dose_gy))
    if maximum_dose < 1:
        return []
    levels = [1.0, 2.0]
    if maximum_dose >= 5:
        levels.extend(np.arange(5.0, maximum_dose + 1e-9, 5.0).tolist())
    return [level for level in levels if level <= maximum_dose]


_ISODOSE_PALETTE = (
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
)


def _isodose_color(index, count):
    palette_index = (
        round(index * (len(_ISODOSE_PALETTE) - 1) / (count - 1))
        if count > 1
        else 0
    )
    color = _ISODOSE_PALETTE[palette_index].lstrip("#")
    return tuple(int(color[offset:offset + 2], 16) for offset in (0, 2, 4))


def _render_calcification_crop(volume, finding, size_mm=120, dose_grid=None):
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "A biblioteca Pillow é necessária para incluir imagens no relatório PDF."
        ) from error

    row_spacing, column_spacing = volume.pixel_spacing
    crop_rows = max(1, int(round(size_mm / row_spacing)))
    crop_columns = max(1, int(round(size_mm / column_spacing)))
    center_row, center_column = finding["center"]
    center_row = int(round(center_row))
    center_column = int(round(center_column))
    row_start = center_row - crop_rows // 2
    column_start = center_column - crop_columns // 2
    row_end = row_start + crop_rows
    column_end = column_start + crop_columns

    image_slice = volume.image[:, :, finding["slice_index"]]
    hu_crop = np.full((crop_rows, crop_columns), -1000.0, dtype=np.float32)
    structure_crop = np.zeros((crop_rows, crop_columns), dtype=bool)
    dose_crop = np.full((crop_rows, crop_columns), np.nan, dtype=np.float32)
    source_row_start = max(0, row_start)
    source_column_start = max(0, column_start)
    source_row_end = min(image_slice.shape[0], row_end)
    source_column_end = min(image_slice.shape[1], column_end)
    if source_row_start < source_row_end and source_column_start < source_column_end:
        target_rows = slice(
            source_row_start - row_start,
            source_row_end - row_start,
        )
        target_columns = slice(
            source_column_start - column_start,
            source_column_end - column_start,
        )
        source = (
            slice(source_row_start, source_row_end),
            slice(source_column_start, source_column_end),
        )
        hu_crop[target_rows, target_columns] = image_slice[source]
        structure_crop[target_rows, target_columns] = finding["structure_mask"][source]
        if dose_grid is not None:
            row_grid, column_grid = np.meshgrid(
                np.arange(source_row_start, source_row_end),
                np.arange(source_column_start, source_column_end),
                indexing="ij",
            )
            patient_points = (
                volume.origins[finding["slice_index"]]
                + (column_grid.ravel() * volume.pixel_spacing[1])[:, None]
                * volume.row_direction
                + (row_grid.ravel() * volume.pixel_spacing[0])[:, None]
                * volume.column_direction
            )
            dose_values, _ = _sample_dose_at_patient_points(patient_points, dose_grid)
            dose_crop[target_rows, target_columns] = dose_values.reshape(
                row_grid.shape
            )

    window_low, window_high = -100.0, 300.0
    grayscale = np.clip(
        (hu_crop - window_low) * (255.0 / (window_high - window_low)),
        0,
        255,
    ).astype(np.uint8)
    rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
    rgb[_mask_boundary(structure_crop)] = (255, 40, 40)
    image = Image.fromarray(rgb, mode="RGB")
    if dose_grid is not None and np.isfinite(dose_crop).any():
        levels = _rt_dose_isodose_levels(dose_grid)
        if levels:
            line_scale = 5
            image = image.resize(
                (image.width * line_scale, image.height * line_scale),
                resample=Image.Resampling.BICUBIC,
            )
            image_draw = ImageDraw.Draw(image)
            for index, level in enumerate(levels):
                for contour in measure.find_contours(
                    dose_crop,
                    level,
                    mask=np.isfinite(dose_crop),
                ):
                    if len(contour) < 2:
                        continue
                    image_draw.line(
                        [
                            (float(column) * line_scale, float(row) * line_scale)
                            for row, column in contour
                        ],
                        fill=_isodose_color(index, len(levels)),
                        width=3,
                    )
            image = image.resize(
                (image.width // line_scale, image.height // line_scale),
                resample=Image.Resampling.LANCZOS,
            )

    output = BytesIO()
    display_columns = max(
        1,
        int(round(crop_columns * column_spacing / row_spacing)),
    )
    if display_columns != crop_columns:
        image = image.resize(
            (display_columns, crop_rows),
            resample=Image.Resampling.BILINEAR,
        )
    image.save(output, format="PNG")
    output.seek(0)
    return output


def results_as_text(dataframe, patient_name, patient_id, dose_grid=None):
    dataframe = normalize_results_dataframe(dataframe)
    output = StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["Paciente", patient_name])
    writer.writerow(["ID", patient_id])
    if dose_grid is not None:
        writer.writerow(["RTDOSE", dose_grid.path.name])
        writer.writerow(["DoseSummationType", dose_grid.summation_type])
        if dose_grid.frame_mismatch:
            writer.writerow([
                "Aviso",
                "Frames of Reference diferentes; coordenadas físicas assumidas "
                "comparáveis, sem registro espacial.",
            ])
    writer.writerow([])
    columns = ["ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação"]
    headers = ["Estrutura", "Agatston Score", "Volume (mm³)", "Interpretação"]
    if DOSE_MEAN_COLUMN in dataframe and DOSE_D0_5CC_COLUMN in dataframe:
        columns.extend([DOSE_MEAN_COLUMN, DOSE_D0_5CC_COLUMN])
        headers.extend(["Dose média (Gy)", "D0,5cc (Gy)"])
    writer.writerow(headers)
    for _, row in dataframe.iterrows():
        values = [row[column] for column in columns]
        if DOSE_MEAN_COLUMN in dataframe and DOSE_D0_5CC_COLUMN in dataframe:
            values[-2:] = [
                _format_dose_value(row[DOSE_MEAN_COLUMN]),
                _format_dose_value(row[DOSE_D0_5CC_COLUMN]),
            ]
        writer.writerow(values)
    if all(column in dataframe for column in (
        DOSE_V8_COLUMN, DOSE_V16_COLUMN, DOSE_V20_COLUMN
    )):
        writer.writerow([])
        writer.writerow(["Constraints cardíacos — RTOG 1005"])
        writer.writerow([
            "Limites de dose utilizados",
            "Arm II: V16 Gy ideal / V20 Gy aceitável no constraint 1; V8 Gy no constraint 2",
        ])
        writer.writerow([
            "Classificação",
            "Lateralidade",
            "Métrica",
            "Limite",
            "Medido",
            "Status",
        ])
        for constraint in _heart_constraint_results(dataframe):
            writer.writerow([
                constraint["Classificação"],
                constraint["Lateralidade"],
                constraint["Métrica"],
                _format_percent_value(constraint["Limite (%)"]),
                _format_percent_value(constraint["Medido (%)"]),
                constraint["Status"],
            ])
    return output.getvalue()


def _format_dose_value(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def _format_percent_value(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def open_pdf(path):
    pdf_path = Path(path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"O PDF não foi encontrado: {pdf_path}")
    if not webbrowser.open(pdf_path.as_uri()):
        raise RuntimeError(f"Não foi possível abrir o PDF: {pdf_path}")


def save_report_pdf(
    path,
    dataframe,
    patient_name,
    patient_id,
    tomography_date,
    analysis_date,
    volume=None,
    structures=None,
    dose_grid=None,
):
    # A geração do PDF usa bibliotecas opcionais, importadas nesta função.
    dataframe = normalize_results_dataframe(dataframe)
    if (volume is None) != (structures is None):
        raise ValueError("Volume CT e estruturas RTSTRUCT devem ser fornecidos juntos.")
    has_calcification_pages = (
        volume is not None
        and pd.to_numeric(
            dataframe[AGATSTON_SCORE_COLUMN], errors="coerce"
        ).gt(0).any()
    )
    isodose_levels = (
        _rt_dose_isodose_levels(dose_grid)
        if has_calcification_pages and dose_grid is not None
        else []
    )
    from xml.sax.saxutils import escape

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.platypus import (
            Flowable,
            Image as ReportImage,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table as ReportTable,
            TableStyle,
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("reportlab"):
            raise RuntimeError(
                "Para exportar relatórios em PDF, instale a dependência reportlab "
                "com o comando: python -m pip install reportlab"
            ) from error
        raise

    page_size = A4
    report_title = (
        "Relatório de Agatston e dose de radioterapia"
        if dose_grid is not None
        else "Resultado da análise de escore de Agatston"
    )
    document = SimpleDocTemplate(
        str(path),
        pagesize=page_size,
        rightMargin=11 * mm,
        leftMargin=11 * mm,
        topMargin=12 * mm,
        bottomMargin=37 * mm if has_calcification_pages else 23 * mm,
        title=report_title,
        author="CAC",
    )
    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Title"],
        alignment=TA_CENTER,
        fontSize=16,
        leading=20,
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "ReportSection",
        parent=base_styles["Heading2"],
        fontSize=11,
        leading=14,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=base_styles["BodyText"],
        fontSize=9,
        leading=12,
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=body_style,
        fontSize=8,
        leading=10,
    )
    header_style = ParagraphStyle(
        "ReportHeader",
        parent=cell_style,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    story = [
        Paragraph(escape(report_title), title_style),
        Paragraph(f"<b>Paciente:</b> {escape(patient_name)}", body_style),
        Paragraph(f"<b>ID:</b> {escape(patient_id)}", body_style),
        Paragraph(f"<b>Data da tomografia:</b> {escape(tomography_date)}", body_style),
        Paragraph(f"<b>Data da análise:</b> {escape(analysis_date)}", body_style),
        Spacer(1, 4 * mm),
        Paragraph("Resultados", section_style),
    ]
    headers = ["Estrutura", "Agatston Score", "Volume (mm³)", "Interpretação"]
    has_dose_metrics = (
        DOSE_MEAN_COLUMN in dataframe and DOSE_D0_5CC_COLUMN in dataframe
    )
    if has_dose_metrics:
        headers.extend(["Dose média (Gy)", "D0,5cc (Gy)"])
    table_data = [[Paragraph(escape(value), header_style) for value in headers]]
    risk_colors = {
        "green": colors.HexColor("#d9ead3"),
        "yellow": colors.HexColor("#fff2cc"),
        "orange": colors.HexColor("#fce5cd"),
        "red": colors.HexColor("#f4cccc"),
    }
    for _, row in dataframe.iterrows():
        interpretation = str(row["Interpretação"]).split(" ", 1)[-1]
        table_row = [
            Paragraph(escape(str(row["ROI"])), cell_style),
            Paragraph(escape(str(row[AGATSTON_SCORE_COLUMN])), cell_style),
            Paragraph(escape(str(row["Volume_score"])), cell_style),
            Paragraph(escape(interpretation), cell_style),
        ]
        if has_dose_metrics:
            table_row.extend([
                Paragraph(escape(_format_dose_value(row[DOSE_MEAN_COLUMN])), cell_style),
                Paragraph(escape(_format_dose_value(row[DOSE_D0_5CC_COLUMN])), cell_style),
            ])
        table_data.append(table_row)

    if has_dose_metrics:
        result_widths = [35 * mm, 27 * mm, 25 * mm, 33 * mm, 32 * mm, 29 * mm]
    else:
        result_widths = [46 * mm, 37 * mm, 37 * mm, 62 * mm]
    table = LongTable(
        table_data,
        colWidths=result_widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a6a6a6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for index, (_, row) in enumerate(dataframe.iterrows(), start=1):
        color = risk_colors.get(str(row["Cor"]))
        if color:
            table_styles.append(("BACKGROUND", (0, index), (-1, index), color))
    table.setStyle(TableStyle(table_styles))
    story.append(table)
    if has_dose_metrics:
        story.extend([
            Spacer(1, 2 * mm),
            Paragraph(
                "A dose média é ponderada pelo volume dos voxels da estrutura. "
                "D0,5cc corresponde à dose do volume mais quente de 0,5 cm³; "
                "as doses são interpoladas nos centros dos voxels da CT. "
                "Estruturas menores que 0,5 cm³ não têm D0,5cc calculável.",
                body_style,
            ),
        ])
    if all(column in dataframe for column in (
        DOSE_V8_COLUMN, DOSE_V16_COLUMN, DOSE_V20_COLUMN
    )):
        treatment_side = _treatment_laterality(dataframe)
        side_note = (
            f"Lateralidade inferida pela estrutura mamária com maior dose média: "
            f"mama {treatment_side.lower()}."
            if treatment_side
            else "Lateralidade não identificada por ausência de estrutura mamária "
                 "lateralizada ou empate na maior dose média; constraints não avaliados."
        )
        story.extend([
            PageBreak(),
            Paragraph("Constraints cardíacos — RTOG 1005", section_style),
            Paragraph(
                "Indicadores calculados com os limites de dose mais restritivos "
                "do Arm II: V16 Gy (ideal) e V20 Gy (aceitável) para o constraint 1; "
                "V8 Gy para o constraint 2. "
                f"{escape(side_note)} "
                "Fonte: tabela Normal Tissue Constraints do RTOG 1005 fornecida.",
                body_style,
            ),
            Spacer(1, 3 * mm),
        ])
        constraint_rows = _heart_constraint_results(dataframe)
        constraint_headers = [
            "Critério",
            "Mama",
            "Volume (Arm II)",
            "Limite",
            "Medido",
            "Indicativo",
        ]
        constraint_table_data = [[
            Paragraph(escape(value), header_style)
            for value in constraint_headers
        ]]
        for constraint in constraint_rows:
            constraint_table_data.append([
                Paragraph(escape(constraint["Classificação"]), cell_style),
                Paragraph(escape(constraint["Lateralidade"]), cell_style),
                Paragraph(escape(constraint["Métrica"]), cell_style),
                Paragraph(
                    escape(_format_percent_value(constraint["Limite (%)"])),
                    cell_style,
                ),
                Paragraph(
                    escape(_format_percent_value(constraint["Medido (%)"])),
                    cell_style,
                ),
                Paragraph(escape(constraint["Status"]), cell_style),
            ])
        constraint_table = LongTable(
            constraint_table_data,
            colWidths=[25 * mm, 25 * mm, 35 * mm, 25 * mm, 25 * mm, 35 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        constraint_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a6a6a6")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        status_colors = {
            "PASSA": colors.HexColor("#d9ead3"),
            "NÃO PASSA": colors.HexColor("#f4cccc"),
            "NÃO AVALIADO": colors.HexColor("#fff2cc"),
        }
        for index, constraint in enumerate(constraint_rows, start=1):
            constraint_styles.append((
                "BACKGROUND",
                (5, index),
                (5, index),
                status_colors[constraint["Status"]],
            ))
        constraint_table.setStyle(TableStyle(constraint_styles))
        story.extend([
            constraint_table,
            Spacer(1, 3 * mm),
            Paragraph(
                "Vx Gy representa o percentual do volume da área cardíaca que "
                "recebe dose igual ou superior a x Gy. PASSA/NÃO PASSA compara "
                "o valor medido com o limite indicado; NÃO AVALIADO indica que "
                "a ROI da área cardíaca não foi encontrada ou não possui dose válida.",
                body_style,
            ),
        ])
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("Como o escore é calculado", section_style),
        Paragraph(
            "São considerados os agrupamentos de pelo menos 4 pixels com densidade "
            "superior a 129 HU dentro de cada estrutura. Para cada agrupamento, "
            "calcula-se a área (número de pixels × área do pixel) e multiplica-se "
            "pelo fator de densidade máxima:",
            body_style,
        ),
        Spacer(1, 2 * mm),
    ])
    factor_rows = [
        ["Densidade máxima do agrupamento", "Fator aplicado à área"],
        ["130–199 HU", "1"],
        ["200–299 HU", "2"],
        ["300–399 HU", "3"],
        ["≥ 400 HU", "4"],
    ]
    factor_table = LongTable(factor_rows, colWidths=[90 * mm, 55 * mm], repeatRows=1)
    factor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a6a6a6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([
        factor_table,
        Spacer(1, 2 * mm),
        Paragraph(
            "O escore da estrutura é a soma dos escores dos agrupamentos em todos "
            "os cortes. O volume é calculado a partir da área dos pixels e do "
            "espaçamento entre cortes. Interpretação: 0 = nenhum risco detectável; "
            "1–100 = risco leve; 101–400 = risco moderado; acima de 400 = risco alto.",
            body_style,
        ),
    ])

    if volume is not None:
        class CalcificationFooterMarker(Flowable):
            def __init__(self):
                super().__init__()
                self.width = 1
                self.height = 1

            def draw(self):
                self.canv._calcification_footer_requested = True

        caption_style = ParagraphStyle(
            "ReportImageCaption",
            parent=body_style,
            alignment=TA_CENTER,
            fontSize=8,
            leading=10,
        )
        structure_items = {str(name): contours for name, contours in structures.items()}
        for _, row in dataframe.iterrows():
            score = float(row[AGATSTON_SCORE_COLUMN])
            if score <= 0:
                continue
            roi_name = str(row["ROI"])
            contours = structure_items.get(roi_name)
            if contours is None:
                raise ValueError(
                    f"Não foi possível localizar os contornos de {roi_name} "
                    "para incluir a imagem no relatório."
                )
            findings = _top_calcifications(contours, volume)
            if not findings:
                raise ValueError(
                    f"Não foi possível localizar o agrupamento calcificado de "
                    f"{roi_name} para incluir a imagem no relatório."
                )
            panels = []
            for index, finding in enumerate(findings, start=1):
                slice_dataset = volume.slices[finding["slice_index"]]
                slice_label = getattr(slice_dataset, "InstanceNumber", None)
                if slice_label is None:
                    slice_label = finding["slice_index"] + 1
                panels.append([
                    Paragraph(
                        f"<b>Grupo {index}</b><br/>"
                        f"Contribuição: {finding['score']:.1f}<br/>"
                        f"Corte axial: {escape(str(slice_label))}",
                        caption_style,
                    ),
                    ReportImage(
                        _render_calcification_crop(
                            volume, finding, dose_grid=dose_grid
                        ),
                        width=55 * mm,
                        height=55 * mm,
                        hAlign="CENTER",
                    ),
                ])
            panel_table = ReportTable(
                [panels],
                colWidths=[59 * mm] * len(panels),
                hAlign="CENTER",
            )
            panel_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.extend([
                PageBreak(),
                Paragraph(
                    f"Calcificação: {escape(roi_name)}",
                    section_style,
                ),
                Paragraph(
                    f"<b>Agatston score da estrutura:</b> {score:.1f} &nbsp;&nbsp; "
                    f"<b>Agrupamentos exibidos:</b> {len(findings)}",
                    body_style,
                ),
                Spacer(1, 4 * mm),
                panel_table,
            ])
            story.append(CalcificationFooterMarker())

    def draw_page_number(canvas, _document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        reference_style = ParagraphStyle(
            "ReportReference",
            parent=body_style,
            fontSize=6.5,
            leading=7.5,
            textColor=colors.HexColor("#666666"),
        )
        reference = Paragraph(
            "<b>Referência:</b> " + ARTICLE_REFERENCE_ABNT,
            reference_style,
        )
        reference_width = page_size[0] - document.leftMargin - document.rightMargin
        reference.wrap(reference_width, document.bottomMargin)
        reference.drawOn(canvas, document.leftMargin, 5 * mm)
        canvas.drawRightString(
            page_size[0] - 14 * mm,
            2 * mm,
            f"Página {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    if has_calcification_pages:
        class CalcificationFooterCanvas(pdf_canvas.Canvas):
            def __init__(self, *args, dose_levels, **kwargs):
                super().__init__(*args, **kwargs)
                self._dose_levels = dose_levels
                self._calcification_footer_requested = False

            def showPage(self):
                if self._calcification_footer_requested:
                    self.saveState()
                    footer_text = (
                        "Cada imagem mostra um recorte de 12 × 12 cm. O contorno "
                        "vermelho delimita a estrutura."
                    )
                    if self._dose_levels:
                        footer_text += (
                            " As linhas contínuas indicam isodoses de 1 Gy, "
                            "2 Gy, 5 Gy e, depois, em intervalos de 5 Gy."
                        )
                    footer_style = ParagraphStyle(
                        "CalcificationFooter",
                        parent=body_style,
                        fontSize=6.5,
                        leading=8,
                        alignment=TA_CENTER,
                        textColor=colors.HexColor("#444444"),
                    )
                    footer = Paragraph(escape(footer_text), footer_style)
                    footer_width = page_size[0] - 22 * mm
                    footer_height = footer.wrap(footer_width, 11 * mm)[1]
                    footer.drawOn(
                        self,
                        11 * mm,
                        24 * mm + (11 * mm - footer_height) / 2,
                    )
                    if self._dose_levels:
                        scale_left = 11 * mm
                        scale_width = page_size[0] - 22 * mm
                        cell_width = scale_width / len(self._dose_levels)
                        self.setFont("Helvetica", 5.5)
                        self.setFillColor(colors.HexColor("#444444"))
                        for index, level in enumerate(self._dose_levels):
                            center_x = scale_left + (index + 0.5) * cell_width
                            color = _isodose_color(index, len(self._dose_levels))
                            self.setFillColorRGB(
                                *(channel / 255 for channel in color)
                            )
                            bar_width = cell_width * 0.82
                            self.rect(
                                center_x - bar_width / 2,
                                17 * mm,
                                bar_width,
                                1.6 * mm,
                                stroke=0,
                                fill=1,
                            )
                            self.setFillColor(colors.HexColor("#444444"))
                            label = f"{level:g}"
                            self.drawCentredString(
                                center_x,
                                14 * mm,
                                label,
                            )
                    self.restoreState()
                self._calcification_footer_requested = False
                super().showPage()

        canvas_maker = lambda *args, **kwargs: CalcificationFooterCanvas(
            *args,
            dose_levels=isodose_levels,
            **kwargs,
        )
    else:
        canvas_maker = pdf_canvas.Canvas

    document.build(
        story,
        onFirstPage=draw_page_number,
        onLaterPages=draw_page_number,
        canvasmaker=canvas_maker,
    )


def show_results(root, dataframe, volume, structures, dose_grid=None):
    dataframe = normalize_results_dataframe(dataframe)
    first_slice = volume.slices[0]
    patient_name = str(getattr(first_slice, "PatientName", "Desconhecido"))
    patient_id = str(getattr(first_slice, "PatientID", "Desconhecido"))
    tomography_date = _tomography_date(first_slice)
    window = tk.Toplevel(root)
    title = f"Resultados - Paciente: {patient_name} (ID: {patient_id})"
    if dose_grid is not None:
        title += (
            f" | RTDOSE: {dose_grid.path.name} "
            f"({dose_grid.summation_type})"
        )
        if dose_grid.frame_mismatch:
            title += " | Frames distintos (coordenadas assumidas)"
    window.title(title)

    columns = ["ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação"]
    if DOSE_MEAN_COLUMN in dataframe and DOSE_D0_5CC_COLUMN in dataframe:
        columns.extend([DOSE_MEAN_COLUMN, DOSE_D0_5CC_COLUMN])
    tree = ttk.Treeview(
        window,
        columns=columns,
        show="headings",
        height=max(1, min(len(dataframe), 15)),
    )
    labels = {
        "ROI": "Estrutura",
        AGATSTON_SCORE_COLUMN: "Agatston Score",
        "Volume_score": "Volume (mm³)",
        "Interpretação": "Risco",
        DOSE_MEAN_COLUMN: "Dose média (Gy)",
        DOSE_D0_5CC_COLUMN: "D0,5cc (Gy)",
    }
    widths = {"ROI": 240, AGATSTON_SCORE_COLUMN: 150,
              "Volume_score": 150, "Interpretação": 260,
              DOSE_MEAN_COLUMN: 150, DOSE_D0_5CC_COLUMN: 150}
    for column in columns:
        tree.heading(column, text=labels[column])
        tree.column(column, width=widths[column], minwidth=90, stretch=True)
    def copy_results():
        try:
            root.clipboard_clear()
            root.clipboard_append(
                results_as_text(dataframe, patient_name, patient_id, dose_grid)
            )
            root.update()
        except tk.TclError as error:
            messagebox.showerror("Erro", f"Não foi possível copiar os resultados:\n{error}",
                                 parent=window)
            return
        messagebox.showinfo("Copiado", "Resultados copiados para a área de transferência.",
                            parent=window)

    actions = ttk.Frame(window)
    actions.pack(side="bottom", fill="x", padx=10, pady=10)
    saved_pdf_path = {"value": None}

    def open_saved_pdf():
        path = saved_pdf_path["value"]
        if path is None:
            return
        try:
            open_pdf(path)
        except (OSError, RuntimeError, webbrowser.Error) as error:
            messagebox.showerror("Erro ao abrir o PDF", str(error), parent=window)

    def export_pdf():
        prefix = "Relatorio_Dose" if dose_grid is not None else "Agatston"
        suggested_name = f"{prefix}_{patient_id}.pdf"
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in suggested_name
        )
        output_path = filedialog.asksaveasfilename(
            parent=window,
            title="Salvar relatório em PDF",
            initialdir=str(_pdf_initial_directory(volume)),
            initialfile=safe_name,
            defaultextension=".pdf",
            filetypes=[("Documento PDF", "*.pdf")],
        )
        if not output_path:
            return
        try:
            analysis_date = datetime.now().strftime("%d/%m/%Y")
            save_report_pdf(
                output_path,
                dataframe,
                patient_name,
                patient_id,
                tomography_date,
                analysis_date,
                volume=volume,
                structures=structures,
                dose_grid=dose_grid,
            )
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            messagebox.showerror("Erro ao salvar o PDF", str(error), parent=window)
            return
        saved_pdf_path["value"] = Path(output_path)
        open_pdf_button.configure(state="normal")
        messagebox.showinfo(
            "PDF salvo",
            f"O relatório foi salvo em:\n{output_path}",
            parent=window,
        )

    ttk.Button(actions, text="Copiar todos os dados", command=copy_results).pack(
        side="left", padx=(0, 8)
    )
    ttk.Button(actions, text="Salvar relatório em PDF...", command=export_pdf).pack(
        side="left", padx=(0, 8)
    )
    open_pdf_button = ttk.Button(
        actions,
        text="Abrir PDF gerado",
        command=open_saved_pdf,
        state="disabled",
    )
    open_pdf_button.pack(side="left", padx=(0, 8))
    if all(column in dataframe for column in (
        DOSE_V8_COLUMN, DOSE_V16_COLUMN, DOSE_V20_COLUMN
    )):
        ttk.Button(
            actions,
            text="Constraints cardíacos...",
            command=lambda: _show_heart_constraints(window, dataframe),
        ).pack(side="left")

    scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(10, 0))
    scrollbar.pack(side="left", fill="y", padx=(0, 10), pady=(10, 0))

    style = ttk.Style(window)
    style.configure("Treeview", rowheight=25)
    for _, row in dataframe.iterrows():
        values = [
            _format_dose_value(row[column])
            if column in {DOSE_MEAN_COLUMN, DOSE_D0_5CC_COLUMN}
            else row[column]
            for column in columns
        ]
        tree.insert(
            "",
            "end",
            values=tuple(values),
            tags=(row["Cor"],),
        )
    tree.tag_configure("green", background="lightgreen")
    tree.tag_configure("yellow", background="lightyellow")
    tree.tag_configure("orange", background="orange")
    tree.tag_configure("red", background="tomato")
    window.update_idletasks()
    window_height = min(520, max(150, window.winfo_reqheight()))
    _center_window(window, 1220 if dose_grid is not None else 920, window_height)
    window.protocol("WM_DELETE_WINDOW", root.destroy)


def _show_heart_constraints(parent, dataframe):
    constraints = _heart_constraint_results(dataframe)
    laterality = _treatment_laterality(dataframe)
    window = tk.Toplevel(parent)
    window.title("Constraints cardíacos — RTOG 1005 (Arm II)")
    window.transient(parent)
    window.grab_set()
    ttk.Label(
        window,
        text=(
            "Arm II: V16 Gy ideal / V20 Gy aceitável (constraint 1) e "
            "V8 Gy (constraint 2). "
            + (
                f"Lateralidade inferida pela estrutura mamária com maior dose "
                f"média: mama {laterality.lower()}."
                if laterality
                else "Lateralidade não identificada; os constraints não serão avaliados."
            )
        ),
        wraplength=880,
        justify="left",
    ).pack(fill="x", padx=12, pady=(12, 6))
    columns = (
        "Classificação",
        "Lateralidade",
        "Métrica",
        "Limite",
        "Medido",
        "Indicativo",
    )
    tree = ttk.Treeview(window, columns=columns, show="headings", height=8)
    widths = (120, 130, 110, 110, 110, 130)
    for column, width in zip(columns, widths):
        tree.heading(column, text=column)
        tree.column(column, width=width, minwidth=80, stretch=True)
    for constraint in constraints:
        tree.insert(
            "",
            "end",
            values=(
                constraint["Classificação"],
                constraint["Lateralidade"],
                constraint["Métrica"],
                _format_percent_value(constraint["Limite (%)"]),
                _format_percent_value(constraint["Medido (%)"]),
                constraint["Status"],
            ),
            tags=(constraint["Status"],),
        )
    tree.tag_configure("PASSA", background="lightgreen")
    tree.tag_configure("NÃO PASSA", background="tomato")
    tree.tag_configure("NÃO AVALIADO", background="lightyellow")
    tree.pack(fill="both", expand=True, padx=12, pady=6)
    ttk.Label(
        window,
        text=(
            "Vx Gy: percentual do volume da área cardíaca com dose ≥ x Gy. "
            "Os indicadores são auxiliares e não substituem a revisão clínica."
        ),
        wraplength=880,
        justify="left",
    ).pack(fill="x", padx=12, pady=6)
    ttk.Button(window, text="Fechar", command=window.destroy).pack(
        anchor="e", padx=12, pady=(0, 12)
    )
    _center_window(window, 920, 420)


def main():
    # Fluxo da interface: CT -> RTSTRUCTs -> RTDOSE opcional -> resultados.
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda _type, error, _traceback: messagebox.showerror(
        "Erro inesperado", str(error), parent=root
    )

    ct_folder = _choose_ct_folder(root, Path.home())
    if not ct_folder:
        root.destroy()
        return

    try:
        series, ignored_count, notices = scan_ct_series(ct_folder)
    except OSError as error:
        messagebox.showerror("Erro ao ler a pasta CT", str(error), parent=root)
        root.destroy()
        return
    if notices and series:
        details = "\n".join(notices[:8])
        messagebox.showwarning(
            "Validação da pasta CT",
            f"{ignored_count} arquivo(s) ignorado(s) e {len(notices)} aviso(s):\n{details}"
            + ("\n..." if len(notices) > 8 else ""),
            parent=root,
        )
    if not series:
        series_notices = [
            notice for notice in notices
            if notice.startswith("Série ")
            or "imagem CT sem SeriesInstanceUID" in notice
            or "arquivo marcado como CT" in notice
        ]
        details = "\n".join((series_notices or notices)[:8])
        messagebox.showerror(
            "Nenhuma série CT válida",
            "Não foi encontrada uma série CT DICOM válida na pasta selecionada."
            + (f"\n\nMotivos detectados:\n{details}" if details else ""),
            parent=root,
        )
        root.destroy()
        return

    series_uid = _choose_ct_series(root, series)
    if not series_uid:
        root.destroy()
        return
    try:
        volume = load_ct_volume(series_uid, series[series_uid])
    except (OSError, ValueError, InvalidDicomError) as error:
        messagebox.showerror("Erro ao carregar a série CT", str(error), parent=root)
        root.destroy()
        return

    rtstructs = _choose_rtstruct_files(root, ct_folder, volume)
    if not rtstructs:
        root.destroy()
        return
    structures, _merge_notices = merge_rtstructs(rtstructs)
    if not structures:
        messagebox.showerror(
            "Sem contornos",
            "Os RTSTRUCTs selecionados não contêm contornos de ROIs utilizáveis.",
            parent=root,
        )
        root.destroy()
        return

    dose_grid = _choose_rt_dose_file(root, ct_folder, volume)
    if dose_grid is None and not messagebox.askyesno(
        "Sem RTDOSE",
        "Nenhum RTDOSE foi selecionado. Deseja continuar sem calcular as métricas de dose?",
        parent=root,
    ):
        root.destroy()
        return

    try:
        dataframe, _invalid_contours, _outside_contours = calculate_results(
            structures, volume, dose_grid=dose_grid
        )
    except ValueError as error:
        messagebox.showerror("Erro ao calcular os resultados", str(error), parent=root)
        root.destroy()
        return
    show_results(root, dataframe, volume, structures, dose_grid=dose_grid)
    root.mainloop()


if __name__ == "__main__":
    main()
