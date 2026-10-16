from collections import defaultdict
from dataclasses import dataclass
import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import CTImageStorage, RTStructureSetStorage
from skimage import draw


CT_IMAGE_UID = str(CTImageStorage)
RTSTRUCT_UID = str(RTStructureSetStorage)
AGATSTON_SCORE_COLUMN = "Agatston_score"
ROI_TRANSLATIONS = {
    "cardiac_area": "Área cardíaca",
    "aorta": "Aorta",
    "myocardium": "Miocárdio",
    "heart_myocardium": "Miocárdio",
    "atrium_left": "Átrio esquerdo",
    "heart_atrium_left": "Átrio esquerdo",
    "atrium_right": "Átrio direito",
    "heart_atrium_right": "Átrio direito",
    "ventricle_left": "Ventrículo esquerdo",
    "heart_ventricle_left": "Ventrículo esquerdo",
    "ventricle_right": "Ventrículo direito",
    "heart_ventricle_right": "Ventrículo direito",
    "pulmonary_artery": "Artéria pulmonar",
    "artéria pulmonar": "Artéria pulmonar",
    "coronary_arteries": "Artérias coronárias",
}
LEGACY_AGATSTON_SCORE_COLUMN = "Agatson_score"
READ_ERRORS = (InvalidDicomError, OSError, EOFError, ValueError)
PIXEL_ERRORS = (AttributeError, OSError, RuntimeError, ValueError, NotImplementedError)


@dataclass
class CTVolume:
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


def _center_window(window, width, height):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.minsize(width, height)


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
    active = np.isfinite(matrix) & (matrix > threshold)
    visited = np.zeros(active.shape, dtype=bool)
    clusters = []
    rows, columns = active.shape
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
    for row, column in np.argwhere(active):
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(int(row), int(column))]
        cluster = []
        while stack:
            current_row, current_column = stack.pop()
            cluster.append((current_row, current_column))
            for d_row, d_column in neighbors:
                next_row = current_row + d_row
                next_column = current_column + d_column
                if (0 <= next_row < rows and 0 <= next_column < columns
                        and active[next_row, next_column]
                        and not visited[next_row, next_column]):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if len(cluster) >= min_size:
            clusters.append(cluster)
    return clusters


def agatson_score(filtered_array, cluster, volume, slice_index):
    pixel_area = volume.pixel_spacing[0] * volume.pixel_spacing[1]
    maximum = max(float(filtered_array[row, column]) for row, column in cluster)
    area_score = len(cluster) * pixel_area
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


def calculate_results(structures, volume):
    results = []
    invalid_contours = 0
    outside_contours = 0
    for roi_name, contours in structures.items():
        mask, invalid, outside = get_mask_from_contours(contours, volume)
        invalid_contours += invalid
        outside_contours += outside
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
        results.append({
            "ROI": roi_name,
            AGATSTON_SCORE_COLUMN: round(total_score, 1),
            "Volume_score": round(total_volume, 1),
            "Interpretação": risk,
            "Cor": color,
        })
    columns = ["ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação", "Cor"]
    dataframe = pd.DataFrame(results, columns=columns)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=AGATSTON_SCORE_COLUMN, ascending=False)
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


def results_as_text(dataframe, patient_name, patient_id):
    dataframe = normalize_results_dataframe(dataframe)
    output = StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(["Paciente", patient_name])
    writer.writerow(["ID", patient_id])
    writer.writerow([])
    columns = ["ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação"]
    writer.writerow(["Estrutura", "Agatston Score", "Volume (mm³)", "Interpretação"])
    for _, row in dataframe.iterrows():
        writer.writerow([row[column] for column in columns])
    return output.getvalue()


def save_report_pdf(
    path,
    dataframe,
    patient_name,
    patient_id,
    tomography_date,
    analysis_date,
):
    dataframe = normalize_results_dataframe(dataframe)
    from xml.sax.saxutils import escape

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            LongTable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
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
    document = SimpleDocTemplate(
        str(path),
        pagesize=page_size,
        rightMargin=11 * mm,
        leftMargin=11 * mm,
        topMargin=12 * mm,
        bottomMargin=16 * mm,
        title="Resultado da análise de escore de Agatston",
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
        Paragraph("Resultado da análise de escore de Agatston", title_style),
        Paragraph(f"<b>Paciente:</b> {escape(patient_name)}", body_style),
        Paragraph(f"<b>ID:</b> {escape(patient_id)}", body_style),
        Paragraph(f"<b>Data da tomografia:</b> {escape(tomography_date)}", body_style),
        Paragraph(f"<b>Data da análise:</b> {escape(analysis_date)}", body_style),
        Spacer(1, 4 * mm),
        Paragraph("Resultados", section_style),
    ]

    headers = ["Estrutura", "Agatston Score", "Volume (mm³)", "Interpretação"]
    table_data = [[Paragraph(escape(value), header_style) for value in headers]]
    risk_colors = {
        "green": colors.HexColor("#d9ead3"),
        "yellow": colors.HexColor("#fff2cc"),
        "orange": colors.HexColor("#fce5cd"),
        "red": colors.HexColor("#f4cccc"),
    }
    for _, row in dataframe.iterrows():
        interpretation = str(row["Interpretação"]).split(" ", 1)[-1]
        table_data.append([
            Paragraph(escape(str(row["ROI"])), cell_style),
            Paragraph(escape(str(row[AGATSTON_SCORE_COLUMN])), cell_style),
            Paragraph(escape(str(row["Volume_score"])), cell_style),
            Paragraph(escape(interpretation), cell_style),
        ])

    table = LongTable(
        table_data,
        colWidths=[46 * mm, 37 * mm, 37 * mm, 62 * mm],
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

    def draw_page_number(canvas, _document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(
            page_size[0] - 14 * mm,
            9 * mm,
            f"Página {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)


def show_results(root, dataframe, volume):
    dataframe = normalize_results_dataframe(dataframe)
    first_slice = volume.slices[0]
    patient_name = str(getattr(first_slice, "PatientName", "Desconhecido"))
    patient_id = str(getattr(first_slice, "PatientID", "Desconhecido"))
    tomography_date = _tomography_date(first_slice)
    window = tk.Toplevel(root)
    window.title(f"Resultados - Paciente: {patient_name} (ID: {patient_id})")
    _center_window(window, 920, 520)

    columns = ("ROI", AGATSTON_SCORE_COLUMN, "Volume_score", "Interpretação")
    tree = ttk.Treeview(window, columns=columns, show="headings")
    labels = {
        "ROI": "Estrutura",
        AGATSTON_SCORE_COLUMN: "Agatston Score",
        "Volume_score": "Volume (mm³)",
        "Interpretação": "Risco",
    }
    widths = {"ROI": 240, AGATSTON_SCORE_COLUMN: 150,
              "Volume_score": 150, "Interpretação": 260}
    for column in columns:
        tree.heading(column, text=labels[column])
        tree.column(column, width=widths[column], minwidth=90, stretch=True)

    def copy_results():
        try:
            root.clipboard_clear()
            root.clipboard_append(results_as_text(dataframe, patient_name, patient_id))
            root.update()
        except tk.TclError as error:
            messagebox.showerror("Erro", f"Não foi possível copiar os resultados:\n{error}",
                                 parent=window)
            return
        messagebox.showinfo("Copiado", "Resultados copiados para a área de transferência.",
                            parent=window)

    actions = ttk.Frame(window)
    actions.pack(side="bottom", fill="x", padx=10, pady=10)

    def export_pdf():
        suggested_name = f"Agatston_{patient_id}.pdf"
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in suggested_name
        )
        output_path = filedialog.asksaveasfilename(
            parent=window,
            title="Salvar relatório em PDF",
            initialdir=str(Path.home()),
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
            )
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            messagebox.showerror("Erro ao salvar o PDF", str(error), parent=window)
            return
        messagebox.showinfo(
            "PDF salvo",
            f"O relatório foi salvo em:\n{output_path}",
            parent=window,
        )

    ttk.Button(actions, text="Copiar todos os dados", command=copy_results).pack(
        side="left", padx=(0, 8)
    )
    ttk.Button(actions, text="Salvar relatório em PDF...", command=export_pdf).pack(
        side="left"
    )

    results_frame = ttk.Frame(window)
    results_frame.pack(side="top", fill="both", expand=True, padx=10, pady=0)
    scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    style = ttk.Style(window)
    style.configure("Treeview", rowheight=25)
    for _, row in dataframe.iterrows():
        tree.insert(
            "",
            "end",
            values=tuple(row[column] for column in columns),
            tags=(row["Cor"],),
        )
    tree.tag_configure("green", background="lightgreen")
    tree.tag_configure("yellow", background="lightyellow")
    tree.tag_configure("orange", background="orange")
    tree.tag_configure("red", background="tomato")
    window.protocol("WM_DELETE_WINDOW", root.destroy)


def main():
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda _type, error, _traceback: messagebox.showerror(
        "Erro inesperado", str(error), parent=root
    )

    ct_folder = filedialog.askdirectory(
        parent=root,
        title="Selecione a pasta com as imagens CT DICOM",
        initialdir=str(Path.home()),
        mustexist=True,
    )
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
    structures, merge_notices = merge_rtstructs(rtstructs)
    if merge_notices:
        messagebox.showwarning(
            "Contornos ignorados",
            "\n".join(merge_notices[:8])
            + ("\n..." if len(merge_notices) > 8 else ""),
            parent=root,
        )
    if not structures:
        messagebox.showerror(
            "Sem contornos",
            "Os RTSTRUCTs selecionados não contêm contornos de ROIs utilizáveis.",
            parent=root,
        )
        root.destroy()
        return

    dataframe, invalid_contours, outside_contours = calculate_results(structures, volume)
    if invalid_contours or outside_contours:
        messagebox.showwarning(
            "Contornos ignorados",
            f"{invalid_contours} contorno(s) inválido(s) e "
            f"{outside_contours} fora da série CT foram ignorados.",
            parent=root,
        )
    show_results(root, dataframe, volume)
    root.mainloop()


if __name__ == "__main__":
    main()
