#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unir_rtstructs_xio_full.py
Versão otimizada e completa com traduções estendidas e criação de medula_PRV.
Uso: python unir_rtstructs_xio_full.py -o saida.dcm arquivo1.dcm arquivo2.dcm ...
Dependências: pydicom, numpy, shapely
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import unicodedata
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from shapely.geometry import Polygon, LinearRing
from shapely.ops import unary_union

# ---------------------------
# Configurações e tradução
# ---------------------------
MAX_NAME_LEN = 15
RTSTRUCT_SOPCLASS = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage

TRADUCAO = {
    "breast": "mama",
    "breast_right": "mama_D",
    "breast_left": "mama_E",
    "heart_myocardium": "miocardio",
    "heart_atrium_left": "atri_E",
    "heart_ventricle_left": "ventri_E",
    "heart_atrium_right": "atri_D",
    "heart_ventricle_right": "ventri_D",
    "aorta": "aorta",
    "pulmonary_artery": "art_pulm",
    "cardiac_area": "area_card",
    "coronary_arteries": "art_coron",
    "corpo": "corpo",
    "subcutaneous_fat": "gord_subcut",
    "torso_fat": "gord_torax",
    "skeletal_muscle": "musculo",
    "intermuscular_fat": "gord_inter",
    "spleen": "baco",
    "gallbladder": "vesicula",
    "liver": "figado",
    "stomach": "estomago",
    "pancreas": "pancreas",
    "lung_upper_lobe_left": "pulmao_sup_E",
    "lung_lower_lobe_left": "pulmao_inf_E",
    "lung_upper_lobe_right": "pulmao_sup_D",
    "lung_middle_lobe_right": "pulmao_med_D",
    "lung_lower_lobe_right": "pulmao_inf_D",
    "esophagus": "esofago",
    "trachea": "traqueia",
    "thyroid_gland": "tireoide",
    "duodenum": "duodeno",
    "vertebrae_l2": "vertebra_L2",
    "vertebrae_l1": "vertebra_L1",
    "vertebrae_t11": "vertebra_T11",
    "vertebrae_t10": "vertebra_T10",
    "vertebrae_t9": "vertebra_T9",
    "vertebrae_t8": "vertebra_T8",
    "vertebrae_t7": "vertebra_T7",
    "vertebrae_t6": "vertebra_T6",
    "vertebrae_t5": "vertebra_T5",
    "vertebrae_t4": "vertebra_T4",
    "vertebrae_t3": "vertebra_T3",
    "vertebrae_t2": "vertebra_T2",
    "vertebrae_t1": "vertebra_T1",
    "vertebrae_c7": "vertebra_C7",
    "vertebrae_c6": "vertebra_C6",
    "vertebrae_c5": "vertebra_C5",
    "vertebrae_c4": "vertebra_C4",
    "vertebrae_c3": "vertebra_C3",
    "vertebrae_c2": "vertebra_C2",
    "vertebrae_c1": "vertebra_C1",
    "humerus_left": "umero_E",
    "humerus_right": "umero_D",
    "spinal_cord": "medula",
    "rib_left_1": "costela_E1",
    "rib_left_2": "costela_E2",
    "rib_left_3": "costela_E3",
    "rib_left_4": "costela_E4",
    "rib_left_5": "costela_E5",
    "rib_left_6": "costela_E6",
    "rib_left_7": "costela_E7",
    "rib_left_8": "costela_E8",
    "rib_left_9": "costela_E9",
    "rib_left_10": "costela_E10",
    "rib_left_11": "costela_E11",
    "rib_left_12": "costela_E12",
    "rib_right_1": "costela_D1",
    "rib_right_2": "costela_D2",
    "rib_right_3": "costela_D3",
    "rib_right_4": "costela_D4",
    "rib_right_5": "costela_D5",
    "rib_right_6": "costela_D6",
    "rib_right_7": "costela_D7",
    "rib_right_8": "costela_D8",
    "rib_right_9": "costela_D9",
    "rib_right_10": "costela_D10",
    "rib_right_11": "costela_D11",
    "rib_right_12": "costela_D12",
    "sternum": "esterno",
    "costal_cartilages": "cartil_cost"
}

# ---------------------------
# Cores (RGB)
# ---------------------------
COLOR_PULMAO_E = (0, 200, 0)
COLOR_PULMAO_D = (0, 0, 200)
COLOR_PULMOES = (135, 206, 250)
COLOR_MEDULA_PRV = (255, 165, 0)
COLOR_COSTELAS = (220, 220, 220)

# ---------------------------
# Utilitários de string
# ---------------------------
def strip_accents(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    s = strip_accents(name).strip().lower()
    for ch in [" ", "-", "/", "\\", "(", ")", ",", "."]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def translate_name(original: str) -> str:
    norm = normalize_name(original)
    if norm in TRADUCAO:
        t = TRADUCAO[norm]
    else:
        low = strip_accents(str(original)).lower().replace(" ", "_")
        t = TRADUCAO.get(low, norm)
    t = t.strip()
    return t[:MAX_NAME_LEN] if len(t) > MAX_NAME_LEN else t

def vertebra_display_name(translated: str) -> str:
    if not translated:
        return ""
    tl = translated.lower()
    if tl.startswith("vertebra_"):
        return translated.split("_", 1)[1]
    if tl.startswith("vertebra"):
        parts = translated.split("_")
        if len(parts) > 1:
            return parts[1]
        return translated.replace("vertebra", "").strip("_")
    return translated

# ---------------------------
# Geometria: conversões
# ---------------------------
def contour_points_from_ds(contour_ds: Dataset) -> List[Tuple[float, float, float]]:
    data = getattr(contour_ds, "ContourData", None)
    if data is None:
        return []
    pts = [float(x) for x in data]
    return [(pts[i], pts[i + 1], pts[i + 2]) for i in range(0, len(pts), 3)]

def polygon_from_contour_points(pts: List[Tuple[float, float, float]]) -> Optional[Polygon]:
    if not pts:
        return None
    xy = [(p[0], p[1]) for p in pts]
    try:
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None

def polygon_to_contour_sequence(poly: Polygon, z: float, n_points: int = 120) -> List[List[Tuple[float, float, float]]]:
    if poly is None or poly.is_empty:
        return []
    exterior = poly.exterior
    coords = list(exterior.coords)
    if len(coords) < 4:
        return []
    ring = LinearRing(coords)
    total = ring.length
    if total == 0:
        return []
    samples = []
    for i in range(n_points):
        d = (i / float(n_points)) * total
        pt = ring.interpolate(d)
        samples.append((float(pt.x), float(pt.y), float(z)))
    return [samples]

def create_contour_dataset_from_points(pts: List[Tuple[float, float, float]]) -> Dataset:
    item = Dataset()
    item.ContourGeometricType = "CLOSED_PLANAR"
    item.NumberOfContourPoints = len(pts)
    flat = []
    for (x, y, z) in pts:
        flat.extend([float(x), float(y), float(z)])
    item.ContourData = flat
    return item

def polygons_group_by_z(polygons_with_z: List[Tuple[float, Polygon]]) -> Dict[float, Polygon]:
    d = defaultdict(list)
    for (z, poly) in polygons_with_z:
        key = round(float(z), 3)
        d[key].append(poly)
    out = {}
    for z, lst in d.items():
        out[z] = unary_union(lst) if lst else None
    return out

# ---------------------------
# Buffer / erosão
# ---------------------------
def dilate_by_margin(polys_with_z: List[Tuple[float, Polygon]], margin_mm: float = 3.0) -> Dict[float, Polygon]:
    grouped = polygons_group_by_z([(z, poly) for (z, poly) in polys_with_z])
    out = {}
    for z, poly in grouped.items():
        if poly is None:
            continue
        try:
            dilated = poly.buffer(margin_mm)
            if dilated is None or dilated.is_empty:
                continue
            out[z] = dilated
        except Exception:
            continue
    return out

def erode_grouped_by_margin(grouped_by_z: Dict[float, Polygon], margin_mm: float = 3.0) -> Dict[float, Polygon]:
    out = {}
    for z, poly in grouped_by_z.items():
        if poly is None:
            continue
        try:
            eroded = poly.buffer(-margin_mm)
            if eroded is None or eroded.is_empty:
                continue
            out[z] = eroded
        except Exception:
            continue
    return out

# ---------------------------
# Coleta de ROIs a partir de múltiplos RTSTRUCTs
# ---------------------------
def collect_rois_from_ds(ds: Dataset) -> Dict[str, Dict]:
    roi_map = {}
    if not hasattr(ds, "StructureSetROISequence"):
        return roi_map
    for roi in ds.StructureSetROISequence:
        num = getattr(roi, "ROINumber", None)
        name = getattr(roi, "ROIName", "") or ""
        norm = normalize_name(name)
        if norm == "":
            continue
        if norm not in roi_map:
            roi_map[norm] = {"samples": [], "original_names": set()}
        roi_map[norm]["samples"].append((ds, num, roi))
        roi_map[norm]["original_names"].add(name)
    return roi_map

def build_polygons_for_roi_samples(entry_samples: List[Tuple[Dataset, int, Dataset]]) -> List[Tuple[float, Polygon, Dataset, int]]:
    polygons = []
    for (src_ds, src_num, roi_ds) in entry_samples:
        if not hasattr(src_ds, "ROIContourSequence"):
            continue
        for roi_contour in src_ds.ROIContourSequence:
            if getattr(roi_contour, "ReferencedROINumber", None) != src_num:
                continue
            if not hasattr(roi_contour, "ContourSequence"):
                continue
            for contour in roi_contour.ContourSequence:
                pts = contour_points_from_ds(contour)
                if not pts:
                    continue
                zs = [p[2] for p in pts]
                z = float(sum(zs) / len(zs))
                poly = polygon_from_contour_points(pts)
                if poly is None:
                    continue
                polygons.append((z, poly, src_ds, src_num))
    return polygons

# ---------------------------
# Utilitários DICOM
# ---------------------------
def safe_truncate_name(name: str) -> str:
    name = str(name).strip()
    if len(name) > MAX_NAME_LEN:
        return name[:MAX_NAME_LEN]
    return name

def copy_contour_image_sequence_from_sample(sample_roi_contour: Optional[Dataset]) -> Optional[List[Dataset]]:
    if sample_roi_contour is None:
        return None
    cis = getattr(sample_roi_contour, "ContourImageSequence", None)
    if cis is None:
        return None
    return [copy.deepcopy(x) for x in cis]

def create_roi_contour_item_from_grouped_polys(grouped_by_z: Dict[float, Polygon],
                                               new_number: int,
                                               color_rgb: Optional[Tuple[int, int, int]] = None,
                                               sample_roi_contour_for_images: Optional[Dataset] = None) -> Dataset:
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = int(new_number)
    contour_seq = []
    cis_template = copy_contour_image_sequence_from_sample(sample_roi_contour_for_images)
    for z in sorted(grouped_by_z.keys()):
        geom = grouped_by_z[z]
        if geom is None or geom.is_empty:
            continue
        geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for g in geoms:
            contours_pts_list = polygon_to_contour_sequence(g, z, n_points=120)
            for pts in contours_pts_list:
                contour_item = create_contour_dataset_from_points(pts)
                if cis_template:
                    contour_item.ContourImageSequence = [copy.deepcopy(x) for x in cis_template]
                contour_seq.append(contour_item)
    if contour_seq:
        roi_contour.ContourSequence = contour_seq
    if color_rgb is not None:
        roi_contour.ROIDisplayColor = [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])]
    return roi_contour

# ---------------------------
# Parser de argumentos
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-o", "--output", required=False, help="arquivo de saída (ex: /caminho/saida.dcm)")
    p.add_argument("--debug", action="store_true", help="imprime resumo de ROIs e contagens antes de salvar")
    p.add_argument("inputs", nargs="*", help="arquivos RTSTRUCT DICOM")
    args = p.parse_args()
    if not args.inputs:
        print("Uso mínimo: python unir_rtstructs_xio_full.py -o saida.dcm arquivo1.dcm arquivo2.dcm")
        sys.exit(1)
    out = args.output or "Grupo_unido_pulmoes_xio_full.dcm"
    return args.inputs, out, args.debug

# ---------------------------
# Função principal
# ---------------------------
def main():
    inputs, out_name, debug = parse_args()
    ds_base: Optional[Dataset] = None
    roi_map_all: Dict[str, Dict] = {}

    # Ler e agregar amostras de todos os RTSTRUCTs
    for path in inputs:
        if not os.path.exists(path):
            print(f"Aviso: arquivo não encontrado {path}, pulando.")
            continue
        try:
            ds = pydicom.dcmread(path)
        except Exception as e:
            print(f"Aviso: não foi possível ler {path}: {e}")
            continue

        if ds_base is None:
            ds_base = copy.deepcopy(ds)
            for seq_name in ("StructureSetROISequence", "ROIContourSequence", "RTROIObservationsSequence"):
                if hasattr(ds_base, seq_name):
                    delattr(ds_base, seq_name)

        roi_map = collect_rois_from_ds(ds)
        for norm, info in roi_map.items():
            if norm not in roi_map_all:
                roi_map_all[norm] = {"samples": [], "original_names": set()}
            roi_map_all[norm]["samples"].extend(info["samples"])
            roi_map_all[norm]["original_names"].update(info["original_names"])

    if ds_base is None:
        print("Nenhum RTSTRUCT válido fornecido.")
        sys.exit(1)

    # Construir polígonos por ROI
    roi_polygons: Dict[str, List[Tuple[float, Polygon, Dataset, int]]] = {}
    for norm, entry in roi_map_all.items():
        polygons = build_polygons_for_roi_samples(entry["samples"])
        roi_polygons[norm] = polygons

    # Definir chaves dos lobos pulmonares originais
    left_lobe_keys = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    right_lobe_keys = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]

    def collect_polygons_for_keys(keys_list: List[str]) -> List[Tuple[float, Polygon]]:
        lst: List[Tuple[float, Polygon]] = []
        for k in keys_list:
            if k in roi_polygons:
                lst.extend([(z, poly) for (z, poly, *rest) in roi_polygons[k]])
        return lst

    # Step 1: dilatar lobos originais por 3 mm (margem)
    left_lobes_polys = collect_polygons_for_keys(left_lobe_keys)
    right_lobes_polys = collect_polygons_for_keys(right_lobe_keys)

    dilated_left = dilate_by_margin(left_lobes_polys, margin_mm=3.0) if left_lobes_polys else {}
    dilated_right = dilate_by_margin(right_lobes_polys, margin_mm=3.0) if right_lobes_polys else {}

    # Step 2 & 3: unir por lado e erodir 3 mm -> pulmao_E / pulmao_D
    eroded_left = erode_grouped_by_margin(dilated_left, margin_mm=3.0) if dilated_left else {}
    eroded_right = erode_grouped_by_margin(dilated_right, margin_mm=3.0) if dilated_right else {}

    created_items = OrderedDict()

    # tentar obter ContourImageSequence de amostras para preservar referências de fatia
    def find_sample_roi_contour_for_keys(keys):
        for k in keys:
            if k in roi_map_all and roi_map_all[k]["samples"]:
                s_ds, s_num, _ = roi_map_all[k]["samples"][0]
                if hasattr(s_ds, "ROIContourSequence"):
                    for rc in s_ds.ROIContourSequence:
                        if getattr(rc, "ReferencedROINumber", None) == s_num:
                            return rc
        return None

    sample_left = find_sample_roi_contour_for_keys(left_lobe_keys)
    sample_right = find_sample_roi_contour_for_keys(right_lobe_keys)

    if eroded_left:
        created_items["pulmao_E"] = {
            "grouped": eroded_left,
            "color": COLOR_PULMAO_E,
            "label": "pulmao_E",
            "sample_roi_contour": sample_left
        }

    if eroded_right:
        created_items["pulmao_D"] = {
            "grouped": eroded_right,
            "color": COLOR_PULMAO_D,
            "label": "pulmao_D",
            "sample_roi_contour": sample_right
        }

    # Step 4: pulmoes = união(pulmao_E, pulmao_D)
    all_eroded = []
    sample_for_pulmoes = None
    if "pulmao_E" in created_items:
        all_eroded.extend([(z, poly) for (z, poly) in created_items["pulmao_E"]["grouped"].items()])
        sample_for_pulmoes = sample_for_pulmoes or created_items["pulmao_E"].get("sample_roi_contour")
    if "pulmao_D" in created_items:
        all_eroded.extend([(z, poly) for (z, poly) in created_items["pulmao_D"]["grouped"].items()])
        sample_for_pulmoes = sample_for_pulmoes or created_items["pulmao_D"].get("sample_roi_contour")
    if all_eroded:
        grouped_combined = polygons_group_by_z(all_eroded)
        created_items["pulmoes"] = {
            "grouped": grouped_combined,
            "color": COLOR_PULMOES,
            "label": "pulmoes",
            "sample_roi_contour": sample_for_pulmoes
        }

    # medula_PRV (spinal_cord buffer 3mm) - criado explicitamente
    if "spinal_cord" in roi_polygons:
        spinal_polys = [(z, poly) for (z, poly, *rest) in roi_polygons["spinal_cord"]]
        grouped_spinal = polygons_group_by_z(spinal_polys)
        buffered = {}
        for z, poly in grouped_spinal.items():
            if poly is None:
                continue
            buf = poly.buffer(3.0)
            if buf is None or buf.is_empty:
                continue
            buffered[z] = buf
        if buffered:
            sample_spinal = None
            if roi_map_all.get("spinal_cord", {}).get("samples"):
                s_ds, s_num, _ = roi_map_all["spinal_cord"]["samples"][0]
                if hasattr(s_ds, "ROIContourSequence"):
                    for rc in s_ds.ROIContourSequence:
                        if getattr(rc, "ReferencedROINumber", None) == s_num:
                            sample_spinal = rc
                            break
            created_items["medula_PRV"] = {
                "grouped": buffered,
                "color": COLOR_MEDULA_PRV,
                "label": "medula_PRV",
                "sample_roi_contour": sample_spinal
            }

    # costelas: unir todas as costelas em "costelas" (mantido)
    rib_keys = [k for k in roi_polygons.keys() if k.startswith("rib_") or k.startswith("rib_left_") or k.startswith("rib_right_") or "costela" in k]
    if rib_keys:
        all_ribs = []
        sample_rib_roi_contour = None
        for k in rib_keys:
            entries = roi_polygons.get(k, [])
            all_ribs.extend([(z, poly) for (z, poly, *rest) in entries])
            if sample_rib_roi_contour is None and roi_map_all.get(k, {}).get("samples"):
                s_ds, s_num, _ = roi_map_all[k]["samples"][0]
                if hasattr(s_ds, "ROIContourSequence"):
                    for rc in s_ds.ROIContourSequence:
                        if getattr(rc, "ReferencedROINumber", None) == s_num:
                            sample_rib_roi_contour = rc
                            break
        grouped_ribs = polygons_group_by_z(all_ribs) if all_ribs else {}
        if grouped_ribs:
            created_items["costelas"] = {
                "grouped": grouped_ribs,
                "color": COLOR_COSTELAS,
                "label": "costelas",
                "sample_roi_contour": sample_rib_roi_contour
            }

    # ---------------------------
    # Preparar lista final ordenada de nomes (usaremos nomes traduzidos)
    # ---------------------------
    translated_map = {}
    for norm, entry in roi_map_all.items():
        orig_name = list(entry["original_names"])[0] if entry["original_names"] else norm
        t = translate_name(orig_name)
        if t.lower().startswith("vertebra_") or t.lower().startswith("vertebra"):
            t_disp = vertebra_display_name(t)
        else:
            t_disp = t
        if "costela" in t_disp.lower():
            t_disp = t_disp.replace("costela", "cost")
        translated_map[norm] = t_disp

    final_order: List[Tuple[str, str, Optional[str]]] = []

    # 1. corpo
    if "corpo" in roi_map_all:
        final_order.append(("orig", "corpo", None))

    # 2. mamas: incluir apenas mama_D e mama_E se presentes; não exportar genérico "mama"
    for key in ("breast_right", "breast_left"):
        if key in roi_map_all:
            rep = list(roi_map_all[key]["original_names"])[0]
            if translate_name(rep) == "mama":
                continue
            final_order.append(("orig", key, None))

    # 3. Pulmões: incluir apenas as criadas (pulmao_E, pulmao_D, pulmoes)
    for name in ("pulmao_E", "pulmao_D", "pulmoes"):
        if name in created_items:
            final_order.append(("created", name, None))

    # 4. coração e relacionados
    heart_related_norms = sorted([k for k in roi_map_all.keys() if k.startswith("heart_") or k in ("aorta", "cardiac_area", "coronary_arteries")])
    for k in heart_related_norms:
        final_order.append(("orig", k, None))

    # 5. medula and medula_PRV
    if "spinal_cord" in roi_map_all:
        final_order.append(("orig", "spinal_cord", None))
    if "medula_PRV" in created_items:
        final_order.append(("created", "medula_PRV", None))

    # 6. fígado e vesícula próximos
    if "liver" in roi_map_all:
        final_order.append(("orig", "liver", None))
    if "gallbladder" in roi_map_all:
        final_order.append(("orig", "gallbladder", None))

    # 7. vértebras: cervical, thoracic, lumbar (não excluir nenhuma)
    for i in range(1, 8):
        key = f"vertebrae_c{i}"
        if key in roi_map_all:
            final_order.append(("orig", key, "cervical"))
    for i in range(1, 12):
        key = f"vertebrae_t{i}"
        if key in roi_map_all:
            final_order.append(("orig", key, "thoracic"))
    for i in range(1, 3):
        key = f"vertebrae_l{i}"
        if key in roi_map_all:
            final_order.append(("orig", key, "lumbar"))

    # 8. costelas agregada antes das separadas
    if "costelas" in created_items:
        final_order.append(("created", "costelas", None))
    else:
        for k in sorted([k for k in roi_map_all.keys() if "rib_" in k or "costela" in k]):
            final_order.append(("orig", k, None))

    # 9. restantes (alfabético por nome traduzido), excluindo já usados e excluindo "mama" e lobos pulmonares
    used = set((entry[0], entry[1]) for entry in final_order)
    remaining = []
    lobes_to_exclude = set(left_lobe_keys + right_lobe_keys)
    for k in roi_map_all.keys():
        if ("orig", k) in used:
            continue
        if "costelas" in created_items and (k in rib_keys):
            continue
        if k in lobes_to_exclude:
            continue
        rep = list(roi_map_all[k]["original_names"])[0]
        if translate_name(rep) == "mama":
            continue
        remaining.append(k)
    remaining_sorted = sorted(remaining, key=lambda x: translated_map.get(x, x))
    for k in remaining_sorted:
        final_order.append(("orig", k, None))

    # ---------------------------
    # Atribuir números sequenciais e construir sequências DICOM
    # ---------------------------
    ds_base.StructureSetROISequence = []
    ds_base.ROIContourSequence = []
    ds_base.RTROIObservationsSequence = []

    roi_number_map: Dict[Tuple[str, str], int] = {}
    next_roi_number = 1

    # First pass: assign numbers
    for typ, key, region in final_order:
        roi_number_map[(typ, key)] = next_roi_number
        next_roi_number += 1

    # Helper para adicionar StructureSetROISequence e RTROIObservationsSequence
    def add_structure_and_obs(number: int, name: str, generation: str = "MANUAL", interpreted_type: Optional[str] = None):
        s_item = Dataset()
        s_item.ROINumber = int(number)
        s_item.ROIName = safe_truncate_name(name)
        s_item.ROIGenerationAlgorithm = generation
        if hasattr(ds_base, "FrameOfReferenceUID"):
            try:
                s_item.ReferencedFrameOfReferenceUID = ds_base.FrameOfReferenceUID
            except Exception:
                pass
        ds_base.StructureSetROISequence.append(s_item)

        obs = Dataset()
        obs.ReferencedROINumber = int(number)
        obs.ObservationNumber = int(number)
        obs.ROIObservationLabel = safe_truncate_name(name)
        if interpreted_type:
            obs.RTROIInterpretedType = interpreted_type
        else:
            obs.RTROIInterpretedType = "ORGAN"
        ds_base.RTROIObservationsSequence.append(obs)

    # Second pass: criar StructureSetROISequence entries (com nomes traduzidos)
    for typ, key, region in final_order:
        number = roi_number_map[(typ, key)]
        if typ == "created":
            label = created_items[key]["label"]
            add_structure_and_obs(number, label, generation="MANUAL", interpreted_type="ORGAN")
        else:
            entry = roi_map_all.get(key)
            if not entry:
                continue
            orig_name = list(entry["original_names"])[0] if entry["original_names"] else key
            if translate_name(orig_name) == "mama":
                continue
            t = translate_name(orig_name)
            if t.lower().startswith("vertebra"):
                t_disp = vertebra_display_name(t)
            else:
                t_disp = t
            if "costela" in t_disp.lower():
                t_disp = t_disp.replace("costela", "cost")
            if key == "corpo":
                add_structure_and_obs(number, t_disp, generation="MANUAL", interpreted_type="EXTERNAL")
            else:
                add_structure_and_obs(number, t_disp, generation="MANUAL", interpreted_type="ORGAN")

    # Terceira parte: construir ROIContourSequence
    exclude_individual_ribs = "costelas" in created_items
    for typ, key, region in final_order:
        if typ != "orig":
            continue
        if exclude_individual_ribs and (key in rib_keys):
            continue
        entry = roi_map_all.get(key)
        if not entry:
            continue
        rep = list(entry["original_names"])[0] if entry["original_names"] else key
        if translate_name(rep) == "mama":
            continue
        sample = entry["samples"][0] if entry["samples"] else None
        if sample is None:
            continue
        src_ds, src_num, roi_ds = sample
        if hasattr(src_ds, "ROIContourSequence"):
            for roi_contour in src_ds.ROIContourSequence:
                if getattr(roi_contour, "ReferencedROINumber", None) != src_num:
                    continue
                new_roi_contour = copy.deepcopy(roi_contour)
                new_roi_contour.ReferencedROINumber = roi_number_map.get((typ, key))
                if not hasattr(new_roi_contour, "ROIDisplayColor"):
                    new_roi_contour.ROIDisplayColor = [0, 255, 0]
                if hasattr(new_roi_contour, "ContourSequence"):
                    valid_contours = []
                    for c in new_roi_contour.ContourSequence:
                        if hasattr(c, "ContourData") and c.ContourData:
                            valid_contours.append(c)
                    if valid_contours:
                        new_roi_contour.ContourSequence = valid_contours
                        ds_base.ROIContourSequence.append(new_roi_contour)

    # Adicionar criados (pulmao_E, pulmao_D, pulmoes, medula_PRV, costelas)
    for typ, key, region in final_order:
        if typ != "created":
            continue
        item = created_items.get(key)
        if not item:
            continue
        refnum = roi_number_map.get((typ, key))
        rc = create_roi_contour_item_from_grouped_polys(item["grouped"], refnum, color_rgb=item.get("color"), sample_roi_contour_for_images=item.get("sample_roi_contour"))
        if hasattr(rc, "ContourSequence") and rc.ContourSequence:
            ds_base.ROIContourSequence.append(rc)

    # Metadados para compatibilidade XiO
    try:
        ds_base.SOPClassUID = RTSTRUCT_SOPCLASS
        ds_base.SOPInstanceUID = generate_uid()
        ds_base.SeriesInstanceUID = generate_uid()
        ds_base.StructureSetLabel = getattr(ds_base, "StructureSetLabel", "XiO_Converted")
    except Exception:
        pass

    # Salvar
    try:
        ds_base.save_as(out_name)
        if debug:
            print(f"Arquivo salvo: {out_name}")
            print("ROIs criadas:", list(created_items.keys()))
            print("Ordem final:", final_order)
    except Exception as e:
        print(f"Erro ao salvar {out_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
