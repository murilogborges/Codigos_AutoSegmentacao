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
import math
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

# órgãos
"spleen":"baco",
"kidney_right":"rim_D",
"kidney_left":"rim_E",
"gallbladder":"vesicula",
"liver":"figado",
"stomach":"estomago",
"pancreas":"pancreas",

# adrenais
"adrenal_gland_right":"supra_D",
"adrenal_gland_left":"supra_E",

# pulmões
"lung_left":"pulmao_E",
"lung_right":"pulmao_D",
"lung":"pulmao",

"lung_upper_lobe_left":"pulmao_sup_E",
"lung_lower_lobe_left":"pulmao_inf_E",

"lung_upper_lobe_right":"pulmao_sup_D",
"lung_middle_lobe_right":"pulmao_med_D",
"lung_lower_lobe_right":"pulmao_inf_D",

# digestivo
"small_bowel":"int_delgado",
"duodenum":"duodeno",
"colon":"colon",

# urinário
"urinary_bladder":"bexiga",
"prostate":"prostata",

"kidney_cyst_left":"cisto_rim_E",
"kidney_cyst_right":"cisto_rim_D",

# coração
"heart":"coracao",
"heart_myocardium":"miocardio",
"heart_atrium_left":"atri_E",
"heart_atrium_right":"atri_D",
"heart_ventricle_left":"ventri_E",
"heart_ventricle_right":"ventri_D",

# vasos
"aorta":"aorta",
"pulmonary_artery":"art_pulm",
"pulmonary_vein":"veia_pulm",

"superior_vena_cava":"vc_sup",
"inferior_vena_cava":"vc_inf",

"portal_vein_and_splenic_vein":"v_portoespl",

"iliac_artery_left":"art_ilia_E",
"iliac_artery_right":"art_ilia_D",

"iliac_vena_left":"v_ilia_E",
"iliac_vena_right":"v_ilia_D",

"brachiocephalic_trunk":"tronco_brac",

"subclavian_artery_left":"subclav_E",
"subclavian_artery_right":"subclav_D",

"common_carotid_artery_left":"carot_com_E",
"common_carotid_artery_right":"carot_com_D",

"brachiocephalic_vein_left":"v_brac_E",
"brachiocephalic_vein_right":"v_brac_D",

"atrial_appendage_left":"auric_E",

# SNC
"brain":"cerebro",
"brainstem":"tronco_enc",
"skull":"cranio",

# vias aéreas
"trachea":"traqueia",
"esophagus":"esofago",
"thyroid_gland":"tireoide",

# ossos
"humerus_left":"umero_E",
"humerus_right":"umero_D",

"scapula_left":"escapula_E",
"scapula_right":"escapula_D",

"clavicula_left":"clavicula_E",
"clavicula_right":"clavicula_D",

"femur_left":"femur_E",
"femur_right":"femur_D",

"hip_left":"quadril_E",
"hip_right":"quadril_D",

"sacrum":"sacro",

# medula
"spinal_cord":"medula",

# glúteos
"gluteus_maximus_left":"glutmax_E",
"gluteus_maximus_right":"glutmax_D",

"gluteus_medius_left":"glutmed_E",
"gluteus_medius_right":"glutmed_D",

"gluteus_minimus_left":"glutmin_E",
"gluteus_minimus_right":"glutmin_D",

# músculos
"autochthon_left":"autoct_E",
"autochthon_right":"autoct_D",

"iliopsoas_left":"iliopsoas_E",
"iliopsoas_right":"iliopsoas_D",

# costelas e cartilagens
"sternum":"esterno",
"costal_cartilages":"cartil_cost",

# pulmão vasos
"lung_airways":"vias_aereas",
"lung_airways_wall":"parede_va",
"lung_arteries":"art_pulmres",
"lung_veins":"veias_pulm",
"lung_vessels":"vasos_pulm",
"lung_trachea_bronchia":"traq_bronq",

# covid
"lung_covid_infiltrate":"infl_covid",

# hemorragia
"intracerebral_hemorrhage":"hem_cereb",

# implantes
"hip_implant":"imp_quadril",

# coronárias
"coronary_arteries":"art_coron",

# corpo
"body_trunc":"corpo_tronco",
"body_extremities":"extremid",

# tecidos
"subcutaneous_fat":"gord_subcut",
"torso_fat":"gord_torax",
"skeletal_muscle":"musculo",
"intermuscular_fat":"gord_inter",

# pleura
"lung_pleural":"pleura",
"pleural_effusion":"derr_pleura",
"pericardial_effusion":"derr_peric",

# fígado
"liver_vessels":"vasos_fig",
"liver_tumor":"tumor_fig",

"liver_segment_1":"seg_fig_1",
"liver_segment_2":"seg_fig_2",
"liver_segment_3":"seg_fig_3",
"liver_segment_4":"seg_fig_4",
"liver_segment_5":"seg_fig_5",
"liver_segment_6":"seg_fig_6",
"liver_segment_7":"seg_fig_7",
"liver_segment_8":"seg_fig_8",

"liver_lesions":"lesoes_fig",

# vértebras especiais
"vertebrae":"vertebras",
"vertebrae_body":"corpo_vert",
"intervertebral_discs":"disc_iv",

"vertebrae_s1":"vertebra_S1",
"vertebrae_l1":"vertebra_L1",
"vertebrae_l2":"vertebra_L2",
"vertebrae_l3":"vertebra_L3",
"vertebrae_l4":"vertebra_L4",
"vertebrae_l5":"vertebra_L5",
"vertebrae_l6":"vertebra_L6",

"vertebrae_t1":"vertebra_T1",
"vertebrae_t2":"vertebra_T2",
"vertebrae_t3":"vertebra_T3",
"vertebrae_t4":"vertebra_T4",
"vertebrae_t5":"vertebra_T5",
"vertebrae_t6":"vertebra_T6",
"vertebrae_t7":"vertebra_T7",
"vertebrae_t8":"vertebra_T8",
"vertebrae_t9":"vertebra_T9",
"vertebrae_t10":"vertebra_T10",
"vertebrae_t11":"vertebra_T11",
"vertebrae_t12":"vertebra_T12",

"vertebrae_c1":"vertebra_C1",
"vertebrae_c2":"vertebra_C2",
"vertebrae_c3":"vertebra_C3",
"vertebrae_c4":"vertebra_C4",
"vertebrae_c5":"vertebra_C5",
"vertebrae_c6":"vertebra_C6",
"vertebrae_c7":"vertebra_C7",

# olhos
"eye_left":"olho_E",
"eye_right":"olho_D",

"eye_lens_left":"cristal_E",
"eye_lens_right":"cristal_D",

"optic_nerve_left":"nervo_opt_E",
"optic_nerve_right":"nervo_opt_D",

# glândulas
"parotid_gland_left":"parotida_E",
"parotid_gland_right":"parotida_D",

"submandibular_gland_left":"submand_E",
"submandibular_gland_right":"submand_D",

# faringe
"nasopharynx":"nasofaringe",
"oropharynx":"orofaringe",
"hypopharynx":"hipofaringe",

# cavidades nasais
"nasal_cavity_left":"cav_nasal_E",
"nasal_cavity_right":"cav_nasal_D",

# ouvido
"auditory_canal_left":"can_aud_E",
"auditory_canal_right":"can_aud_D",

# palato
"soft_palate":"palato_mol",
"hard_palate":"palato_dur",

# laringe
"larynx_air":"laringe_ar",
"thyroid_cartilage":"cart_tireo",
"hyoid":"hioide",
"cricoid_cartilage":"cart_crico",

# zigoma
"zygomatic_arch_left":"zigoma_E",
"zygomatic_arch_right":"zigoma_D",

# estiloide
"styloid_process_left":"estiloide_E",
"styloid_process_right":"estiloide_D",

# carótidas
"internal_carotid_artery_left":"carot_int_E",
"internal_carotid_artery_right":"carot_int_D",

# jugulares
"internal_jugular_vein_left":"jugular_E",
"internal_jugular_vein_right":"jugular_D",

# mastigação
"masseter_left":"masseter_E",
"masseter_right":"masseter_D",

"temporalis_left":"temporal_E",
"temporalis_right":"temporal_D",

"lateral_pterygoid_left":"pterig_lat_E",
"lateral_pterygoid_right":"pterig_lat_D",

"medial_pterygoid_left":"pterig_med_E",
"medial_pterygoid_right":"pterig_med_D",

"tongue":"lingua",

"digastric_left":"digastr_E",
"digastric_right":"digastr_D",

# pescoço
"sternocleidomastoid_left":"ecm_E",
"sternocleidomastoid_right":"ecm_D",

"superior_pharyngeal_constrictor":"const_far_sup",
"middle_pharyngeal_constrictor":"const_far_med",
"inferior_pharyngeal_constrictor":"const_far_inf",

"trapezius":"trapezio",
"trapezius_left":"trap_E",
"trapezius_right":"trap_D",

"platysma_left":"platisma_E",
"platysma_right":"platisma_D",

"levator_scapulae_left":"lev_esc_E",
"levator_scapulae_right":"lev_esc_D",

"anterior_scalene_left":"escal_ant_E",
"anterior_scalene_right":"escal_ant_D",

"middle_scalene_left":"escal_med_E",
"middle_scalene_right":"escal_med_D",

"posterior_scalene_left":"escal_pos_E",
"posterior_scalene_right":"escal_pos_D",

"sterno_thyroid_left":"estern_tir_E",
"sterno_thyroid_right":"estern_tir_D",

"thyrohyoid_left":"tireo_hio_E",
"thyrohyoid_right":"tireo_hio_D",

"prevertebral_left":"prevert_E",
"prevertebral_right":"prevert_D",

# ombro/coxa
"quadriceps_femoris_left":"quadric_E",
"quadriceps_femoris_right":"quadric_D",

"thigh_medial_compartment_left":"coxa_med_E",
"thigh_medial_compartment_right":"coxa_med_D",

"thigh_posterior_compartment_left":"coxa_pos_E",
"thigh_posterior_compartment_right":"coxa_pos_D",

"sartorius_left":"sartorio_E",
"sartorius_right":"sartorio_D",

"deltoid":"deltoide",
"supraspinatus":"supraespin",
"infraspinatus":"infraespin",
"subscapularis":"subescap",
"coracobrachial":"coracobr",
"pectoralis_minor":"peit_menor",
"serratus_anterior":"serr_ant",
"teres_major":"redond_mai",
"triceps_brachii":"triceps",

# ossos apendiculares
"patella":"patela",
"tibia":"tibia",
"fibula":"fibula",
"tarsal":"tarso",
"metatarsal":"metatarso",
"phalanges_feet":"falanges_pe",

"ulna":"ulna",
"radius":"radio",

"carpal":"carpo",
"metacarpal":"metacarpo",
"phalanges_hand":"falanges_mao",

# neuro
"subarachnoid_space":"esp_subarac",
"venous_sinuses":"seios_ven",
"septum_pellucidum":"septo_pel",
"cerebellum":"cerebelo",
"caudate_nucleus":"nuc_caud",
"lentiform_nucleus":"nuc_lent",
"insular_cortex":"cortex_ins",
"internal_capsule":"caps_int",
"ventricle":"ventriculo",
"central_sulcus":"sulco_cent",
"frontal_lobe":"lobo_front",
"parietal_lobe":"lobo_pariet",
"occipital_lobe":"lobo_occ",
"temporal_lobe":"lobo_temp",
"thalamus":"talamo",

# cavidades
"abdominal_cavity":"cav_abdom",
"thoracic_cavity":"cav_torax",
"pericardium":"pericardio",
"mediastinum":"mediastino",

# aneurisma
"brain_aneurysm":"aneur_cerb",

# craniofacial
"mandible":"mandibula",
"head":"cabeca",
"sinus_maxillary":"seio_max",
"sinus_frontal":"seio_front",
"teeth_lower":"dentes_inf",
"teeth_upper":"dentes_sup",

# mama
"breast":"mama"

}

# ---------------------------
# Cores (RGB)
# ---------------------------
COLOR_PULMAO_E = (0, 200, 0)
COLOR_PULMAO_D = (0, 0, 200)
COLOR_PULMOES = (135, 206, 250)
COLOR_MEDULA_PRV = (255, 165, 0)
COLOR_COSTELAS = (220, 220, 220)
COLOR_MAMA_D = (80, 120, 220)
COLOR_MAMA_E = (220, 100, 160)
COLOR_RETO = (101, 50, 20)
COLOR_SIGMOIDE = (204, 85, 0)
LEFT_STRUCTURE_COLORS = (
    (0, 70, 160), (0, 105, 210), (25, 135, 190), (45, 85, 180),
    (70, 150, 220), (20, 170, 200), (90, 100, 210), (0, 145, 170),
)
RIGHT_STRUCTURE_COLORS = (
    (0, 100, 40), (0, 145, 60), (20, 170, 75), (55, 125, 35),
    (80, 180, 90), (0, 125, 105), (105, 155, 35), (35, 190, 120),
)
ANATOMICAL_COLORS = {
    "corpo": (238, 190, 120),
    "colon": (150, 75, 0),
    "sacro": (150, 150, 150),
    "spinal_cord": (255, 215, 0),
    "heart": (210, 0, 0),
    "liver": (125, 45, 25),
    "kidney": (180, 70, 100),
    "lung": (100, 200, 220),
    "brain": (210, 170, 210),
    "bone": (190, 190, 175),
}
NEUTRAL_STRUCTURE_COLORS = (
    (120, 80, 150), (160, 100, 45), (90, 120, 140), (190, 110, 150),
    (110, 160, 160), (145, 125, 70), (80, 80, 120), (170, 145, 100),
)
THORACIC_VERTEBRA_COLORS = (
    (30, 90, 180), (0, 145, 210), (0, 175, 190), (55, 125, 220),
    (85, 80, 190), (0, 120, 165), (70, 165, 220), (25, 150, 145),
    (100, 105, 210), (0, 160, 175), (45, 110, 205), (80, 145, 195),
)
CERVICAL_VERTEBRA_COLORS = (
    (0, 85, 35), (0, 120, 50), (20, 155, 60), (55, 185, 75),
    (75, 125, 45), (0, 145, 105), (80, 175, 100),
)
LUMBAR_VERTEBRA_COLORS = (
    (180, 45, 25), (220, 75, 15), (200, 105, 0), (235, 125, 20),
    (165, 55, 50), (210, 80, 55), (190, 130, 25),
)
SACRAL_STRUCTURE_COLORS = (
    (125, 55, 135), (155, 75, 155), (180, 95, 130), (110, 70, 145),
    (145, 90, 110), (170, 65, 120),
)

MUSCLE_ROI_KEYS = {
    "autochthon_left", "autochthon_right",
    "iliopsoas_left", "iliopsoas_right",
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right",
    "masseter_left", "masseter_right",
    "temporalis_left", "temporalis_right",
    "lateral_pterygoid_left", "lateral_pterygoid_right",
    "medial_pterygoid_left", "medial_pterygoid_right",
    "digastric_left", "digastric_right",
    "trapezius", "trapezius_left", "trapezius_right",
    "platysma_left", "platysma_right",
    "levator_scapulae_left", "levator_scapulae_right",
    "anterior_scalene_left", "anterior_scalene_right",
    "middle_scalene_left", "middle_scalene_right",
    "posterior_scalene_left", "posterior_scalene_right",
    "thyrohyoid_left", "thyrohyoid_right",
    "prevertebral_left", "prevertebral_right",
    "quadriceps_femoris_left", "quadriceps_femoris_right",
    "thigh_medial_compartment_left", "thigh_medial_compartment_right",
    "thigh_posterior_compartment_left", "thigh_posterior_compartment_right",
    "sartorius_left", "sartorius_right",
    "deltoid", "supraspinatus", "infraspinatus", "subscapularis",
    "coracobrachial", "pectoralis_minor", "serratus_anterior",
    "teres_major", "triceps_brachii", "skeletal_muscle",
}

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

def colon_components_by_z(
    colon_polys: List[Tuple[float, Polygon]],
) -> Dict[float, List[Polygon]]:
    components_by_z = defaultdict(list)
    for z, poly in colon_polys:
        if poly is None or poly.is_empty:
            continue
        geometry = poly if isinstance(poly, Polygon) else unary_union(poly)
        geometries = (
            [geometry]
            if isinstance(geometry, Polygon)
            else [item for item in geometry.geoms if isinstance(item, Polygon)]
        )
        components_by_z[round(float(z), 3)].extend(geometries)
    return dict(components_by_z)

def component_similarity(previous: Polygon, candidate: Polygon) -> float:
    previous_buffer = previous.buffer(3.0)
    candidate_buffer = candidate.buffer(3.0)
    union_area = previous_buffer.union(candidate_buffer).area
    overlap = (
        previous_buffer.intersection(candidate_buffer).area / union_area
        if union_area > 0
        else 0.0
    )
    centroid_distance = previous.centroid.distance(candidate.centroid)
    area_ratio = min(previous.area, candidate.area) / max(previous.area, candidate.area)
    return overlap * 5.0 + area_ratio * 2.0 - centroid_distance / 25.0

def track_colon_component_path(
    colon_polys: List[Tuple[float, Polygon]],
) -> Dict[float, Polygon]:
    """Track one continuous colon component from caudal to cranial slices."""
    components_by_z = colon_components_by_z(colon_polys)
    if not components_by_z:
        return {}

    ordered_z = sorted(components_by_z)
    path = {}
    previous = max(
        components_by_z[ordered_z[0]],
        key=lambda component: component.area,
    )
    path[ordered_z[0]] = previous

    for z in ordered_z[1:]:
        candidates = components_by_z[z]
        if not candidates:
            continue
        previous = max(
            candidates,
            key=lambda candidate: component_similarity(previous, candidate),
        )
        path[z] = previous
    return path

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

def find_body_roi_key(roi_polygons: Dict[str, List[Tuple[float, Polygon, Dataset, int]]]) -> Optional[str]:
    for key in ("corpo", "body"):
        if key in roi_polygons:
            return key
    return None

def is_muscle_roi(key: str, original_names: Optional[set] = None) -> bool:
    if key in MUSCLE_ROI_KEYS:
        return True
    names = original_names or set()
    normalized_names = {normalize_name(name) for name in names}
    return bool(normalized_names & MUSCLE_ROI_KEYS)

def structure_side(name: str) -> Optional[str]:
    normalized = normalize_name(name)
    if normalized.endswith("_e"):
        return "left"
    if normalized.endswith("_d"):
        return "right"
    return None

def choose_structure_color(
    key: str,
    display_name: str,
    used_colors: set,
    side_indexes: Dict[str, int],
) -> Tuple[int, int, int]:
    if key in ("breast_right", "breast_left"):
        return (0, 0, 0)

    normalized_key = normalize_name(key)
    normalized_display = normalize_name(display_name)
    vertebra_palette = None
    is_thoracic_display = (
        normalized_display.startswith("t")
        and normalized_display[1:].isdigit()
    )
    is_lumbar_display = (
        normalized_display.startswith("l")
        and normalized_display[1:].isdigit()
    )
    is_sacral_display = (
        normalized_display.startswith("s")
        and normalized_display[1:].isdigit()
    )
    is_cervical_display = (
        normalized_display.startswith("c")
        and normalized_display[1:].isdigit()
    )
    if normalized_key.startswith("vertebrae_c") or is_cervical_display:
        vertebra_palette = CERVICAL_VERTEBRA_COLORS
    elif normalized_key.startswith("vertebrae_t") or is_thoracic_display:
        vertebra_palette = THORACIC_VERTEBRA_COLORS
    elif normalized_key.startswith("vertebrae_l") or is_lumbar_display:
        vertebra_palette = LUMBAR_VERTEBRA_COLORS
    elif normalized_key.startswith("vertebrae_s") or is_sacral_display:
        vertebra_palette = SACRAL_STRUCTURE_COLORS
    elif normalized_key in ("sacro", "sacrum"):
        vertebra_palette = SACRAL_STRUCTURE_COLORS
    if vertebra_palette is not None:
        for color in vertebra_palette:
            if color not in used_colors:
                return color

    if normalized_key == "urinary_bladder" or normalized_display == "bexiga":
        bladder_color = (255, 220, 0)
        if bladder_color not in used_colors:
            return bladder_color

    side = structure_side(display_name) or structure_side(key)
    if side is not None:
        palette = LEFT_STRUCTURE_COLORS if side == "left" else RIGHT_STRUCTURE_COLORS
        start = side_indexes[side]
        for offset in range(len(palette)):
            color = palette[(start + offset) % len(palette)]
            if color not in used_colors:
                side_indexes[side] = (start + offset + 1) % len(palette)
                return color

    for anatomy_key, color in ANATOMICAL_COLORS.items():
        if anatomy_key in key or anatomy_key in normalize_name(display_name):
            if color not in used_colors:
                return color

    for color in NEUTRAL_STRUCTURE_COLORS:
        if color not in used_colors:
            return color

    # The palettes above provide enough colors for the expected RTSTRUCTs.
    # This deterministic fallback keeps colors unique for unusually large sets.
    candidate = tuple((37 * (len(used_colors) + channel + 1)) % 256 for channel in range(3))
    while candidate in used_colors:
        candidate = tuple((value + 17) % 256 for value in candidate)
    return candidate

def find_grouped_polygon_at_z(grouped: Dict[float, Polygon],
                              z: float,
                              tolerance_mm: float = 1.0) -> Optional[Polygon]:
    polygon = grouped.get(z)
    if polygon is not None:
        return polygon
    nearest_z = min(grouped, key=lambda candidate: abs(candidate - z), default=None)
    if nearest_z is not None and abs(nearest_z - z) <= tolerance_mm:
        return grouped[nearest_z]
    return None

def centroid_z_range(polygons: List[Tuple[float, Polygon]]) -> Tuple[float, float]:
    zs = [float(z) for z, _ in polygons]
    return min(zs), max(zs)

def estimate_colorectal_transition_z(
    roi_polygons: Dict[str, List[Tuple[float, Polygon, Dataset, int]]],
    colon_polys: List[Tuple[float, Polygon]],
) -> Tuple[float, str]:
    """Estimate the rectosigmoid transition from the 3-D colon trajectory.

    The transition is selected where the colon leaves the sacral midline and
    develops a persistent lateral displacement or turn. Sacral structures are
    used to define a patient-specific axis; z is used only to order slices.
    """
    grouped_colon = polygons_group_by_z(colon_polys)
    trajectory = sorted(
        (z, poly.centroid.x, poly.centroid.y)
        for z, poly in grouped_colon.items()
        if poly is not None and not poly.is_empty
    )
    if len(trajectory) < 3:
        colon_min, colon_max = centroid_z_range(colon_polys)
        return colon_min + 0.40 * (colon_max - colon_min), "colon_fallback_short"

    landmark_keys = ("vertebrae_s3", "vertebra_s3", "sacro", "sacrum")
    landmarks = []
    landmark_source = "colon_axis"
    for key in landmark_keys:
        if key not in roi_polygons:
            continue
        grouped = polygons_group_by_z([
            (z, poly) for z, poly, *_ in roi_polygons[key]
        ])
        landmarks = sorted(
            (z, poly.centroid.x, poly.centroid.y)
            for z, poly in grouped.items()
            if poly is not None and not poly.is_empty
        )
        if landmarks:
            landmark_source = key
            break

    if landmarks:
        landmark_z = [item[0] for item in landmarks]
        landmark_x = [item[1] for item in landmarks]
        landmark_y = [item[2] for item in landmarks]
        if len(landmarks) >= 2:
            axis_x = np.polyfit(landmark_z, landmark_x, 1)
            axis_y = np.polyfit(landmark_z, landmark_y, 1)
            axis_at = lambda z: (
                float(np.polyval(axis_x, z)),
                float(np.polyval(axis_y, z)),
            )
        else:
            axis_at = lambda z: (landmark_x[0], landmark_y[0])
        anchor_z = float(np.median(landmark_z))
    else:
        colon_z = [item[0] for item in trajectory]
        colon_x = [item[1] for item in trajectory]
        colon_y = [item[2] for item in trajectory]
        axis_x = np.polyfit(colon_z, colon_x, 1)
        axis_y = np.polyfit(colon_z, colon_y, 1)
        axis_at = lambda z: (
            float(np.polyval(axis_x, z)),
            float(np.polyval(axis_y, z)),
        )
        anchor_z = float(np.median(colon_z))

    # Median smoothing suppresses isolated contour artifacts without changing
    # the slice grid or the original contour geometry.
    smoothed = []
    for index, (z, x, y) in enumerate(trajectory):
        window = trajectory[max(0, index - 2):min(len(trajectory), index + 3)]
        smoothed.append((
            z,
            float(np.median([item[1] for item in window])),
            float(np.median([item[2] for item in window])),
        ))

    distances = []
    for z, x, y in smoothed:
        axis_x_at, axis_y_at = axis_at(z)
        distances.append(math.hypot(x - axis_x_at, y - axis_y_at))

    turns = [0.0] * len(smoothed)
    for index in range(1, len(smoothed) - 1):
        before = np.subtract(smoothed[index], smoothed[index - 1])[1:]
        after = np.subtract(smoothed[index + 1], smoothed[index])[1:]
        before_norm = np.linalg.norm(before)
        after_norm = np.linalg.norm(after)
        if before_norm > 0 and after_norm > 0:
            cosine = float(np.dot(before, after) / (before_norm * after_norm))
            turns[index] = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))

    # Require the lateral/curvature change to persist for at least three
    # slices, avoiding classification of isolated sigmoid loops as transition.
    candidates = []
    for index in range(1, len(smoothed) - 2):
        persistent = all(
            distances[offset] >= 25.0 or turns[offset] >= 25.0
            for offset in range(index, min(index + 3, len(smoothed)))
        )
        if persistent:
            candidates.append(index)

    if candidates:
        index = min(candidates, key=lambda item: abs(smoothed[item][0] - anchor_z))
        transition_z = (smoothed[index - 1][0] + smoothed[index][0]) / 2.0
        return transition_z, f"{landmark_source}_trajectory"

    if landmarks:
        return anchor_z, f"{landmark_source}_fallback"

    colon_min, colon_max = centroid_z_range(colon_polys)
    return colon_min + 0.40 * (colon_max - colon_min), "colon_fallback"

def estimate_sigmoid_cranial_limit_z(
    roi_polygons: Dict[str, List[Tuple[float, Polygon, Dataset, int]]],
    colon_polys: List[Tuple[float, Polygon]],
    margin_mm: float = 10.0,
) -> Tuple[float, str]:
    """Use the last cranial sacral slice as the sigmoid cranial limit."""
    for key in ("sacro", "sacrum"):
        if key not in roi_polygons:
            continue
        sacrum_zs = [float(z) for z, *_ in roi_polygons[key]]
        if sacrum_zs:
            return max(sacrum_zs), key

    _, colon_max = centroid_z_range(colon_polys)
    return colon_max, "colon_fallback"

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

def get_roi_display_color(roi_contour: Optional[Dataset],
                          fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    color = getattr(roi_contour, "ROIDisplayColor", None) if roi_contour is not None else None
    if color is None or len(color) < 3:
        return fallback
    return (int(color[0]), int(color[1]), int(color[2]))

def lighten_color(color: Tuple[int, int, int], amount: float = 0.45) -> Tuple[int, int, int]:
    amount = min(max(amount, 0.0), 1.0)
    return tuple(
        int(round(channel + (255 - channel) * amount))
        for channel in color
    )

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
            if not isinstance(g, Polygon):
                continue
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

    # Avaliação das mamas: interseção com o corpo erodido em 4 mm.
    # O corpo erodido é mantido somente em memória e não é exportado.
    eroded_body = {}
    body_key = find_body_roi_key(roi_polygons)
    if body_key is not None:
        body_polys = [(z, poly) for (z, poly, *rest) in roi_polygons[body_key]]
        eroded_body = erode_grouped_by_margin(
            polygons_group_by_z(body_polys),
            margin_mm=4.0,
        )

    for breast_key, evaluation_name in (
        ("breast_right", "mama_D_aval"),
        ("breast_left", "mama_E_aval"),
    ):
        if breast_key not in roi_polygons:
            if debug and body_key is not None:
                print(f"Aviso: ROI ausente para {evaluation_name}: {breast_key}.")
            continue

        breast_polys = [(z, poly) for (z, poly, *rest) in roi_polygons[breast_key]]
        grouped_breast = polygons_group_by_z(breast_polys)
        evaluated_breast = {}
        for z, breast_poly in grouped_breast.items():
            body_poly = find_grouped_polygon_at_z(eroded_body, z)
            if body_poly is None:
                continue
            intersection = breast_poly.intersection(body_poly)
            if intersection is not None and not intersection.is_empty:
                evaluated_breast[z] = intersection

        if not evaluated_breast:
            if debug:
                reason = "corpo ausente" if body_key is None else "interseção vazia"
                print(
                    f"Aviso: não foi possível criar {evaluation_name} ({reason}); "
                    f"fatias mama={len(grouped_breast)}, corpo_erodido={len(eroded_body)}."
                )
            continue

        sample_breast = None
        if roi_map_all.get(breast_key, {}).get("samples"):
            s_ds, s_num, _ = roi_map_all[breast_key]["samples"][0]
            if hasattr(s_ds, "ROIContourSequence"):
                for rc in s_ds.ROIContourSequence:
                    if getattr(rc, "ReferencedROINumber", None) == s_num:
                        sample_breast = rc
                        break
        created_items[evaluation_name] = {
            "grouped": evaluated_breast,
            "color": lighten_color(
                get_roi_display_color(
                    sample_breast,
                    COLOR_MAMA_D if breast_key == "breast_right" else COLOR_MAMA_E,
                )
            ),
            "label": evaluation_name,
            "sample_roi_contour": sample_breast
        }

    # Separação anatômico-geométrica do cólon em reto e sigmoide.
    if "colon" in roi_polygons:
        colon_polys = [(z, poly) for (z, poly, *rest) in roi_polygons["colon"]]
        grouped_colon = polygons_group_by_z(colon_polys)
        tracked_colon = track_colon_component_path(colon_polys)
        transition_z, transition_source = estimate_colorectal_transition_z(
            roi_polygons,
            colon_polys,
        )
        cranial_limit_z, cranial_limit_source = estimate_sigmoid_cranial_limit_z(
            roi_polygons,
            colon_polys,
            margin_mm=10.0,
        )
        rectum_grouped = {
            z: poly for z, poly in tracked_colon.items()
            if z <= transition_z
        }
        sigmoid_grouped = {
            z: poly for z, poly in tracked_colon.items()
            if transition_z < z <= cranial_limit_z
        }
        sample_colon = find_sample_roi_contour_for_keys(["colon"])
        if rectum_grouped:
            created_items["reto"] = {
                "grouped": rectum_grouped,
                "color": COLOR_RETO,
                "label": "reto",
                "sample_roi_contour": sample_colon
            }
        if sigmoid_grouped:
            created_items["sigmoide"] = {
                "grouped": sigmoid_grouped,
                "color": COLOR_SIGMOIDE,
                "label": "sigmoide",
                "sample_roi_contour": sample_colon
            }
        if debug:
            uncertain = sorted(
                z for z in grouped_colon
                if abs(z - transition_z) <= 5.0
            )
            print(
                f"Separação colon: transição z={transition_z:.3f} mm "
                f"(fonte={transition_source}), reto={len(rectum_grouped)} "
                f"fatias, sigmoide={len(sigmoid_grouped)} fatias, "
                f"limite_cranial_sigmoide={cranial_limit_z:.3f} mm "
                f"(fonte={cranial_limit_source}), "
                f"zona_incerteza={len(uncertain)} fatias, "
                f"componentes_rastreados={len(tracked_colon)}."
            )

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

    # 2. mamas: renomear as estruturas originais e incluir as avaliações
    for key in ("breast_right", "breast_left"):
        if key in roi_map_all:
            final_order.append(("orig", key, None))
    for name in ("mama_D_aval", "mama_E_aval"):
        if name in created_items:
            final_order.append(("created", name, None))
    for name in ("reto", "sigmoide"):
        if name in created_items:
            final_order.append(("created", name, None))

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
        if key not in ("breast_right", "breast_left") and translate_name(rep) == "mama":
            continue
        remaining.append(k)
    remaining_sorted = sorted(remaining, key=lambda x: translated_map.get(x, x))
    for k in remaining_sorted:
        final_order.append(("orig", k, None))

    muscle_keys_removed = {
        key for key, entry in roi_map_all.items()
        if is_muscle_roi(key, entry.get("original_names"))
    }
    if muscle_keys_removed:
        final_order = [
            item for item in final_order
            if not (item[0] == "orig" and item[1] in muscle_keys_removed)
        ]
        if debug:
            print(
                "Estruturas musculares removidas:",
                sorted(muscle_keys_removed),
            )

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
            if key not in ("breast_right", "breast_left") and translate_name(orig_name) == "mama":
                continue
            t = translate_name(orig_name)
            if key == "breast_right":
                t_disp = "mama_D"
            elif key == "breast_left":
                t_disp = "mama_E"
            elif t.lower().startswith("vertebra"):
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
    used_colors = {
        tuple(item["color"])
        for item in created_items.values()
        if item.get("color") is not None
    }
    side_indexes = {"left": 0, "right": 0}
    for typ, key, region in final_order:
        if typ != "orig":
            continue
        if exclude_individual_ribs and (key in rib_keys):
            continue
        entry = roi_map_all.get(key)
        if not entry:
            continue
        rep = list(entry["original_names"])[0] if entry["original_names"] else key
        if key not in ("breast_right", "breast_left") and translate_name(rep) == "mama":
            continue
        translated = translate_name(rep)
        if key == "breast_right":
            display_name = "mama_D"
        elif key == "breast_left":
            display_name = "mama_E"
        else:
            display_name = translated
        source_color = None
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
                if key in ("breast_right", "breast_left"):
                    source_color = get_roi_display_color(
                        roi_contour,
                        COLOR_MAMA_D if key == "breast_right" else COLOR_MAMA_E,
                    )
                    color = source_color
                elif key == "corpo":
                    color = get_roi_display_color(
                        roi_contour,
                        ANATOMICAL_COLORS["corpo"],
                    )
                else:
                    color = choose_structure_color(
                        key,
                        display_name,
                        used_colors,
                        side_indexes,
                    )
                new_roi_contour.ROIDisplayColor = [int(channel) for channel in color]
                used_colors.add(color)
                if hasattr(new_roi_contour, "ContourSequence"):
                    valid_contours = []
                    for c in new_roi_contour.ContourSequence:
                        if hasattr(c, "ContourData") and c.ContourData:
                            valid_contours.append(c)
                    if valid_contours:
                        new_roi_contour.ContourSequence = valid_contours
                        ds_base.ROIContourSequence.append(new_roi_contour)

    # Adicionar criados (mama_D_aval, pulmao_E, pulmao_D, pulmoes, medula_PRV, costelas)
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
            print(
                "Avaliações de mama exportadas:",
                [name for name in ("mama_D_aval", "mama_E_aval") if name in created_items],
            )
            print("Ordem final:", final_order)
    except Exception as e:
        print(f"Erro ao salvar {out_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
