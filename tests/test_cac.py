import tempfile
import unittest
from pathlib import Path
import base64
import re
from unittest.mock import patch
import zlib

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTDoseStorage,
    RTStructureSetStorage,
    generate_uid,
)

import CAC


def _new_file(path, sop_class, modality):
    instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = instance_uid
    dataset.Modality = modality
    return dataset


def _write_ct(path, series_uid, frame_uid, z, value, slope=2, intercept=-100,
              sop_class=CTImageStorage):
    dataset = _new_file(path, sop_class, "CT")
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.StudyDate = "20240630"
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.PixelSpacing = [1.0, 2.0]
    dataset.SliceThickness = 2.0
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.PixelData = np.full((4, 4), value, dtype="<i2").tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _write_rtstruct(path, series_uid, frame_uid, contour_data, roi_name="coronary"):
    dataset = _new_file(path, RTStructureSetStorage, "RTSTRUCT")
    dataset.StructureSetROISequence = Sequence()
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = roi_name
    dataset.StructureSetROISequence.append(roi)

    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = len(contour_data) // 3
    contour.ContourData = contour_data
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = Sequence([contour])
    dataset.ROIContourSequence = Sequence([roi_contour])

    frame = Dataset()
    frame.FrameOfReferenceUID = frame_uid
    study = Dataset()
    series = Dataset()
    series.SeriesInstanceUID = series_uid
    study.RTReferencedSeriesSequence = Sequence([series])
    frame.RTReferencedStudySequence = Sequence([study])
    dataset.ReferencedFrameOfReferenceSequence = Sequence([frame])
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def _write_rt_dose(
    path,
    frame_uid,
    pixels,
    units="GY",
    scaling=1.0,
    origin=(0, 0, 0),
    frame_offsets=None,
):
    dataset = _new_file(path, RTDoseStorage, "RTDOSE")
    pixels = np.asarray(pixels, dtype="<u2")
    dataset.FrameOfReferenceUID = frame_uid
    dataset.DoseUnits = units
    dataset.DoseType = "PHYSICAL"
    dataset.DoseSummationType = "PLAN"
    dataset.DoseGridScaling = scaling
    dataset.NumberOfFrames = pixels.shape[0]
    dataset.Rows = pixels.shape[1]
    dataset.Columns = pixels.shape[2]
    dataset.ImagePositionPatient = list(origin)
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.PixelSpacing = [10, 10]
    dataset.GridFrameOffsetVector = (
        frame_offsets if frame_offsets is not None else [0, 2][:pixels.shape[0]]
    )
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return dataset


class CACDicomTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_directory.name)
        self.series_uid = generate_uid()
        self.frame_uid = generate_uid()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_volume(self):
        _write_ct(self.folder / "slice-z2.dcm", self.series_uid,
                  self.frame_uid, 2, 200)
        _write_ct(self.folder / "slice-z0", self.series_uid,
                  self.frame_uid, 0, 100, slope=1, intercept=-50)
        groups, _, _ = CAC.scan_ct_series(self.folder)
        return CAC.load_ct_volume(self.series_uid, groups[self.series_uid])

    def test_ct_scan_ignores_non_ct_and_loads_slices_in_physical_order(self):
        volume = self._create_volume()

        self.assertEqual(volume.image.shape, (4, 4, 2))
        self.assertEqual(volume.image[0, 0, 0], 50)
        self.assertEqual(volume.image[0, 0, 1], 300)
        self.assertEqual(volume.pixel_spacing, (1.0, 2.0))
        self.assertEqual(volume.slice_spacing, 2.0)

    def test_rtstruct_scan_checks_series_and_merges_same_named_rois(self):
        volume = self._create_volume()
        polygon = [0, 0, 2, 4, 0, 2, 4, 3, 2, 0, 3, 2]
        _write_rtstruct(self.folder / "rt-one", self.series_uid,
                        self.frame_uid, polygon)
        _write_rtstruct(self.folder / "rt-two.dcm", self.series_uid,
                        self.frame_uid, polygon)
        _write_rtstruct(self.folder / "wrong-series.dcm", generate_uid(),
                        self.frame_uid, polygon)

        rtstructs, _, notices = CAC.scan_rtstructs(self.folder, volume)
        self.assertEqual(len(rtstructs), 2)
        self.assertEqual(len(notices), 1)
        selected_rtstructs, ignored, notices = CAC.scan_rtstruct_paths(
            [self.folder / "rt-one"], volume
        )
        self.assertEqual(len(selected_rtstructs), 1)
        self.assertEqual(ignored, 0)
        self.assertEqual(notices, [])
        selected_ct, ignored, _ = CAC.scan_rtstruct_paths(
            [self.folder / "slice-z2.dcm"], volume
        )
        self.assertEqual(selected_ct, [])
        self.assertEqual(ignored, 1)

        structures, notices = CAC.merge_rtstructs(rtstructs)
        self.assertEqual(len(structures["coronary"]), 2)
        self.assertEqual(notices, [])

        mask, invalid, outside = CAC.get_mask_from_contours(
            structures["coronary"], volume
        )
        self.assertTrue(mask[1, 1, 1])
        self.assertEqual((invalid, outside), (0, 0))

        dataframe, invalid, outside = CAC.calculate_results(structures, volume)
        result = dataframe.iloc[0]
        self.assertEqual(result[CAC.AGATSTON_SCORE_COLUMN], 72.0)
        self.assertIn(CAC.AGATSTON_SCORE_COLUMN, dataframe.columns)
        self.assertNotIn("RTSTRUCTs", dataframe.columns)
        self.assertEqual(result["Volume_score"], 48.0)
        self.assertEqual((invalid, outside), (0, 0))

        dose = CAC.RTDoseVolume(
            path=self.folder / "dose.dcm",
            dose_gy=np.full((2, 4, 4), 2.5),
            origin=np.array([0.0, 0.0, 0.0]),
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            pixel_spacing=volume.pixel_spacing,
            frame_offsets=np.array([0.0, 2.0]),
            frame_uid=self.frame_uid,
            summation_type="PLAN",
        )
        dose_dataframe, _, _ = CAC.calculate_results(
            structures, volume, dose_grid=dose
        )
        self.assertAlmostEqual(dose_dataframe.iloc[0][CAC.DOSE_MEAN_COLUMN], 2.5)
        self.assertTrue(np.isnan(dose_dataframe.iloc[0][CAC.DOSE_D0_5CC_COLUMN]))

    def test_rt_dose_loads_cgy_as_gy_and_requires_matching_frame(self):
        volume = self._create_volume()
        dose_path = self.folder / "dose.dcm"
        _write_rt_dose(
            dose_path,
            self.frame_uid,
            np.full((2, 4, 4), 200),
            units="CGY",
            scaling=1.0,
        )

        dose = CAC.load_rt_dose(dose_path, volume)

        self.assertEqual(dose.dose_gy.shape, (2, 4, 4))
        self.assertEqual(dose.dose_gy[0, 0, 0], 2.0)
        self.assertEqual(dose.summation_type, "PLAN")

        absolute_path = self.folder / "absolute-offset-dose.dcm"
        _write_rt_dose(
            absolute_path,
            self.frame_uid,
            np.full((2, 4, 4), 200),
            origin=(0, 0, 2),
            frame_offsets=[2, 4],
        )
        absolute_dose = CAC.load_rt_dose(absolute_path, volume)
        np.testing.assert_array_equal(absolute_dose.frame_offsets, [0, 2])

        wrong_frame_path = self.folder / "wrong-frame-dose.dcm"
        _write_rt_dose(
            wrong_frame_path, generate_uid(), np.full((2, 4, 4), 200)
        )
        dose_with_different_frame = CAC.load_rt_dose(wrong_frame_path, volume)
        self.assertTrue(dose_with_different_frame.frame_mismatch)
        self.assertEqual(dose_with_different_frame.dose_gy[0, 0, 0], 200.0)

    def test_rt_dose_reader_retries_without_file_meta_header(self):
        volume = self._create_volume()
        dose_path = self.folder / "dose-without-preamble.dcm"
        dataset = _write_rt_dose(
            dose_path,
            self.frame_uid,
            np.full((2, 4, 4), 200),
        )
        dataset.save_as(dose_path, write_like_original=True)
        dose_path.write_bytes(dose_path.read_bytes()[132:])

        with patch.object(CAC.pydicom, "dcmread", wraps=CAC.pydicom.dcmread) as reader:
            dose = CAC.load_rt_dose(dose_path, volume)

        self.assertEqual(dose.dose_gy.shape, (2, 4, 4))
        self.assertEqual(reader.call_count, 2)
        self.assertFalse(reader.call_args_list[0].kwargs.get("force", False))
        self.assertTrue(reader.call_args_list[1].kwargs["force"])

    def test_rt_dose_reader_infers_transfer_syntax_when_file_meta_is_missing(self):
        volume = self._create_volume()
        dose_path = self.folder / "dose-without-file-meta.dcm"
        _write_rt_dose(
            dose_path,
            self.frame_uid,
            np.full((2, 4, 4), 200),
        )
        file_bytes = dose_path.read_bytes()
        file_meta_length = int.from_bytes(file_bytes[140:144], "little")
        dataset_offset = 132 + 12 + file_meta_length
        dose_path.write_bytes(file_bytes[dataset_offset:])

        dose = CAC.load_rt_dose(dose_path, volume)

        self.assertEqual(dose.dose_gy.shape, (2, 4, 4))
        self.assertEqual(dose.dose_gy[0, 0, 0], 200.0)

    def test_rt_dose_picker_loads_the_selected_file(self):
        volume = self._create_volume()
        dose_path = self.folder / "selected-dose.dcm"
        _write_rt_dose(
            dose_path, self.frame_uid, np.full((2, 4, 4), 20)
        )

        with patch.object(
            CAC.filedialog,
            "askopenfilename",
            return_value=str(dose_path),
        ) as picker:
            dose = CAC._choose_rt_dose_file(object(), self.folder, volume)

        self.assertEqual(dose.path, dose_path)
        picker.assert_called_once()
        self.assertEqual(picker.call_args.kwargs["initialdir"], str(self.folder))

    def test_dose_metrics_use_weighted_mean_and_hottest_half_cc(self):
        volume = self._create_volume()
        volume.pixel_spacing = (10.0, 10.0)
        volume.slice_spacings = np.array([10.0, 10.0])
        dose_values = np.zeros((2, 4, 4), dtype=float)
        dose_values[:, :, 0] = np.arange(1, 5)[None, :]
        dose = CAC.RTDoseVolume(
            path=self.folder / "dose.dcm",
            dose_gy=dose_values,
            origin=np.array([0.0, 0.0, 0.0]),
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            pixel_spacing=(10.0, 10.0),
            frame_offsets=np.array([0.0, 2.0]),
            frame_uid=self.frame_uid,
            summation_type="PLAN",
        )
        mask = np.zeros(volume.image.shape, dtype=bool)
        mask[:, 0, 0] = True

        mean_dose, d0_5cc = CAC._sample_rt_dose_on_mask(
            mask, volume, dose, "test ROI"
        )

        self.assertEqual(mean_dose, 2.5)
        self.assertEqual(d0_5cc, 4.0)

        smaller_dose = CAC.RTDoseVolume(
            **{
                **dose.__dict__,
                "dose_gy": dose_values[:, :2, :2],
            }
        )
        with self.assertRaisesRegex(ValueError, "cobre 50.0%"):
            CAC._sample_rt_dose_on_mask(mask, volume, smaller_dose, "test ROI")

        volume.slice_spacings = np.array([3.0, 3.0])
        weighted_values = np.zeros((2, 4, 4), dtype=float)
        weighted_values[:, :, 0] = [10.0, 8.0, 0.0, 0.0]
        weighted_dose = CAC.RTDoseVolume(
            **{
                **dose.__dict__,
                "dose_gy": weighted_values,
            }
        )
        fractional_cc_mask = np.zeros(volume.image.shape, dtype=bool)
        fractional_cc_mask[:2, 0, 0] = True
        mean_dose, d0_5cc = CAC._sample_rt_dose_on_mask(
            fractional_cc_mask, volume, weighted_dose, "test ROI"
        )
        self.assertAlmostEqual(mean_dose, 9.0)
        self.assertEqual(d0_5cc, 8.0)

    def test_heart_constraints_compare_arm_ii_thresholds_by_laterality(self):
        dataframe = pd.DataFrame([
            {
                "ROI": "Área cardíaca",
                CAC.DOSE_V8_COLUMN: 32.0,
                CAC.DOSE_V16_COLUMN: 4.0,
                CAC.DOSE_V20_COLUMN: 6.0,
            },
            {"ROI": "mama_D", CAC.DOSE_MEAN_COLUMN: 20.0},
            {"ROI": "mama_E", CAC.DOSE_MEAN_COLUMN: 40.0},
        ])

        constraints = CAC._heart_constraint_results(dataframe)
        statuses = {
            (row["Classificação"], row["Métrica"]): row
            for row in constraints
        }

        self.assertEqual(len(constraints), 4)
        self.assertEqual(CAC._treatment_laterality(dataframe), "Esquerda")
        self.assertEqual(
            statuses[("Ideal", "V16 Gy")]["Status"],
            "PASSA",
        )
        self.assertEqual(
            statuses[("Aceitável", "V20 Gy")]["Medido (%)"],
            6.0,
        )
        self.assertEqual(
            statuses[("Ideal", "V8 Gy")]["Status"],
            "NÃO PASSA",
        )
        self.assertEqual(
            statuses[("Aceitável", "V8 Gy")]["Status"],
            "PASSA",
        )

    def test_dose_volume_percentages_are_volume_weighted(self):
        volume = self._create_volume()
        volume.pixel_spacing = (10.0, 10.0)
        volume.slice_spacings = np.array([10.0, 10.0])
        dose_values = np.zeros((2, 4, 4), dtype=float)
        dose_values[0, 0, :2] = [16, 20]
        dose_values[1, 0, :2] = [8, 7]
        dose = CAC.RTDoseVolume(
            path=self.folder / "dose.dcm",
            dose_gy=dose_values,
            origin=np.array([0.0, 0.0, 0.0]),
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            pixel_spacing=(10.0, 10.0),
            frame_offsets=np.array([0.0, 2.0]),
            frame_uid=self.frame_uid,
            summation_type="PLAN",
        )
        mask = np.zeros(volume.image.shape, dtype=bool)
        mask[0, :2, :] = True

        _mean, _d0_5cc, percentages = CAC._sample_rt_dose_metrics(
            mask, volume, dose, "Área cardíaca"
        )

        self.assertEqual(percentages[16], 50.0)
        self.assertEqual(percentages[8], 75.0)
        self.assertEqual(percentages[20], 25.0)

    def test_treatment_laterality_is_not_guessed_when_breast_doses_tie(self):
        dataframe = pd.DataFrame([
            {"ROI": "mama_D", CAC.DOSE_MEAN_COLUMN: 20.0},
            {"ROI": "mama_E", CAC.DOSE_MEAN_COLUMN: 20.0},
            {
                "ROI": "Área cardíaca",
                CAC.DOSE_V8_COLUMN: 10.0,
                CAC.DOSE_V16_COLUMN: 0.0,
                CAC.DOSE_V20_COLUMN: 0.0,
            },
        ])

        self.assertIsNone(CAC._treatment_laterality(dataframe))
        constraints = CAC._heart_constraint_results(dataframe)
        self.assertTrue(all(row["Status"] == "NÃO AVALIADO" for row in constraints))
        self.assertTrue(all(
            row["Lateralidade"] == "Não identificada" for row in constraints
        ))

    def test_rtog_metrics_are_taken_only_from_area_cardiaca(self):
        dataframe = pd.DataFrame([
            {
                "ROI": "heart",
                CAC.DOSE_V8_COLUMN: 0.0,
                CAC.DOSE_V16_COLUMN: 0.0,
                CAC.DOSE_V20_COLUMN: 0.0,
            },
            {
                "ROI": "Área cardíaca",
                CAC.DOSE_V8_COLUMN: 32.0,
                CAC.DOSE_V16_COLUMN: 4.0,
                CAC.DOSE_V20_COLUMN: 6.0,
            },
            {"ROI": "mama_E", CAC.DOSE_MEAN_COLUMN: 40.0},
        ])

        constraints = CAC._heart_constraint_results(dataframe)

        self.assertEqual(len(constraints), 4)
        self.assertEqual(
            {row["Métrica"]: row["Medido (%)"] for row in constraints},
            {"V16 Gy": 4.0, "V20 Gy": 6.0, "V8 Gy": 32.0},
        )

    def test_open_pdf_opens_the_saved_file_in_default_viewer(self):
        pdf_path = self.folder / "saved-report.pdf"
        pdf_path.write_bytes(b"%PDF-")

        with patch.object(CAC.webbrowser, "open", return_value=True) as open_browser:
            CAC.open_pdf(pdf_path)

        open_browser.assert_called_once_with(pdf_path.resolve().as_uri())

    def test_dose_metrics_are_included_in_text_and_pdf_exports(self):
        from types import SimpleNamespace

        dataframe = pd.DataFrame([{
            "ROI": "aorta",
            CAC.AGATSTON_SCORE_COLUMN: 12.5,
            "Volume_score": 24.0,
            "Interpretação": "Risco leve",
            "Cor": "yellow",
            CAC.DOSE_MEAN_COLUMN: 3.25,
            CAC.DOSE_D0_5CC_COLUMN: 7.5,
            CAC.DOSE_V8_COLUMN: 20.0,
            CAC.DOSE_V16_COLUMN: 4.0,
            CAC.DOSE_V20_COLUMN: 6.0,
        }, {
            "ROI": "Área cardíaca",
            CAC.AGATSTON_SCORE_COLUMN: 0.0,
            "Volume_score": 0.0,
            "Interpretação": "Nenhum risco detectável",
            "Cor": "green",
            CAC.DOSE_MEAN_COLUMN: 3.25,
            CAC.DOSE_D0_5CC_COLUMN: 7.5,
            CAC.DOSE_V8_COLUMN: 32.0,
            CAC.DOSE_V16_COLUMN: 4.0,
            CAC.DOSE_V20_COLUMN: 6.0,
        }, {
            "ROI": "mama_E",
            CAC.AGATSTON_SCORE_COLUMN: 0.0,
            "Volume_score": 0.0,
            "Interpretação": "Nenhum risco detectável",
            "Cor": "green",
            CAC.DOSE_MEAN_COLUMN: 40.0,
            CAC.DOSE_D0_5CC_COLUMN: 42.0,
        }])
        dose_grid = SimpleNamespace(
            path=Path("dose.dcm"),
            summation_type="PLAN",
            frame_mismatch=False,
        )
        text = CAC.results_as_text(
            dataframe, "Paciente Teste", "ID-123", dose_grid
        )
        self.assertIn("Dose média (Gy)", text)
        self.assertIn("D0,5cc (Gy)", text)
        self.assertIn("aorta\t12.5\t24.0\tRisco leve\t3.25\t7.50", text)
        self.assertIn("RTDOSE\tdose.dcm", text)
        self.assertIn("DoseSummationType\tPLAN", text)
        self.assertIn("Constraints cardíacos", text)
        self.assertIn("V16 Gy", text)
        self.assertIn("V20 Gy", text)
        self.assertIn("NÃO PASSA", text)
        self.assertEqual(CAC._format_dose_value(np.nan), "N/A")

        dose_grid.frame_mismatch = True
        text = CAC.results_as_text(dataframe, "Paciente Teste", "ID-123", dose_grid)
        self.assertIn("Frames of Reference diferentes", text)

        output_path = self.folder / "dose-report.pdf"
        CAC.save_report_pdf(
            output_path,
            dataframe,
            "Paciente Teste",
            "ID-123",
            "30/06/2024",
            "15/10/2026",
            dose_grid=dose_grid,
        )
        pdf_bytes = output_path.read_bytes()
        page_content = b" ".join(
            zlib.decompress(
                base64.a85decode(match.group(1).rstrip(b"\r\n"), adobe=True)
            )
            for match in re.finditer(
                rb"stream\r?\n(.*?)endstream", pdf_bytes, re.DOTALL
            )
        )
        self.assertIn(b"Dose m", page_content)
        self.assertNotIn(b"M\\351tricas de dose", page_content)
        self.assertIn(b"D0,5cc", page_content)
        self.assertIn(b"7.50", page_content)
        self.assertNotIn(b"dose.dcm", page_content)
        self.assertIn(b"Constraints card", page_content)
        self.assertNotIn(b"Heart dose constraint", page_content)
        self.assertIn(b"Arm II", page_content)
        self.assertIn(b"mama esquerda", page_content)
        self.assertIn(b"PASSA", page_content)
        self.assertIn(b"N\\303O PASSA", page_content)
        dose_grid.frame_mismatch = True
        mismatch_output_path = self.folder / "dose-report-mismatch.pdf"
        CAC.save_report_pdf(
            mismatch_output_path,
            dataframe,
            "Paciente Teste",
            "ID-123",
            "30/06/2024",
            "15/10/2026",
            dose_grid=dose_grid,
        )
        mismatch_bytes = mismatch_output_path.read_bytes()
        mismatch_content = b" ".join(
            zlib.decompress(
                base64.a85decode(match.group(1).rstrip(b"\r\n"), adobe=True)
            )
            for match in re.finditer(
                rb"stream\r?\n(.*?)endstream", mismatch_bytes, re.DOTALL
            )
        )
        self.assertNotIn(b"Frames of Reference", mismatch_content)
        self.assertNotIn(b"sem registro espacial", mismatch_content)

    def test_file_picker_accepts_multiple_rtstruct_files(self):
        volume = self._create_volume()
        paths = [
            self.folder / "rt-one",
            self.folder / "rt-two.dcm",
        ]
        _write_rtstruct(paths[0], self.series_uid, self.frame_uid,
                        [0, 0, 2, 4, 0, 2, 4, 3, 2, 0, 3, 2])
        _write_rtstruct(paths[1], self.series_uid, self.frame_uid,
                        [0, 0, 2, 4, 0, 2, 4, 3, 2, 0, 3, 2])

        with patch.object(
            CAC.filedialog,
            "askopenfilenames",
            return_value=tuple(map(str, paths)),
        ) as picker:
            selected = CAC._choose_rtstruct_files(object(), self.folder, volume)

        self.assertEqual([Path(dataset.filename) for dataset in selected], paths)
        picker.assert_called_once()
        self.assertEqual(picker.call_args.kwargs["initialdir"], str(self.folder))
        self.assertIn("Ctrl/Shift", picker.call_args.kwargs["title"])

    def test_ct_folder_picker_uses_native_directory_dialog(self):
        root = object()
        with patch.object(CAC.filedialog, "askdirectory", return_value=str(self.folder)) as picker:
            selected = CAC._choose_ct_folder(root, self.folder)

        self.assertEqual(selected, str(self.folder))
        picker.assert_called_once_with(
            parent=root,
            title="Selecione a pasta com as imagens CT DICOM",
            initialdir=str(self.folder),
            mustexist=True,
        )

    def test_pdf_suggested_directory_is_sibling_of_ct_folder(self):
        ct_folder = self.folder / "CT"
        ct_folder.mkdir()
        ct_file = ct_folder / "slice.dcm"
        ct_file.touch()
        volume = type("TestVolume", (), {
            "slices": [type("TestSlice", (), {"filename": str(ct_file)})()]
        })()

        self.assertEqual(CAC._pdf_initial_directory(volume), self.folder)

    def test_anatomical_roi_names_are_translated_to_portuguese(self):
        dataset = Dataset()
        dataset.filename = "translated.dcm"
        dataset.StructureSetROISequence = []
        dataset.ROIContourSequence = []
        for number, name in enumerate(
            ["Cardiac_area", "aorta", "heart_myocardium", "coronary_arteries",
             "heart_atrium_left", "heart_atrium_right",
             "heart_ventricle_left", "heart_ventricle_right",
             "artéria pulmonar", "custom_roi"],
            start=1,
        ):
            roi = Dataset()
            roi.ROINumber = number
            roi.ROIName = name
            dataset.StructureSetROISequence.append(roi)
            contour = Dataset()
            contour.ReferencedROINumber = number
            contour.ContourSequence = [Dataset()]
            dataset.ROIContourSequence.append(contour)

        structures, notices = CAC.merge_rtstructs([dataset])

        self.assertEqual(
            set(structures),
            {
                "Área cardíaca",
                "Aorta",
                "Miocárdio",
                "Artérias coronárias",
                "Átrio esquerdo",
                "Átrio direito",
                "Ventrículo esquerdo",
                "Ventrículo direito",
                "Artéria pulmonar",
                "custom_roi",
            },
        )
        self.assertEqual(notices, [])

    def test_text_export_includes_patient_and_every_result_column(self):
        dataframe = pd.DataFrame([{
            "ROI": "aorta",
            CAC.AGATSTON_SCORE_COLUMN: 12.5,
            "Volume_score": 24.0,
            "Interpretação": "Risco leve",
            "Cor": "yellow",
        }])

        output = CAC.results_as_text(dataframe, "Paciente Teste", "ID-123")

        self.assertIn("Paciente\tPaciente Teste", output)
        self.assertIn("ID\tID-123", output)
        self.assertIn("Estrutura\tAgatston Score\tVolume (mm³)\tInterpretação", output)
        self.assertIn("aorta\t12.5\t24.0\tRisco leve", output)

    def test_legacy_misspelled_score_column_is_normalized_for_exports(self):
        dataframe = pd.DataFrame([{
            "ROI": "aorta",
            CAC.LEGACY_AGATSTON_SCORE_COLUMN: 12.5,
            "Volume_score": 24.0,
            "Interpretação": "Risco leve",
            "Cor": "yellow",
        }])

        normalized = CAC.normalize_results_dataframe(dataframe)
        text = CAC.results_as_text(dataframe, "Paciente Teste", "ID-123")
        output_path = self.folder / "legacy-report.pdf"
        CAC.save_report_pdf(
            output_path,
            dataframe,
            "Paciente Teste",
            "ID-123",
            "30/06/2024",
            "15/10/2026",
        )

        self.assertIn(CAC.AGATSTON_SCORE_COLUMN, normalized.columns)
        self.assertNotIn(CAC.LEGACY_AGATSTON_SCORE_COLUMN, normalized.columns)
        self.assertIn("aorta\t12.5\t24.0\tRisco leve", text)
        self.assertTrue(output_path.read_bytes().startswith(b"%PDF-"))

    def test_pdf_report_is_created_with_patient_and_results(self):
        dataframe = pd.DataFrame([{
            "ROI": "aorta",
            CAC.AGATSTON_SCORE_COLUMN: 12.5,
            "Volume_score": 24.0,
            "Interpretação": "🟡 Risco leve",
            "Cor": "yellow",
        }])
        output_path = self.folder / "report.pdf"

        CAC.save_report_pdf(
            output_path,
            dataframe,
            "Paciente Teste",
            "ID-123",
            "30/06/2024",
            "15/10/2026",
        )

        self.assertTrue(output_path.is_file())
        pdf_bytes = output_path.read_bytes()
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertIn(b"/MediaBox [ 0 0 595.2756 841.8898 ]", pdf_bytes)
        page_content = b" ".join(
            zlib.decompress(
                base64.a85decode(match.group(1).rstrip(b"\r\n"), adobe=True)
            )
            for match in re.finditer(
                rb"stream\r?\n(.*?)endstream", pdf_bytes, re.DOTALL
            )
        )
        self.assertIn(b"Data da tomografia", page_content)
        self.assertIn(b"30/06/2024", page_content)
        self.assertIn(b"Data da an", page_content)
        self.assertIn(b"15/10/2026", page_content)
        self.assertNotIn(b"20:45:00", page_content)
        self.assertIn(b"BELARDO, Alfonso et al.", page_content)
        self.assertIn(b"10.1016/j.radonc.2025.111098", page_content)
        self.assertNotIn(b"Acesso em:", page_content)
        ct_dataset = _write_ct(
            self.folder / "date-test.dcm",
            self.series_uid,
            self.frame_uid,
            0,
            100,
        )
        self.assertEqual(CAC._tomography_date(ct_dataset), "30/06/2024")

    def test_pdf_adds_one_physical_crop_page_per_positive_score_structure(self):
        from PIL import Image

        volume = self._create_volume()
        contour_data = [0, 0, 2, 4, 0, 2, 4, 3, 2, 0, 3, 2]
        rtstruct = _write_rtstruct(
            self.folder / "report-rtstruct.dcm",
            self.series_uid,
            self.frame_uid,
            contour_data,
            roi_name="coronary",
        )
        structures, notices = CAC.merge_rtstructs([rtstruct])
        self.assertEqual(notices, [])
        dataframe = pd.DataFrame([
            {
                "ROI": roi_name,
                CAC.AGATSTON_SCORE_COLUMN: 72.0,
                "Volume_score": 48.0,
                "Interpretação": "Risco leve",
                "Cor": "yellow",
            }
            for roi_name in ("coronary", "aorta")
        ])
        dataframe.loc[len(dataframe)] = {
            "ROI": "sem_calcificacao",
            CAC.AGATSTON_SCORE_COLUMN: 0.0,
            "Volume_score": 0.0,
            "Interpretação": "Nenhum risco detectável",
            "Cor": "green",
        }
        structures["aorta"] = structures["coronary"]
        findings = CAC._top_calcifications(structures["coronary"], volume)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["slice_index"], 1)
        self.assertEqual(finding["maximum_hu"], 300.0)
        image_bytes = CAC._render_calcification_crop(volume, finding)
        crop = Image.open(image_bytes)
        self.assertEqual(crop.size, (50, 100))
        self.assertIn((255, 40, 40), crop.getdata())
        self.assertNotIn((0, 255, 255), crop.getdata())
        dose_values = np.broadcast_to(
            np.arange(4, dtype=float)[None, None, :] * 5,
            (2, 4, 4),
        ).copy()
        dose_grid = CAC.RTDoseVolume(
            path=self.folder / "dose-overlay.dcm",
            dose_gy=dose_values,
            origin=np.array([0.0, 0.0, 0.0]),
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            pixel_spacing=volume.pixel_spacing,
            frame_offsets=np.array([0.0, 2.0]),
            frame_uid=self.frame_uid,
            summation_type="PLAN",
        )
        dose_image = Image.open(
            CAC._render_calcification_crop(volume, finding, dose_grid=dose_grid)
        )
        self.assertTrue(any(
            color in dose_image.getdata()
            for color in ((0, 255, 255), (255, 220, 0), (255, 0, 255))
        ))
        output_path = self.folder / "report-with-images.pdf"

        with patch.object(
            CAC,
            "_top_calcifications",
            return_value=[finding, finding, finding],
        ):
            CAC.save_report_pdf(
                output_path,
                dataframe,
                "Paciente Teste",
                "ID-123",
                "30/06/2024",
                "15/10/2026",
                volume=volume,
                structures=structures,
                dose_grid=dose_grid,
            )

        pdf_bytes = output_path.read_bytes()
        self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", pdf_bytes)), 3)
        self.assertGreaterEqual(len(re.findall(rb"/Subtype\s*/Image", pdf_bytes)), 1)
        page_content = b" ".join(
            zlib.decompress(
                base64.a85decode(match.group(1).rstrip(b"\r\n"), adobe=True)
            )
            for match in re.finditer(
                rb"stream\r?\n(.*?)endstream", pdf_bytes, re.DOTALL
            )
        )
        self.assertIn(b"Calcifica", page_content)
        self.assertIn(b"coronary", page_content)
        self.assertIn(b"aorta", page_content)
        self.assertIn(b"recorte de 10", page_content)
        self.assertEqual(page_content.count(b"Isodoses"), 2)
        self.assertEqual(page_content.count(b"Calcifica"), 2)
        self.assertEqual(page_content.count(b"Grupo 1"), 2)
        self.assertEqual(page_content.count(b"Grupo 2"), 2)
        self.assertEqual(page_content.count(b"Grupo 3"), 2)
        self.assertNotIn(b"separados por mais", page_content)

    def test_calcification_selection_keeps_three_groups_over_five_mm_apart(self):
        candidates = [
            {"score": 50, "maximum_hu": 300, "slice_index": 0,
             "center": (10, 10), "cluster": [(10, 10)]},
            {"score": 45, "maximum_hu": 300, "slice_index": 0,
             "center": (10, 12.5), "cluster": [(10, 12)]},
            {"score": 40, "maximum_hu": 250, "slice_index": 0,
             "center": (10, 13), "cluster": [(10, 13)]},
            {"score": 35, "maximum_hu": 200, "slice_index": 0,
             "center": (10, 16), "cluster": [(10, 16)]},
        ]

        selected = CAC._select_distinct_calcifications(
            candidates,
            pixel_spacing=(1.0, 2.0),
        )

        self.assertEqual([item["score"] for item in selected], [50, 40, 35])

    def test_invalid_files_are_ignored_and_clusters_do_not_recurse(self):
        _write_ct(self.folder / "slice.dcm", self.series_uid,
                  self.frame_uid, 0, 100)
        (self.folder / "not-a-dicom.txt").write_text("not a DICOM file")
        (self.folder / "slice.dcm:Zone.Identifier").write_text(
            "[ZoneTransfer]\nZoneId=3\n"
        )
        groups, ignored, notices = CAC.scan_ct_series(self.folder)
        self.assertEqual(len(groups), 1)
        self.assertEqual(ignored, 1)
        self.assertTrue(notices)

        matrix = np.full((100, 100), 200, dtype=float)
        clusters = CAC.find_clusters(matrix)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 10_000)

    def test_nonuniform_slice_spacing_is_supported(self):
        _write_ct(self.folder / "z0.dcm", self.series_uid,
                  self.frame_uid, 0, 100)
        _write_ct(self.folder / "z2.dcm", self.series_uid,
                  self.frame_uid, 2, 100)
        _write_ct(self.folder / "z5.dcm", self.series_uid,
                  self.frame_uid, 5, 100)

        groups, _, notices = CAC.scan_ct_series(self.folder)
        self.assertEqual(len(groups), 1)
        self.assertEqual(notices, [])
        volume = CAC.load_ct_volume(self.series_uid, groups[self.series_uid])
        np.testing.assert_allclose(volume.slice_spacings, [2.0, 2.5, 3.0])

    def test_ct_modality_with_nonstandard_sop_class_is_detected(self):
        _write_ct(
            self.folder / "slice.dcm",
            self.series_uid,
            self.frame_uid,
            0,
            100,
            sop_class=generate_uid(),
        )

        groups, _, notices = CAC.scan_ct_series(self.folder)
        self.assertEqual(set(groups), {self.series_uid})
        self.assertEqual(notices, [])

    def test_multiple_ct_series_are_kept_separate(self):
        _write_ct(self.folder / "first.dcm", self.series_uid,
                  self.frame_uid, 0, 100)
        other_series = generate_uid()
        _write_ct(self.folder / "second.dcm", other_series,
                  self.frame_uid, 0, 100)

        groups, _, _ = CAC.scan_ct_series(self.folder)
        self.assertEqual(set(groups), {self.series_uid, other_series})
        self.assertEqual(len(groups[self.series_uid]), 1)
        self.assertEqual(len(groups[other_series]), 1)


if __name__ == "__main__":
    unittest.main()
