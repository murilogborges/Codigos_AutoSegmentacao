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

        with patch.object(CAC.filedialog, "askopenfilenames",
                          return_value=tuple(map(str, paths))) as picker:
            selected = CAC._choose_rtstruct_files(object(), self.folder, volume)

        self.assertEqual([Path(dataset.filename) for dataset in selected], paths)
        picker.assert_called_once()
        self.assertIn("Ctrl/Shift", picker.call_args.kwargs["title"])

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
        ct_dataset = _write_ct(
            self.folder / "date-test.dcm",
            self.series_uid,
            self.frame_uid,
            0,
            100,
        )
        self.assertEqual(CAC._tomography_date(ct_dataset), "30/06/2024")

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
