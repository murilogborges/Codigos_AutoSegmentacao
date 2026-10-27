import tempfile
import unittest
from pathlib import Path
import struct
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RTDoseStorage,
    RTStructureSetStorage,
    generate_uid,
)

import rt_dvh_app as app


def _new_file(path, sop_class, modality):
    instance_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = instance_uid
    dataset.Modality = modality
    return dataset


class DvhBackendTests(unittest.TestCase):
    def test_default_rtstruct_prefers_total_left_and_valid_ss_right(self):
        files = [
            app.DicomFile(Path("medical.dcm"), "", ""),
            app.DicomFile(Path("TOTAL_auto.dcm"), "", ""),
            app.DicomFile(Path("SS_review.dcm"), "", ""),
        ]

        self.assertEqual(app.choose_default_rtstructs(files), (1, 2))
        self.assertEqual(
            app.preferred_ptv_indices(["Body", "ptv_boost", "PTV"]), [1, 2]
        )
        self.assertEqual(app.preferred_ptv_indices(["Body", "PTV_high"]), [1])
        self.assertEqual(app.preferred_ptv_indices(["Body"]), [])

    def test_isocenter_rois_are_excluded_case_insensitively(self):
        names = [
            "PTV",
            "Isocenter",
            "Plan_Isocenter_Point",
            "Body",
            "isocenter reference",
        ]

        self.assertEqual(
            app.filter_selectable_roi_names(names), ["PTV", "Body"]
        )

    def test_read_roi_names_hides_isocenter_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "structures.dcm"
            structure = _new_file(path, RTStructureSetStorage, "RTSTRUCT")
            structure.StructureSetROISequence = Sequence()
            for index, name in enumerate(
                ["PTV_70Gy", "Isocenter", "Body"], start=1
            ):
                roi = Dataset()
                roi.ROINumber = index
                roi.ROIName = name
                structure.StructureSetROISequence.append(roi)
            structure.save_as(path, enforce_file_format=True)

            names = app.read_roi_names(path)

        self.assertEqual(names, ["PTV_70Gy", "Body"])

    def test_containment_candidates_use_fraction_of_left_roi_inside_right(self):
        right_mask = np.array([1, 1, 0, 0, 1], dtype=bool)
        left_masks = {
            "exact threshold": np.array([1, 1, 0, 0, 0], dtype=bool),
            "below threshold": np.array([1, 0, 1, 0, 0], dtype=bool),
            "empty": np.zeros(5, dtype=bool),
        }

        self.assertEqual(
            app.contained_structure_candidates(left_masks, right_mask),
            [("exact threshold", 1.0)],
        )

    def test_containment_candidates_accept_sixty_percent_and_reject_less(self):
        right_mask = np.array([1, 1, 1, 0, 0], dtype=bool)
        left_masks = {
            "sixty": np.ones(5, dtype=bool),
            "fifty": np.array([1, 1, 0, 1, 1], dtype=bool),
        }

        self.assertEqual(
            app.contained_structure_candidates(left_masks, right_mask),
            [("sixty", 0.6)],
        )

    def test_containment_failure_message_reports_best_overlap(self):
        message = app.containment_failure_message(
            {"Cervical": 0.4, "Thoracic": 0.0}
        )

        self.assertIn("Nenhuma alcançou o limite de 60%", message)
        self.assertIn("Cervical (40.0%)", message)
        self.assertIn("fração do volume", message)

    def test_containment_failure_message_flags_no_spatial_overlap(self):
        message = app.containment_failure_message(
            {"Cervical": 0.0, "Thoracic": 0.0}
        )

        self.assertIn("Não houve sobreposição", message)
        self.assertIn("mesmo paciente", message)

    def test_discover_dicom_files_groups_ct_and_detects_rt_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            series_uid = generate_uid()
            ct = _new_file(folder / "ct.dcm", CTImageStorage, "CT")
            ct.SeriesInstanceUID = series_uid
            ct.SeriesDescription = "Planning CT"
            ct.PatientName = "Paciente^Teste"
            ct.save_as(ct.filename, enforce_file_format=True)
            dose = _new_file(folder / "dose.dcm", RTDoseStorage, "RTDOSE")
            dose.save_as(dose.filename, enforce_file_format=True)
            structure = _new_file(folder / "medico.dcm", RTStructureSetStorage, "RTSTRUCT")
            structure.save_as(structure.filename, enforce_file_format=True)
            total = _new_file(folder / "TOTAL_auto.dcm", RTStructureSetStorage, "RTSTRUCT")
            total.save_as(total.filename, enforce_file_format=True)
            (folder / "noise.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            (folder / "not-dicom.dcm").write_bytes(b"not a DICOM file")

            progress = []
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                ct_files, doses, structures = app.discover_dicom_files(
                    folder, progress=lambda percent, message: progress.append((percent, message))
                )

        self.assertEqual(len(ct_files), 1)
        self.assertEqual(ct_files[0].series_uid, series_uid)
        self.assertEqual(ct_files[0].instance_paths, (folder / "ct.dcm",))
        self.assertIn("Paciente: Paciente^Teste", ct_files[0].label)
        self.assertEqual(len(doses), 1)
        self.assertEqual([item.path.name for item in structures], ["TOTAL_auto.dcm", "medico.dcm"])
        self.assertEqual(captured, [])
        self.assertEqual(progress[-1][0], 100)

    def test_read_dicom_corrects_explicit_implicit_vr_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vr-mismatch.dcm"
            dataset = _new_file(path, CTImageStorage, "CT")
            dataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            dataset.PatientName = "Teste^VR"
            pydicom.dcmwrite(path, dataset, implicit_vr=True, little_endian=True)

            raw_data = bytearray(path.read_bytes())
            transfer_syntax_element = raw_data.index(b"\x02\x00\x10\x00UI")
            length_offset = transfer_syntax_element + 6
            value_offset = transfer_syntax_element + 8
            old_length = struct.unpack_from("<H", raw_data, length_offset)[0]
            explicit_uid = str(ExplicitVRLittleEndian).encode("ascii")
            if len(explicit_uid) % 2:
                explicit_uid += b"\0"
            raw_data[value_offset : value_offset + old_length] = explicit_uid
            struct.pack_into("<H", raw_data, length_offset, len(explicit_uid))
            path.write_bytes(raw_data)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with self.assertLogs("rt_dvh_app", level="INFO"):
                    result = app.read_dicom(path)

        self.assertEqual(result.file_meta.TransferSyntaxUID, ImplicitVRLittleEndian)
        self.assertEqual(result.PatientName, "Teste^VR")
        self.assertEqual(captured, [])

    def test_read_dicom_corrects_implicit_metadata_with_explicit_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reverse-vr-mismatch.dcm"
            dataset = _new_file(path, CTImageStorage, "CT")
            dataset.PatientName = "Teste^VR"
            dataset.save_as(path, enforce_file_format=True)
            raw_data = bytearray(path.read_bytes())
            syntax_element = raw_data.index(b"\x02\x00\x10\x00UI")
            length_offset = syntax_element + 6
            value_offset = syntax_element + 8
            old_length = struct.unpack_from("<H", raw_data, length_offset)[0]
            implicit_uid = str(ImplicitVRLittleEndian).encode("ascii")
            if len(implicit_uid) % 2:
                implicit_uid += b"\0"
            raw_data[value_offset : value_offset + old_length] = implicit_uid
            struct.pack_into("<H", raw_data, length_offset, len(implicit_uid))
            group_length = struct.unpack_from("<I", raw_data, 140)[0]
            struct.pack_into(
                "<I", raw_data, 140, group_length + len(implicit_uid) - old_length
            )
            path.write_bytes(raw_data)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = app.read_dicom(path)

        self.assertEqual(result.file_meta.TransferSyntaxUID, ExplicitVRLittleEndian)
        self.assertEqual(result.PatientName, "Teste^VR")
        self.assertEqual(captured, [])

    def test_dvh_loads_rt_dose_once_for_all_curves(self):
        ct_series = app.CTSeries(
            datasets=[],
            paths=[],
            row_spacing=1.0,
            column_spacing=1.0,
            slice_spacing=1.0,
            row_direction=np.ones(3),
            column_direction=np.ones(3),
        )
        masks = {"Auto": np.ones((1, 1, 1), dtype=bool)}
        grid = object()

        with (
            patch.object(app, "margin_mask", side_effect=lambda mask, *_args: mask),
            patch.object(app, "erode_mask", side_effect=lambda mask, *_args: mask),
            patch.object(app, "load_dose_grid", return_value=grid) as load_grid,
            patch.object(
                app,
                "sample_dose_on_mask",
                return_value=(np.array([1.0, 2.0]), 0.001),
            ) as sample,
        ):
            app.compute_dvh_curves(
                ct_series,
                Path("dose.dcm"),
                masks,
                masks["Auto"],
                np.ones((1, 1, 1), dtype=bool),
            )

        load_grid.assert_called_once_with(Path("dose.dcm"))
        self.assertEqual(sample.call_count, 5)
        self.assertTrue(all(call.kwargs["dose_grid"] is grid for call in sample.call_args_list))

    def test_external_dose_picker_validation_is_independent_of_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            dose_path = Path(temporary) / "external-dose.dcm"
            dose = _new_file(dose_path, RTDoseStorage, "RTDOSE")
            dose.save_as(dose.filename, enforce_file_format=True)
            dose_path.write_bytes(dose_path.read_bytes()[132:])
            selected = app.inspect_dose_file(dose_path)
            ct = _new_file(Path(temporary) / "ct.dcm", CTImageStorage, "CT")
            ct.save_as(ct.filename, enforce_file_format=True)
            with self.assertRaisesRegex(ValueError, "não é um RTDOSE"):
                app.inspect_dose_file(ct.filename)
            discovered = app.discover_dicom_files(temporary)

        self.assertEqual(selected.path, dose_path)
        self.assertEqual([item.path for item in discovered[1]], [dose_path])

    def test_read_dicom_retries_file_without_dicom_preamble(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "without-preamble.dcm"
            dataset = _new_file(path, RTDoseStorage, "RTDOSE")
            dataset.FrameOfReferenceUID = generate_uid()
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.NumberOfFrames = 1
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.DoseUnits = "GY"
            dataset.DoseGridScaling = 0.1
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dataset.ImagePositionPatient = [0, 0, 0]
            dataset.PixelSpacing = [1.0, 1.0]
            dataset.GridFrameOffsetVector = [0.0]
            dataset.PixelData = np.array([[10, 20], [30, 40]], dtype="<u2").tobytes()
            dataset.save_as(path, enforce_file_format=True)
            path.write_bytes(path.read_bytes()[132:])

            dose, *_ = app.load_dose_grid(path)

        np.testing.assert_array_equal(dose, [[[1.0, 2.0], [3.0, 4.0]]])

    def test_read_dose_decodes_pixel_data_when_transfer_syntax_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing-transfer-syntax.dcm"
            dataset = _new_file(path, RTDoseStorage, "RTDOSE")
            dataset.FrameOfReferenceUID = generate_uid()
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.NumberOfFrames = 1
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.DoseUnits = "GY"
            dataset.DoseGridScaling = 0.1
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dataset.ImagePositionPatient = [0, 0, 0]
            dataset.PixelSpacing = [1.0, 1.0]
            dataset.GridFrameOffsetVector = [0.0]
            dataset.PixelData = np.array([[10, 20], [30, 40]], dtype="<u2").tobytes()
            dataset.save_as(path, enforce_file_format=True)
            offset, _ = app._dataset_start_and_transfer_syntax(path)
            path.write_bytes(path.read_bytes()[offset:])

            dose, *_ = app.load_dose_grid(path)

        np.testing.assert_array_equal(dose, [[[1.0, 2.0], [3.0, 4.0]]])

    def test_dvh_curve_and_margin_use_physical_voxel_spacing(self):
        curve = app.create_dvh_curve("ROI", np.array([0.0, 1.0, 2.0, 2.0]), 0.5)
        self.assertEqual(curve.volume_cc[0], 2.0)
        self.assertEqual(curve.volume_percent[0], 100.0)
        self.assertEqual(curve.volume_cc[-1], 1.0)

        class Series:
            row_spacing = 2.0
            column_spacing = 1.0
            slice_spacing = 3.0

        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[3, 3, 3] = True
        expanded = app.margin_mask(mask, Series(), 2.0)
        self.assertTrue(expanded[3, 4, 3])
        self.assertTrue(expanded[3, 5, 3])
        self.assertFalse(expanded[3, 6, 3])

    def test_margin_crop_matches_full_volume_distance_transform(self):
        from scipy.ndimage import distance_transform_edt

        class Series:
            row_spacing = 0.8
            column_spacing = 1.2
            slice_spacing = 2.5

        rng = np.random.default_rng(42)
        mask = rng.random((23, 19, 13)) > 0.995
        mask[0, 0, 0] = True
        for margin_mm in (0.5, 1.0, 5.0):
            expected = distance_transform_edt(
                ~mask,
                sampling=(
                    Series.row_spacing,
                    Series.column_spacing,
                    Series.slice_spacing,
                ),
            ) <= margin_mm
            np.testing.assert_array_equal(
                app.margin_mask(mask, Series(), margin_mm), expected
            )

    def test_erode_mask_uses_physical_five_millimeter_boundary(self):
        class Series:
            row_spacing = 1.0
            column_spacing = 1.0
            slice_spacing = 1.0

        patient = np.ones((15, 15, 15), dtype=bool)

        eroded = app.erode_mask(patient, Series(), 5.0)

        self.assertFalse(eroded[4, 7, 7])
        self.assertTrue(eroded[5, 7, 7])
        self.assertTrue(eroded[7, 7, 7])
        self.assertTrue(eroded.any())

    def test_dvh_individualizes_auto_rois_and_clips_both_margins(self):
        class Series:
            row_spacing = 1.0
            column_spacing = 1.0
            slice_spacing = 1.0
            datasets = []
            shape = (5, 5, 5)

        left_masks = {
            "Lung_L": np.ones((5, 5, 5), dtype=bool),
            "Heart": np.eye(5, dtype=bool)[:, :, None].repeat(5, axis=2),
        }
        patient = np.ones((5, 5, 5), dtype=bool)
        safe_volume = np.zeros_like(patient)
        safe_volume[2, 2, 2] = True
        captured_masks = []

        with (
            patch.object(app, "erode_mask", return_value=safe_volume),
            patch.object(
                app, "margin_mask", side_effect=lambda _mask, *_args: np.ones_like(patient)
            ),
            patch.object(app, "load_dose_grid", return_value=object()),
            patch.object(
                app,
                "sample_dose_on_mask",
                side_effect=lambda mask, *_args, **_kwargs: (
                    captured_masks.append(mask.copy()) or (np.array([1.0]), 0.001)
                ),
            ),
        ):
            curves, _combined, _right = app.compute_dvh_curves(
                Series(),
                Path("dose.dcm"),
                left_masks,
                patient,
                patient,
            )

        self.assertEqual(
            [curve.name for curve in curves],
            [
                "Auto-segmentação — Lung_L",
                "Auto-segmentação — Heart",
                "Auto-segmentação — união",
                "Auto-segmentação — margem 0,5 cm",
                "Auto-segmentação — margem 1 cm",
                "PTV Original",
            ],
        )
        np.testing.assert_array_equal(captured_masks[3], safe_volume)
        np.testing.assert_array_equal(captured_masks[4], safe_volume)
        np.testing.assert_array_equal(captured_masks[0], left_masks["Lung_L"])
        np.testing.assert_array_equal(captured_masks[5], patient)

    def test_dvh_hover_cursor_shows_dose_and_updates_legend_volumes(self):
        figure = app.Figure()
        axes = figure.add_subplot(1, 1, 1)
        curve = app.DVHCurve(
            "PTV Original",
            np.array([0.0, 0.1, 0.2]),
            np.array([1.0, 0.6, 0.0]),
            np.array([100.0, 60.0, 0.0]),
        )
        line, = axes.plot(curve.dose_gy, curve.volume_percent, label=curve.name)
        cursor = axes.axvline(0, visible=False)
        readout = axes.text(0.98, 0.04, "", transform=axes.transAxes)

        class Canvas:
            draw_calls = 0

            def draw_idle(self):
                self.draw_calls += 1

        application = app.DvhApplication.__new__(app.DvhApplication)
        application.ax_percent = axes
        application.ax_cc = None
        application._dvh_plot_lines = {axes: [(curve, line)]}
        application._dvh_cursor_lines = {axes: cursor}
        application._dvh_cursor_labels = {axes: readout}
        application.dvh_canvas = Canvas()

        application._update_dvh_cursor(
            SimpleNamespace(inaxes=axes, xdata=0.15)
        )

        self.assertTrue(cursor.get_visible())
        self.assertEqual(readout.get_text(), "Dose: 0.15 Gy")
        self.assertIn("PTV Original — 30.0 %", line.get_label())
        self.assertEqual(application.dvh_canvas.draw_calls, 1)

    def test_display_downsampling_preserves_a_single_voxel_roi(self):
        mask = np.zeros((120, 90, 70), dtype=bool)
        mask[11, 13, 17] = True
        reduced, strides = app.coarsen_mask(mask)
        self.assertTrue(reduced.any())
        self.assertEqual(strides, (3, 2, 2))
        self.assertLessEqual(max(reduced.shape), 55)

    def test_sample_dose_interpolates_on_patient_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame_uid = generate_uid()
            series_uid = generate_uid()
            ct_datasets = []
            ct_paths = []
            for index, z in enumerate((0.0, 2.0)):
                path = folder / f"ct-{index}.dcm"
                dataset = _new_file(path, CTImageStorage, "CT")
                dataset.SeriesInstanceUID = series_uid
                dataset.FrameOfReferenceUID = frame_uid
                dataset.Rows = 2
                dataset.Columns = 2
                dataset.PixelSpacing = [1.0, 1.0]
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, z]
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 1
                dataset.PixelData = np.zeros((2, 2), dtype="<i2").tobytes()
                dataset.save_as(path, enforce_file_format=True)
                ct_datasets.append(pydicom.dcmread(path))
                ct_paths.append(path)

            dose_path = folder / "dose.dcm"
            dose = _new_file(dose_path, RTDoseStorage, "RTDOSE")
            dose.FrameOfReferenceUID = frame_uid
            dose.Rows = 2
            dose.Columns = 2
            dose.NumberOfFrames = 2
            dose.SamplesPerPixel = 1
            dose.PhotometricInterpretation = "MONOCHROME2"
            dose.BitsAllocated = 16
            dose.BitsStored = 16
            dose.HighBit = 15
            dose.PixelRepresentation = 0
            dose.DoseUnits = "GY"
            dose.DoseGridScaling = 0.01
            dose.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dose.ImagePositionPatient = [0, 0, 0]
            dose.PixelSpacing = [1.0, 1.0]
            dose.GridFrameOffsetVector = [0.0, 2.0]
            dose.PixelData = np.array(
                [[[100, 200], [300, 400]], [[500, 600], [700, 800]]],
                dtype="<u2",
            ).tobytes()
            dose.save_as(dose_path, enforce_file_format=True)
            dose.FrameOfReferenceUID = generate_uid()
            dose.save_as(dose_path, enforce_file_format=True)

            ct_series = app.CTSeries(
                ct_datasets, ct_paths, 1.0, 1.0, 2.0,
                np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            )
            mask = np.zeros((2, 2, 2), dtype=bool)
            mask[0, 0, :] = True
            samples = app.sample_dose_on_mask(mask, ct_series, dose_path)
            shifted_dose = pydicom.dcmread(dose_path)
            shifted_dose.ImagePositionPatient = [1.0, 0.0, 0.0]
            shifted_dose.save_as(dose_path, enforce_file_format=True)
            partial_mask = np.zeros((2, 2, 2), dtype=bool)
            partial_mask[0, 0, 0] = True
            partial_mask[0, 1, 0] = True
            partial_samples = app.sample_dose_on_mask(
                partial_mask, ct_series, dose_path
            )

        np.testing.assert_allclose(samples[0], [1.0, 5.0])
        self.assertEqual(samples[1], 0.002)
        np.testing.assert_allclose(partial_samples[0], [1.0])

    def test_sample_dose_fails_when_no_mask_voxels_are_covered(self):
        ct_datasets = []
        for z in (0.0, 2.0):
            dataset = Dataset()
            dataset.FrameOfReferenceUID = generate_uid()
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.ImagePositionPatient = [0.0, 0.0, z]
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ct_datasets.append(dataset)
        ct_series = app.CTSeries(
            ct_datasets,
            [],
            1.0,
            1.0,
            2.0,
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        dose_grid = (
            np.ones((1, 2, 2), dtype=float),
            np.array([100.0, 100.0, 100.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            (1.0, 1.0, np.array([0.0])),
            str(ct_datasets[0].FrameOfReferenceUID),
        )
        mask = np.ones((2, 2, 2), dtype=bool)

        with self.assertRaisesRegex(ValueError, "Nenhum voxel"):
            app.sample_dose_on_mask(mask, ct_series, dose_grid=dose_grid)

    def test_dvh_curve_uses_tenth_gy_intervals(self):
        curve = app.create_dvh_curve(
            "Estrutura", np.array([0.06, 0.16, 0.27, 0.3]), 0.001
        )

        np.testing.assert_allclose(np.diff(curve.dose_gy), 0.1)
        np.testing.assert_allclose(curve.dose_gy[:3], [0.0, 0.1, 0.2])
        self.assertTrue(np.any(np.isclose(curve.dose_gy, 0.3)))

    def test_rt_utils_masks_are_reordered_to_rows_columns_slices(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame_uid = generate_uid()
            series_uid = generate_uid()
            for index, z in enumerate((0.0, 2.0)):
                path = folder / f"ct-{index}.dcm"
                dataset = _new_file(path, CTImageStorage, "CT")
                dataset.SeriesInstanceUID = series_uid
                dataset.FrameOfReferenceUID = frame_uid
                dataset.Rows = 3
                dataset.Columns = 5
                dataset.PixelSpacing = [1.0, 1.0]
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, z]
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 1
                dataset.PixelData = np.zeros((3, 5), dtype="<i2").tobytes()
                dataset.save_as(path, enforce_file_format=True)

            structure_path = folder / "sem_prefix_rt.dcm"
            structure = _new_file(structure_path, RTStructureSetStorage, "RTSTRUCT")
            roi = Dataset()
            roi.ROINumber = 1
            roi.ROIName = "Auto"
            roi.ReferencedFrameOfReferenceUID = frame_uid
            structure.StructureSetROISequence = Sequence([roi])
            contour = Dataset()
            malformed_contour = Dataset()
            malformed_contour.ContourGeometricType = "CLOSED_PLANAR"
            malformed_contour.NumberOfContourPoints = 0
            malformed_contour.ContourData = []
            contour.ContourGeometricType = "CLOSED_PLANAR"
            contour.NumberOfContourPoints = 4
            contour.ContourData = [1, 1, 0, 4, 1, 0, 4, 2, 0, 1, 2, 0]
            contour_image = Dataset()
            contour_image.ReferencedSOPClassUID = CTImageStorage
            contour_image.ReferencedSOPInstanceUID = generate_uid()
            contour.ContourImageSequence = Sequence([contour_image])
            roi_contour = Dataset()
            roi_contour.ReferencedROINumber = 1
            roi_contour.ContourSequence = Sequence([contour, malformed_contour])
            structure.ROIContourSequence = Sequence([roi_contour])
            frame = Dataset()
            frame.FrameOfReferenceUID = generate_uid()
            study = Dataset()
            series = Dataset()
            series.SeriesInstanceUID = generate_uid()
            study.RTReferencedSeriesSequence = Sequence([series])
            frame.RTReferencedStudySequence = Sequence([study])
            structure.ReferencedFrameOfReferenceSequence = Sequence([frame])
            structure.save_as(structure_path, enforce_file_format=True)

            ct_series = app.load_ct_series(folder, series_uid)
            with self.assertLogs("rt_dvh_app", level="WARNING") as logged:
                masks = app.load_structure_masks(ct_series, structure_path, ["Auto"])
            structure = pydicom.dcmread(structure_path)
            del structure.ReferencedFrameOfReferenceSequence
            del structure.ROIContourSequence[0].ContourSequence[0].ContourImageSequence
            structure.save_as(structure_path, enforce_file_format=True)
            masks_without_references = app.load_structure_masks(
                ct_series, structure_path, ["Auto"]
            )
            structure = pydicom.dcmread(structure_path)
            structure.ROIContourSequence[0].ContourSequence = Sequence(
                [structure.ROIContourSequence[0].ContourSequence[1]]
            )
            structure.save_as(structure_path, enforce_file_format=True)
            with self.assertRaisesRegex(ValueError, "não possui contornos rasterizáveis"):
                app.load_structure_masks(ct_series, structure_path, ["Auto"])

        self.assertEqual(masks["Auto"].shape, (3, 5, 2))
        self.assertTrue(masks["Auto"][:, :, 0].any())
        self.assertFalse(masks["Auto"][:, :, 1].any())
        self.assertTrue(logged.output)
        self.assertTrue(any("Auto" in line for line in logged.output))
        np.testing.assert_array_equal(masks["Auto"], masks_without_references["Auto"])

    def test_auto_and_medical_masks_are_drawn_in_separate_3d_axes(self):
        figure = app.Figure()
        auto_axes = figure.add_subplot(1, 2, 1, projection="3d")
        medical_axes = figure.add_subplot(1, 2, 2, projection="3d")
        auto_mask = np.zeros((4, 4, 4), dtype=bool)
        medical_mask = np.zeros_like(auto_mask)
        auto_mask[1, 1, 1] = True
        medical_mask[2, 2, 2] = True

        app.draw_structure_3d(auto_axes, medical_axes, auto_mask, medical_mask, None)

        self.assertEqual(auto_axes.get_title(), "Auto-segmentação")
        self.assertEqual(medical_axes.get_title(), "Estrutura médica")
        self.assertFalse(auto_axes.axison)
        self.assertFalse(medical_axes.axison)
        self.assertTrue(auto_axes.collections)
        self.assertTrue(medical_axes.collections)
        self.assertEqual(auto_axes.get_zlim(), medical_axes.get_zlim())
        auto_axes.view_init(elev=31, azim=123, roll=17)
        app.sync_3d_view(auto_axes, medical_axes)
        self.assertEqual(medical_axes.elev, 31)
        self.assertEqual(medical_axes.azim, 123)
        self.assertEqual(medical_axes.roll, 17)

    def test_surface_mesh_uses_physical_cranial_caudal_position(self):
        from pydicom.dataset import Dataset

        slices = []
        for z_position in range(10):
            dataset = Dataset()
            dataset.ImagePositionPatient = [0, 0, z_position * 5]
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            slices.append(dataset)
        ct_series = app.CTSeries(
            slices,
            [],
            row_spacing=1.0,
            column_spacing=1.0,
            slice_spacing=5.0,
            row_direction=np.array([1.0, 0.0, 0.0]),
            column_direction=np.array([0.0, 1.0, 0.0]),
        )
        cervical_thoracic = np.zeros((12, 12, 10), dtype=bool)
        lumbar = np.zeros_like(cervical_thoracic)
        cervical_thoracic[3:8, 3:8, 6:9] = True
        lumbar[3:8, 3:8, 1:4] = True
        upper_vertices, _ = app._structure_surface(cervical_thoracic, ct_series)
        lower_vertices, _ = app._structure_surface(lumbar, ct_series)

        self.assertGreater(upper_vertices[:, 2].mean(), lower_vertices[:, 2].mean())
        self.assertAlmostEqual(lower_vertices[:, 2].max(), 17.5, delta=0.15)
        self.assertAlmostEqual(upper_vertices[:, 2].min(), 27.5, delta=0.15)

        figure = app.Figure()
        auto_axes = figure.add_subplot(1, 2, 1, projection="3d")
        medical_axes = figure.add_subplot(1, 2, 2, projection="3d")
        app.draw_structure_3d(
            auto_axes,
            medical_axes,
            cervical_thoracic,
            lumbar,
            ct_series,
        )
        self.assertEqual(auto_axes.get_zlim(), medical_axes.get_zlim())
        self.assertLess(auto_axes.get_zlim()[0], lower_vertices[:, 2].min())
        self.assertGreater(auto_axes.get_zlim()[1], upper_vertices[:, 2].max())

    def test_surface_mesh_is_reduced_for_large_display_masks(self):
        mask = np.zeros((240, 220, 200), dtype=bool)
        mask[28:212, 18:202, 8:192] = True

        vertices, faces = app._structure_surface(mask, None)

        self.assertTrue(len(vertices))
        self.assertTrue(len(faces))
        self.assertLess(len(faces), 100_000)

    def test_export_writes_brazilian_csv_with_volume_percent_and_dose(self):
        curve = app.DVHCurve(
            "ROI",
            np.array([0.0, 1.0]),
            np.array([2.0, 0.5]),
            np.array([100.0, 25.0]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dvh.csv"
            app.export_dvh_csv([curve], output)
            lines = output.read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(lines[0], "Estrutura;Dose (Gy);Volume (cm3);Volume (%)")
        self.assertEqual(lines[1], "ROI;0,0000;2,000000;100,0000")
        self.assertEqual(lines[2], "ROI;1,0000;0,500000;25,0000")

    def test_export_does_not_add_dose_coverage_metadata(self):
        curve = app.DVHCurve(
            "ROI",
            np.array([0.0]),
            np.array([0.5]),
            np.array([100.0]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dvh.csv"

            app.export_dvh_csv([curve], path)

            lines = path.read_text(encoding="utf-8-sig").splitlines()

        self.assertNotIn("cobertura", lines[1].casefold())

    def test_export_writes_two_separate_png_graphs(self):
        curve = app.DVHCurve(
            "ROI",
            np.array([0.0, 1.0]),
            np.array([2.0, 0.5]),
            np.array([100.0, 25.0]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            outputs = app.export_dvh_plots([curve], Path(temporary) / "patient.png")
            self.assertEqual([path.name for path in outputs], [
                "patient_volume_percentual.png",
                "patient_volume_cm3.png",
            ])
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
