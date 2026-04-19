import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy # type: ignore

RAW_DIR = Path("/root/autodl-tmp/projects/Vol2Splat/raw")
CT_NAME = "ct.nii.gz"
SEG_DIR_NAME = "segmentations"
OUT_DIR_NAME = "after_seg_ct"

BACKGROUND_VALUE = -1024


def format_case_name(idx: int) -> str:
    return f"s{idx:04d}"


def ask_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def matrix_to_list(mat):
    if mat is None:
        return None
    return [[mat.GetElement(i, j) for j in range(4)] for i in range(4)]


def matrices_close(a, b, atol=1e-5):
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    for i in range(4):
        for j in range(4):
            if abs(a.GetElement(i, j) - b.GetElement(i, j)) > atol:
                return False
    return True


def apply_nifti_metadata(writer, reader):
    if hasattr(reader, "GetQFac") and hasattr(writer, "SetQFac"):
        writer.SetQFac(reader.GetQFac())

    if hasattr(reader, "GetTimeDimension") and hasattr(writer, "SetTimeDimension"):
        writer.SetTimeDimension(reader.GetTimeDimension())

    qform = reader.GetQFormMatrix() if hasattr(reader, "GetQFormMatrix") else None
    sform = reader.GetSFormMatrix() if hasattr(reader, "GetSFormMatrix") else None

    if qform is not None and hasattr(writer, "SetQFormMatrix"):
        writer.SetQFormMatrix(qform)
    if sform is not None and hasattr(writer, "SetSFormMatrix"):
        writer.SetSFormMatrix(sform)


def read_nifti(path: Path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader


def safe_unique_from_vtk_scalars(scalars, max_items=20):
    arr = vtk_to_numpy(scalars)
    vals = np.unique(arr)
    if len(vals) <= max_items:
        return vals.tolist()
    return vals[:max_items].tolist() + ["..."]


def process_case(case_dir: Path):
    ct_path = case_dir / CT_NAME
    seg_dir = case_dir / SEG_DIR_NAME
    out_dir = case_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    case_log = {
        "case": case_dir.name,
        "ct_path": str(ct_path),
        "seg_dir": str(seg_dir),
        "out_dir": str(out_dir),
        "status": "ok",
        "saved": [],
        "skipped_empty": [],
        "skipped_mismatch": [],
        "errors": [],
    }

    if not ct_path.exists():
        case_log["status"] = "missing_ct"
        case_log["errors"].append(f"missing CT: {ct_path}")
        return case_log

    if not seg_dir.exists():
        case_log["status"] = "missing_segmentations"
        case_log["errors"].append(f"missing segmentation dir: {seg_dir}")
        return case_log

    mask_files = sorted(seg_dir.glob("*.nii.gz"))
    if not mask_files:
        case_log["status"] = "no_masks"
        case_log["errors"].append(f"no masks found in: {seg_dir}")
        return case_log

    # read CT once
    ct_reader = read_nifti(ct_path)
    ct_img = ct_reader.GetOutput()

    ct_dims = ct_img.GetDimensions()
    ct_spacing = ct_img.GetSpacing()
    ct_origin = ct_img.GetOrigin()
    ct_qform = ct_reader.GetQFormMatrix() if hasattr(ct_reader, "GetQFormMatrix") else None
    ct_sform = ct_reader.GetSFormMatrix() if hasattr(ct_reader, "GetSFormMatrix") else None

    for mask_path in mask_files:
        organ_name = mask_path.name[:-7]  # remove .nii.gz

        try:
            mask_reader = read_nifti(mask_path)
            mask_img = mask_reader.GetOutput()

            mask_dims = mask_img.GetDimensions()
            mask_spacing = mask_img.GetSpacing()
            mask_origin = mask_img.GetOrigin()
            mask_qform = mask_reader.GetQFormMatrix() if hasattr(mask_reader, "GetQFormMatrix") else None
            mask_sform = mask_reader.GetSFormMatrix() if hasattr(mask_reader, "GetSFormMatrix") else None

            # geometry checks
            if mask_dims != ct_dims:
                case_log["skipped_mismatch"].append({
                    "organ": organ_name,
                    "reason": "dims_mismatch",
                    "ct_dims": list(ct_dims),
                    "mask_dims": list(mask_dims),
                })
                continue

            if not np.allclose(mask_spacing, ct_spacing, atol=1e-6):
                case_log["skipped_mismatch"].append({
                    "organ": organ_name,
                    "reason": "spacing_mismatch",
                    "ct_spacing": list(ct_spacing),
                    "mask_spacing": list(mask_spacing),
                })
                continue

            if not np.allclose(mask_origin, ct_origin, atol=1e-6):
                case_log["skipped_mismatch"].append({
                    "organ": organ_name,
                    "reason": "origin_mismatch",
                    "ct_origin": list(ct_origin),
                    "mask_origin": list(mask_origin),
                })
                continue

            if not matrices_close(mask_qform, ct_qform):
                case_log["skipped_mismatch"].append({
                    "organ": organ_name,
                    "reason": "qform_mismatch",
                    "ct_qform": matrix_to_list(ct_qform),
                    "mask_qform": matrix_to_list(mask_qform),
                })
                continue

            if not matrices_close(mask_sform, ct_sform):
                case_log["skipped_mismatch"].append({
                    "organ": organ_name,
                    "reason": "sform_mismatch",
                    "ct_sform": matrix_to_list(ct_sform),
                    "mask_sform": matrix_to_list(mask_sform),
                })
                continue

            scalars = mask_img.GetPointData().GetScalars()
            if scalars is None:
                case_log["errors"].append({
                    "organ": organ_name,
                    "mask_path": str(mask_path),
                    "error": "no scalar data",
                })
                continue

            flat_mask = vtk_to_numpy(scalars)
            unique_vals = np.unique(flat_mask)
            non_empty = np.count_nonzero(flat_mask) > 0
            binary_01 = np.all(np.isin(unique_vals, [0, 1]))

            if not non_empty:
                case_log["skipped_empty"].append({
                    "organ": organ_name,
                    "mask_path": str(mask_path),
                    "unique": unique_vals.tolist(),
                    "binary_01": bool(binary_01),
                })
                continue

            # cast mask to unsigned char for vtkImageMask
            cast_mask = vtk.vtkImageCast()
            cast_mask.SetInputData(mask_img)
            cast_mask.SetOutputScalarTypeToUnsignedChar()
            cast_mask.Update()

            # apply mask to CT
            mask_filter = vtk.vtkImageMask()
            mask_filter.SetImageInputData(ct_img)
            mask_filter.SetMaskInputData(cast_mask.GetOutput())
            mask_filter.SetMaskedOutputValue(BACKGROUND_VALUE)
            mask_filter.Update()

            out_path = out_dir / f"{organ_name}_density.nii.gz"

            writer = vtk.vtkNIFTIImageWriter()
            writer.SetFileName(str(out_path))
            writer.SetInputData(mask_filter.GetOutput())
            apply_nifti_metadata(writer, ct_reader)
            writer.Write()

            case_log["saved"].append({
                "organ": organ_name,
                "mask_path": str(mask_path),
                "out_path": str(out_path),
                "foreground_voxels": int(np.count_nonzero(flat_mask)),
                "unique": unique_vals.tolist(),
                "binary_01": bool(binary_01),
            })

        except Exception as e:
            case_log["errors"].append({
                "organ": organ_name,
                "mask_path": str(mask_path),
                "error": str(e),
            })

    case_log_path = out_dir / "preprocess_log.json"
    with open(case_log_path, "w", encoding="utf-8") as f:
        json.dump(case_log, f, indent=2, ensure_ascii=False)

    return case_log


def main():
    print("=" * 80)
    print("VTK TotalSeg density preprocessing")
    print(f"RAW_DIR: {RAW_DIR}")
    print("=" * 80)

    start_num = ask_int("Start number: ")
    end_num = ask_int("End number: ")

    if start_num > end_num:
        start_num, end_num = end_num, start_num

    summary = {
        "raw_dir": str(RAW_DIR),
        "start_num": start_num,
        "end_num": end_num,
        "cases": [],
        "totals": {
            "requested_cases": 0,
            "existing_cases": 0,
            "missing_case_dirs": 0,
            "saved": 0,
            "skipped_empty": 0,
            "skipped_mismatch": 0,
            "errors": 0,
        }
    }

    case_ids = list(range(start_num, end_num + 1))
    summary["totals"]["requested_cases"] = len(case_ids)

    for idx in case_ids:
        case_name = format_case_name(idx)
        case_dir = RAW_DIR / case_name

        if not case_dir.exists():
            print(f"[MISSING] {case_name}")
            summary["cases"].append({
                "case": case_name,
                "status": "missing_case_dir",
            })
            summary["totals"]["missing_case_dirs"] += 1
            continue

        print(f"[PROCESS] {case_name}")
        summary["totals"]["existing_cases"] += 1

        case_log = process_case(case_dir)

        num_saved = len(case_log["saved"])
        num_empty = len(case_log["skipped_empty"])
        num_mismatch = len(case_log["skipped_mismatch"])
        num_errors = len(case_log["errors"])

        print(
            f"  saved={num_saved}, empty={num_empty}, "
            f"mismatch={num_mismatch}, errors={num_errors}"
        )

        summary["cases"].append({
            "case": case_log["case"],
            "status": case_log["status"],
            "num_saved": num_saved,
            "num_skipped_empty": num_empty,
            "num_skipped_mismatch": num_mismatch,
            "num_errors": num_errors,
        })

        summary["totals"]["saved"] += num_saved
        summary["totals"]["skipped_empty"] += num_empty
        summary["totals"]["skipped_mismatch"] += num_mismatch
        summary["totals"]["errors"] += num_errors

    summary_path = RAW_DIR / f"preprocess_summary_{format_case_name(start_num)}_{format_case_name(end_num)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Finished.")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary["totals"], indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()