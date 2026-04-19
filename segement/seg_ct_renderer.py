import json
import random
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


RAW_DIR = Path("/root/autodl-tmp/projects/Vol2Splat/raw")
DEFAULT_OUT_DIR = RAW_DIR / "after_seg_ct_renders"

# after_seg_ct 里的 density 体，器官外部通常是 -1024
BACKGROUND_THRESHOLD = -1023


def format_case_name(idx: int) -> str:
    return f"s{idx:04d}"


def ask_int(prompt: str, default=None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def ask_str(prompt: str, default=None) -> str:
    s = input(prompt).strip()
    if s == "" and default is not None:
        return default
    return s


def collect_density_files(start_num: int, end_num: int):
    files = []
    missing_cases = []

    for idx in range(start_num, end_num + 1):
        case_name = format_case_name(idx)
        case_dir = RAW_DIR / case_name
        after_dir = case_dir / "after_seg_ct"

        if not case_dir.exists():
            missing_cases.append({"case": case_name, "reason": "missing_case_dir"})
            continue
        if not after_dir.exists():
            missing_cases.append({"case": case_name, "reason": "missing_after_seg_ct"})
            continue

        nii_files = sorted(after_dir.glob("*_density.nii.gz"))
        for f in nii_files:
            files.append(f)

    return files, missing_cases


def read_nifti(path: Path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader


def has_foreground(image, threshold=BACKGROUND_THRESHOLD):
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        return False, 0
    arr = vtk_to_numpy(scalars)
    fg = int(np.count_nonzero(arr > threshold))
    return fg > 0, fg


def build_surface_from_density(image, threshold=BACKGROUND_THRESHOLD):
    # 把 density 体阈值成 0/1
    th = vtk.vtkImageThreshold()
    th.SetInputData(image)
    th.ThresholdByUpper(threshold)   # <= threshold 作为背景
    th.ReplaceInOn()
    th.SetInValue(0)
    th.ReplaceOutOn()
    th.SetOutValue(1)
    th.SetOutputScalarTypeToUnsignedChar()
    th.Update()

    # 提取表面
    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputConnection(th.GetOutputPort())
    mc.SetValue(0, 0.5)
    mc.Update()

    # 清理与平滑
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(mc.GetOutputPort())
    clean.Update()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(clean.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.SetRelaxationFactor(0.1)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()

    poly = normals.GetOutput()
    return poly


def render_polydata_to_png(polydata, out_png: Path, image_size=(1200, 1200)):
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        return False, "empty_polydata"

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.75, 0.65)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.15)
    actor.GetProperty().SetSpecularPower(20)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)

    renwin = vtk.vtkRenderWindow()
    renwin.SetOffScreenRendering(1)
    renwin.AddRenderer(renderer)
    renwin.SetSize(*image_size)

    # 灯光
    light1 = vtk.vtkLight()
    light1.SetPosition(1, 1, 1)
    light1.SetFocalPoint(0, 0, 0)
    renderer.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetPosition(-1, -0.5, 0.5)
    light2.SetFocalPoint(0, 0, 0)
    renderer.AddLight(light2)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(35)
    cam.Elevation(25)
    cam.Dolly(1.2)
    renderer.ResetCameraClippingRange()

    renwin.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renwin)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(out_png))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    return True, ""


def main():
    print("=" * 80)
    print("Render spot-check images from after_seg_ct/*_density.nii.gz")
    print(f"RAW_DIR: {RAW_DIR}")
    print("=" * 80)

    start_num = ask_int("Start number: ")
    end_num = ask_int("End number: ")
    if start_num > end_num:
        start_num, end_num = end_num, start_num

    mode = ask_str("Mode [random/first/all] (default=random): ", "random").lower()
    if mode not in {"random", "first", "all"}:
        mode = "random"

    sample_count = None
    if mode != "all":
        sample_count = ask_int("How many files to render? ", 20)

    out_dir_str = ask_str(f"Output folder (default={DEFAULT_OUT_DIR}): ", str(DEFAULT_OUT_DIR))
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files, missing_cases = collect_density_files(start_num, end_num)

    print(f"\nFound {len(all_files)} density files.")

    if len(all_files) == 0:
        print("No files found.")
        manifest = {
            "start_num": start_num,
            "end_num": end_num,
            "mode": mode,
            "output_dir": str(out_dir),
            "missing_cases": missing_cases,
            "rendered": [],
            "failed": [],
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return

    if mode == "random":
        k = min(sample_count, len(all_files))
        selected_files = random.sample(all_files, k)
    elif mode == "first":
        k = min(sample_count, len(all_files))
        selected_files = all_files[:k]
    else:
        selected_files = all_files

    rendered = []
    failed = []

    for i, nii_path in enumerate(selected_files, 1):
        case_name = nii_path.parent.parent.name
        organ_name = nii_path.name.replace("_density.nii.gz", "")
        out_png = out_dir / f"{case_name}__{organ_name}.png"

        print(f"[{i}/{len(selected_files)}] {case_name} - {organ_name}")

        try:
            reader = read_nifti(nii_path)
            image = reader.GetOutput()

            ok_fg, fg_voxels = has_foreground(image)
            if not ok_fg:
                failed.append({
                    "case": case_name,
                    "organ": organ_name,
                    "nii_path": str(nii_path),
                    "reason": "empty_foreground",
                })
                print("  skip: empty foreground")
                continue

            poly = build_surface_from_density(image)
            ok, reason = render_polydata_to_png(poly, out_png)

            if ok:
                rendered.append({
                    "case": case_name,
                    "organ": organ_name,
                    "nii_path": str(nii_path),
                    "png_path": str(out_png),
                    "foreground_voxels": fg_voxels,
                    "num_points": int(poly.GetNumberOfPoints()),
                    "num_polys": int(poly.GetNumberOfPolys()),
                })
                print(f"  saved: {out_png}")
            else:
                failed.append({
                    "case": case_name,
                    "organ": organ_name,
                    "nii_path": str(nii_path),
                    "reason": reason,
                })
                print(f"  failed: {reason}")

        except Exception as e:
            failed.append({
                "case": case_name,
                "organ": organ_name,
                "nii_path": str(nii_path),
                "reason": str(e),
            })
            print(f"  exception: {e}")

    manifest = {
        "start_num": start_num,
        "end_num": end_num,
        "mode": mode,
        "selected_count": len(selected_files),
        "output_dir": str(out_dir),
        "missing_cases": missing_cases,
        "rendered": rendered,
        "failed": failed,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Rendered: {len(rendered)}")
    print(f"Failed  : {len(failed)}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()