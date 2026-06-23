from __future__ import annotations

import argparse
import os
import csv
import math
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Button, RectangleSelector
from scipy.ndimage import gaussian_filter, label as label_components, median_filter
from tkinter import Tk, filedialog


DEFAULT_DATA_DIR = Path(r"F:\05-Jerome Studios\计算光谱专项\测试数据\003")
DEFAULT_EPS = 5.0
DEFAULT_SNR = 3.0
DEFAULT_SPATIAL_SIGMA = 0.8
DEFAULT_MIN_VALID_FRACTION = 0.05
DEFAULT_MIN_DIVISION_PIXELS = 30
DEFAULT_MIN_COMPONENT_PIXELS = 8
DEFAULT_CWL_MIN_PEAK_FRACTION = 0.10
DEFAULT_ROI_TRIM_FRACTION = 0.05
DEFAULT_MIN_PEAK_CONFIDENCE = 0.18
DEFAULT_CWL_MIN_PROMINENCE_FRACTION = 0.12
DEFAULT_CWL_MEDIAN_FILTER_SIZE = 3
DEFAULT_CWL_DOMINANT_WINDOW_NM = 24.0
DEFAULT_GRID_ROWS = 5
DEFAULT_GRID_COLS = 5
DEFAULT_GRID_ROI_FRACTION = 0.30
DEFAULT_AUTO_WORKERS = min(4, max(1, os.cpu_count() or 1))
DEFAULT_ARTIFACT_MIN_NM = 500.0
DEFAULT_ARTIFACT_MAX_NM = 600.0
DEFAULT_ARTIFACT_MODE = "mark"
THEME_FIG_BG = "#edf1f5"
THEME_AX_BG = "#ffffff"
THEME_PANEL_BG = "#f8fafc"
THEME_TEXT = "#203040"
THEME_MUTED_TEXT = "#536579"
THEME_GRID = "#d8e1ea"
THEME_ACCENT = "#2f6f9f"
THEME_BUTTON = "#dbe7f1"
THEME_BUTTON_HOVER = "#c7dbea"
THEME_BORDER = "#8da2b6"
ROI_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def style_plot_axis(ax) -> None:
    ax.set_facecolor(THEME_AX_BG)
    ax.title.set_color(THEME_TEXT)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_color(THEME_MUTED_TEXT)
    ax.yaxis.label.set_color(THEME_MUTED_TEXT)
    ax.tick_params(colors=THEME_MUTED_TEXT)
    for spine in ax.spines.values():
        spine.set_color(THEME_BORDER)


def style_control_axis(ax) -> None:
    ax.set_facecolor(THEME_PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(THEME_BORDER)


def style_button(button: Button) -> None:
    button.label.set_color(THEME_TEXT)
    button.label.set_fontsize(8.5)
    button.label.set_fontweight("bold")
    style_control_axis(button.ax)
    button.ax.set_facecolor(THEME_BUTTON)


def configure_gui_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    fonts = [font for font in preferred_fonts if font in available_fonts]
    if not fonts:
        fonts = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


ENVI_DTYPES = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    12: "u2",
    13: "u4",
    14: "i8",
    15: "u8",
}


@dataclass(frozen=True)
class EnviMeta:
    hdr_path: Path
    raw_path: Path
    lines: int
    samples: int
    bands: int
    dtype: np.dtype
    interleave: str
    header_offset: int
    wavelengths: tuple[float, ...]


@dataclass(frozen=True)
class CubeData:
    label: str
    meta: EnviMeta
    data: np.memmap


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class ReflectanceOptions:
    eps: float
    snr: float
    spatial_sigma: float
    min_valid_fraction: float
    min_division_pixels: int
    min_component_pixels: int
    artifact_min_nm: float
    artifact_max_nm: float
    artifact_mode: str


@dataclass(frozen=True)
class ReflectanceResult:
    values: np.ndarray
    signal: np.ndarray
    ref_signal: np.ndarray
    dark_std: float
    signal_threshold: float
    ref_threshold: float
    low_snr_count: int
    low_signal_snr_count: int
    low_ref_snr_count: int
    negative_signal_count: int
    negative_ref_signal_count: int
    prezero_count: int


def maybe_spatial_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    return gaussian_filter(image, sigma=sigma, mode="nearest")


def remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not np.any(mask):
        return mask
    labels, count = label_components(mask)
    if count == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    keep_labels = component_sizes >= min_pixels
    keep_labels[0] = False
    return keep_labels[labels]


@dataclass(frozen=True)
class RoiRecord:
    roi_id: int
    roi: Roi
    rows: list[dict[str, float | int | str]]
    metrics: dict[str, float | str]
    color: str
    patch: Rectangle | None
    label: Text | None


@dataclass(frozen=True)
class CwlDriftResult:
    cwl_map: np.ndarray
    compute_backend: str


def find_first_file(folder: Path, suffix: str) -> Path:
    files = sorted(folder.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No {suffix} file found in {folder}")
    if len(files) > 1:
        raise ValueError(f"More than one {suffix} file found in {folder}: {files}")
    return files[0]


def find_cube_folders(data_dir: Path) -> dict[str, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    folders: dict[str, Path] = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        name = folder.name.lower()
        if name.endswith("-rec") or "-rec" in name:
            folders["sign"] = folder
        elif name.endswith("-ref") or "-ref" in name:
            folders["ref"] = folder
        elif name.endswith("-dark") or "-dark" in name:
            folders["dark"] = folder

    missing = {"sign", "ref", "dark"} - folders.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Missing cube folder(s): {names}")
    return folders


def looks_like_dataset_dir(data_dir: Path) -> bool:
    try:
        children = [child for child in data_dir.iterdir() if child.is_dir()]
    except OSError:
        return False
    names = [child.name.lower() for child in children]
    has_sign = any(name.endswith("-rec") or "-rec" in name for name in names)
    has_ref = any(name.endswith("-ref") or "-ref" in name for name in names)
    has_dark = any(name.endswith("-dark") or "-dark" in name for name in names)
    return has_sign and has_ref and has_dark


def resolve_data_dir(data_dir: Path) -> Path:
    if data_dir.exists():
        return data_dir

    search_root = Path(r"F:\05-Jerome Studios")
    if search_root.exists():
        candidates = [
            path
            for path in search_root.rglob(data_dir.name)
            if path.is_dir() and looks_like_dataset_dir(path)
        ]
        if candidates:
            return sorted(candidates, key=lambda path: (len(str(path)), str(path)))[0]

    return data_dir


def parse_scalar(text: str, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def parse_number_block(text: str, key: str) -> tuple[float, ...]:
    pattern = re.compile(rf"{re.escape(key)}\s*=\s*\{{(.*?)\}}", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ()
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))
    return tuple(float(value) for value in values)


def envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    if data_type not in ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")
    dtype = np.dtype(ENVI_DTYPES[data_type])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if byte_order == 0 else ">")
    return dtype


def parse_envi_meta(folder: Path) -> EnviMeta:
    hdr_path = find_first_file(folder, ".hdr")
    raw_path = find_first_file(folder, ".raw")
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")

    required_keys = ("lines", "samples", "bands", "data type", "byte order", "interleave")
    values = {key: parse_scalar(text, key) for key in required_keys}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"Missing ENVI header field(s) in {hdr_path}: {missing}")

    lines = int(values["lines"])
    samples = int(values["samples"])
    bands = int(values["bands"])
    dtype = envi_dtype(int(values["data type"]), int(values["byte order"]))
    interleave = str(values["interleave"]).strip().lower()
    header_offset = int(parse_scalar(text, "header offset") or "0")
    wavelengths = parse_number_block(text, "wavelength")

    expected_size = header_offset + lines * samples * bands * dtype.itemsize
    actual_size = raw_path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"RAW size mismatch for {raw_path}: expected {expected_size} bytes, got {actual_size} bytes"
        )

    if wavelengths and len(wavelengths) != bands:
        raise ValueError(
            f"Wavelength count mismatch for {hdr_path}: expected {bands}, got {len(wavelengths)}"
        )

    if not wavelengths:
        wavelengths = tuple(float(i) for i in range(bands))

    return EnviMeta(
        hdr_path=hdr_path,
        raw_path=raw_path,
        lines=lines,
        samples=samples,
        bands=bands,
        dtype=dtype,
        interleave=interleave,
        header_offset=header_offset,
        wavelengths=wavelengths,
    )


def memmap_shape(meta: EnviMeta) -> tuple[int, ...]:
    if meta.interleave == "bsq":
        return (meta.bands, meta.lines, meta.samples)
    if meta.interleave == "bil":
        return (meta.lines, meta.bands, meta.samples)
    if meta.interleave == "bip":
        return (meta.lines, meta.samples, meta.bands)
    raise ValueError(f"Unsupported interleave: {meta.interleave}")


def load_cube(label: str, folder: Path) -> CubeData:
    meta = parse_envi_meta(folder)
    data = np.memmap(
        meta.raw_path,
        dtype=meta.dtype,
        mode="r",
        offset=meta.header_offset,
        shape=memmap_shape(meta),
    )
    return CubeData(label=label, meta=meta, data=data)


def assert_matching_cubes(cubes: Iterable[CubeData]) -> None:
    cubes = list(cubes)
    first = cubes[0].meta
    for cube in cubes[1:]:
        meta = cube.meta
        if (meta.lines, meta.samples, meta.bands) != (first.lines, first.samples, first.bands):
            raise ValueError(f"Cube shape mismatch: {cube.label} differs from {cubes[0].label}")
        if meta.wavelengths != first.wavelengths:
            raise ValueError(f"Wavelength mismatch: {cube.label} differs from {cubes[0].label}")


def get_band(cube: CubeData, band_index: int) -> np.ndarray:
    if cube.meta.interleave == "bsq":
        return cube.data[band_index, :, :]
    if cube.meta.interleave == "bil":
        return cube.data[:, band_index, :]
    if cube.meta.interleave == "bip":
        return cube.data[:, :, band_index]
    raise ValueError(f"Unsupported interleave: {cube.meta.interleave}")


def get_roi_band(cube: CubeData, band_index: int, roi: Roi) -> np.ndarray:
    if cube.meta.interleave == "bsq":
        return cube.data[band_index, roi.y : roi.y1, roi.x : roi.x1]
    if cube.meta.interleave == "bil":
        return cube.data[roi.y : roi.y1, band_index, roi.x : roi.x1]
    if cube.meta.interleave == "bip":
        return cube.data[roi.y : roi.y1, roi.x : roi.x1, band_index]
    raise ValueError(f"Unsupported interleave: {cube.meta.interleave}")


def compute_reflectance(
    sign: np.ndarray,
    ref: np.ndarray,
    dark: np.ndarray,
    options: ReflectanceOptions,
) -> ReflectanceResult:
    sign_f = sign.astype(np.float32, copy=False)
    ref_f = ref.astype(np.float32, copy=False)
    dark_f = dark.astype(np.float32, copy=False)
    signal = sign_f
    ref_signal = ref_f
    dark_std = float(np.std(dark_f))
    signal = maybe_spatial_gaussian(signal, options.spatial_sigma)
    ref_signal = maybe_spatial_gaussian(ref_signal, options.spatial_sigma)
    signal_threshold = max(options.eps, options.snr * dark_std)
    ref_threshold = max(options.eps, options.snr * dark_std)
    negative_signal = signal < 0
    negative_ref_signal = np.zeros(ref_signal.shape, dtype=bool)
    prezero = negative_signal
    low_signal_snr = (~prezero) & (signal <= signal_threshold)
    low_ref_snr = np.zeros(ref_signal.shape, dtype=bool)
    low_snr = low_signal_snr
    valid = (~prezero) & (~low_snr)
    connected_valid = remove_small_components(valid, options.min_component_pixels)
    isolated_valid = valid & (~connected_valid)
    low_snr = low_snr | isolated_valid
    valid = connected_valid
    out = np.full(signal.shape, np.nan, dtype=np.float32)
    out[prezero] = 0.0
    out[low_snr] = 0.0
    out[valid] = signal[valid]

    low_snr_count = int(np.count_nonzero(low_snr))
    low_signal_snr_count = int(np.count_nonzero(low_signal_snr))
    low_ref_snr_count = int(np.count_nonzero(low_ref_snr))
    negative_signal_count = int(np.count_nonzero(negative_signal))
    negative_ref_signal_count = int(np.count_nonzero(negative_ref_signal))
    prezero_count = int(np.count_nonzero(prezero))
    return ReflectanceResult(
        values=out,
        signal=signal,
        ref_signal=ref_signal,
        dark_std=dark_std,
        signal_threshold=signal_threshold,
        ref_threshold=ref_threshold,
        low_snr_count=low_snr_count,
        low_signal_snr_count=low_signal_snr_count,
        low_ref_snr_count=low_ref_snr_count,
        negative_signal_count=negative_signal_count,
        negative_ref_signal_count=negative_ref_signal_count,
        prezero_count=prezero_count,
    )


def clamp_roi(roi: Roi, samples: int, lines: int) -> Roi:
    x0 = max(0, min(samples, roi.x))
    y0 = max(0, min(lines, roi.y))
    x1 = max(0, min(samples, roi.x1))
    y1 = max(0, min(lines, roi.y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 == x0 or y1 == y0:
        raise ValueError(f"Empty ROI after clamping: {roi}")
    return Roi(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def roi_from_selector(x0: float, y0: float, x1: float, y1: float, samples: int, lines: int) -> Roi:
    roi = Roi(
        x=math.floor(min(x0, x1)),
        y=math.floor(min(y0, y1)),
        width=math.ceil(max(x0, x1)) - math.floor(min(x0, x1)),
        height=math.ceil(max(y0, y1)) - math.floor(min(y0, y1)),
    )
    return clamp_roi(roi, samples=samples, lines=lines)


def auto_grid_rois(
    samples: int,
    lines: int,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    area_fraction: float = DEFAULT_GRID_ROI_FRACTION,
) -> list[Roi]:
    scale = math.sqrt(max(0.0, min(1.0, area_fraction)))
    x_edges = np.linspace(0, samples, cols + 1)
    y_edges = np.linspace(0, lines, rows + 1)
    rois: list[Roi] = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(round(x_edges[col]))
            x1 = int(round(x_edges[col + 1]))
            y0 = int(round(y_edges[row]))
            y1 = int(round(y_edges[row + 1]))
            block_width = max(1, x1 - x0)
            block_height = max(1, y1 - y0)
            roi_width = max(1, int(round(block_width * scale)))
            roi_height = max(1, int(round(block_height * scale)))
            roi_x = x0 + (block_width - roi_width) // 2
            roi_y = y0 + (block_height - roi_height) // 2
            rois.append(clamp_roi(Roi(roi_x, roi_y, roi_width, roi_height), samples=samples, lines=lines))
    return rois


def padded_roi(roi: Roi, samples: int, lines: int, sigma: float) -> tuple[Roi, tuple[slice, slice]]:
    if sigma <= 0:
        return roi, (slice(None), slice(None))
    pad = max(1, int(math.ceil(3 * sigma)))
    padded = clamp_roi(
        Roi(x=roi.x - pad, y=roi.y - pad, width=roi.width + 2 * pad, height=roi.height + 2 * pad),
        samples=samples,
        lines=lines,
    )
    y_start = roi.y - padded.y
    x_start = roi.x - padded.x
    return padded, (slice(y_start, y_start + roi.height), slice(x_start, x_start + roi.width))


def is_artifact_wavelength(wavelength: float, options: ReflectanceOptions) -> bool:
    return options.artifact_min_nm <= wavelength <= options.artifact_max_nm


def apply_artifact_mode(
    rows: list[dict[str, float | int | str]],
    options: ReflectanceOptions,
) -> None:
    if options.artifact_mode == "none":
        return

    for row in rows:
        wavelength = float(row["wavelength_nm"])
        if not is_artifact_wavelength(wavelength, options):
            row["artifact_status"] = "normal"
            continue

        row["artifact_status"] = options.artifact_mode


def compute_roi_stats(
    cubes: dict[str, CubeData],
    roi: Roi,
    options: ReflectanceOptions,
    roi_id: int,
) -> list[dict[str, float | int | str]]:
    wavelengths = cubes["sign"].meta.wavelengths
    meta = cubes["sign"].meta
    total_pixels = roi.width * roi.height
    compute_roi, roi_slices = padded_roi(roi, meta.samples, meta.lines, options.spatial_sigma)
    rows: list[dict[str, float | int | str]] = []

    for band_index, wavelength in enumerate(wavelengths):
        sign_roi = get_roi_band(cubes["sign"], band_index, compute_roi)
        ref_roi = get_roi_band(cubes["ref"], band_index, compute_roi)
        dark_roi = get_roi_band(cubes["dark"], band_index, compute_roi)
        result = compute_reflectance(
            sign_roi,
            ref_roi,
            dark_roi,
            options=options,
        )
        values = result.values[roi_slices]
        signal = result.signal[roi_slices]
        ref_signal = result.ref_signal[roi_slices]
        negative_signal = signal < 0
        negative_ref_signal = np.zeros(ref_signal.shape, dtype=bool)
        prezero = negative_signal
        low_signal_snr = (~prezero) & (signal <= result.signal_threshold)
        low_ref_snr = np.zeros(ref_signal.shape, dtype=bool)
        low_snr = low_signal_snr
        valid_mask = (~prezero) & (~low_snr)
        connected_valid = remove_small_components(valid_mask, options.min_component_pixels)
        isolated_valid = valid_mask & (~connected_valid)
        low_snr = low_snr | isolated_valid
        result = ReflectanceResult(
            values=values,
            signal=signal,
            ref_signal=ref_signal,
            dark_std=result.dark_std,
            signal_threshold=result.signal_threshold,
            ref_threshold=result.ref_threshold,
            low_snr_count=int(np.count_nonzero(low_snr)),
            low_signal_snr_count=int(np.count_nonzero(low_signal_snr)),
            low_ref_snr_count=int(np.count_nonzero(low_ref_snr)),
            negative_signal_count=int(np.count_nonzero(negative_signal)),
            negative_ref_signal_count=int(np.count_nonzero(negative_ref_signal)),
            prezero_count=int(np.count_nonzero(prezero)),
        )
        reflectance = result.values
        finite = np.isfinite(reflectance)
        positive_division = (
            finite
            & (result.signal > result.signal_threshold)
            & connected_valid
        )
        valid_count = int(np.count_nonzero(finite))
        division_count = int(np.count_nonzero(positive_division))
        valid_fraction = valid_count / total_pixels
        division_fraction = division_count / total_pixels
        raw_mean = float("nan")
        raw_std = float("nan")
        raw_min_value = float("nan")
        raw_max_value = float("nan")
        raw_ratio_of_means = float("nan")
        robust_mean = float("nan")
        robust_std = float("nan")
        robust_min_value = float("nan")
        robust_max_value = float("nan")
        if division_count:
            valid_values = reflectance[finite]
            raw_mean = float(np.mean(valid_values))
            raw_std = float(np.std(valid_values))
            raw_min_value = float(np.min(valid_values))
            raw_max_value = float(np.max(valid_values))
            robust_mean, robust_std, robust_min_value, robust_max_value = robust_stats(reflectance[positive_division])
            ref_signal_mean = float(np.mean(result.ref_signal[positive_division]))
            signal_mean = float(np.mean(result.signal[positive_division]))
            raw_ratio_of_means = robust_mean
        elif valid_count:
            valid_values = reflectance[finite]
            raw_mean = float(np.mean(valid_values))
            raw_std = float(np.std(valid_values))
            raw_min_value = float(np.min(valid_values))
            raw_max_value = float(np.max(valid_values))
            robust_mean, robust_std, robust_min_value, robust_max_value = robust_stats(valid_values)
            signal_mean = ref_signal_mean = float("nan")
            raw_ratio_of_means = 0.0
        else:
            signal_mean = ref_signal_mean = float("nan")

        enough_fraction = division_fraction >= options.min_valid_fraction
        enough_pixels = division_count >= options.min_division_pixels
        if enough_fraction and enough_pixels:
            quality_status = "ok"
            metric_mean = robust_mean
            metric_std = robust_std
            metric_min_value = robust_min_value
            metric_max_value = robust_max_value
            metric_ratio_of_means = raw_ratio_of_means
        else:
            if not enough_pixels:
                quality_status = "low_division_pixels"
            elif not enough_fraction:
                blockers = {
                    "prezero": result.prezero_count,
                    "low_signal_snr": result.low_signal_snr_count,
                    "low_ref_snr": result.low_ref_snr_count,
                }
                quality_status = max(blockers, key=blockers.get)
            else:
                quality_status = "low_quality"
            raw_ratio_of_means = 0.0
            raw_mean = 0.0 if np.isfinite(raw_mean) else raw_mean
            metric_mean = metric_std = metric_min_value = metric_max_value = metric_ratio_of_means = float("nan")

        rows.append(
            {
                "roi_id": roi_id,
                "band_index": band_index,
                "wavelength_nm": float(wavelength),
                "raw_pixel_mean_reflectance": raw_mean,
                "raw_mean_reflectance": raw_mean,
                "pixel_mean_reflectance": robust_mean,
                "mean_reflectance": raw_ratio_of_means,
                "metric_pixel_mean_reflectance": metric_mean,
                "metric_mean_reflectance": metric_ratio_of_means,
                "signal_mean": signal_mean,
                "ref_signal_mean": ref_signal_mean,
                "raw_std_reflectance": raw_std,
                "raw_min_reflectance": raw_min_value,
                "raw_max_reflectance": raw_max_value,
                "std_reflectance": raw_std,
                "min_reflectance": raw_min_value,
                "max_reflectance": raw_max_value,
                "metric_std_reflectance": metric_std,
                "metric_min_reflectance": metric_min_value,
                "metric_max_reflectance": metric_max_value,
                "valid_pixel_count": valid_count,
                "division_pixel_count": division_count,
                "total_pixel_count": total_pixels,
                "valid_fraction": valid_fraction,
                "division_fraction": division_fraction,
                "min_division_pixels": options.min_division_pixels,
                "min_component_pixels": options.min_component_pixels,
                "quality_status": quality_status,
                "low_snr_count": result.low_snr_count,
                "low_signal_snr_count": result.low_signal_snr_count,
                "low_ref_snr_count": result.low_ref_snr_count,
                "negative_signal_count": result.negative_signal_count,
                "negative_ref_signal_count": result.negative_ref_signal_count,
                "prezero_count": result.prezero_count,
                "dark_std": result.dark_std,
                "signal_threshold": result.signal_threshold,
                "ref_threshold": result.ref_threshold,
                "spatial_sigma": options.spatial_sigma,
                "roi_x": roi.x,
                "roi_y": roi.y,
                "roi_width": roi.width,
                "roi_height": roi.height,
                "artifact_status": "normal",
            }
        )
    apply_artifact_mode(rows, options)
    return rows


def rows_with_metrics(records: Iterable[RoiRecord]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for record in records:
        for row in record.rows:
            rows.append({**row, **record.metrics})
    return rows


def write_stats_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).with_name(f"roi_reflectance_{stamp}.csv")


def robust_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def finite_range_text(values: np.ndarray, precision: int) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no valid pixels"
    return f"{np.min(finite):.{precision}g} to {np.max(finite):.{precision}g}"


def trimmed_values(values: np.ndarray, trim_fraction: float = DEFAULT_ROI_TRIM_FRACTION) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite
    trim_count = int(finite.size * max(0.0, min(0.45, trim_fraction)))
    if trim_count <= 0 or finite.size <= 2 * trim_count:
        return finite
    sorted_values = np.sort(finite)
    return sorted_values[trim_count:-trim_count]


def robust_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    kept = trimmed_values(values)
    if kept.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(kept)), float(np.std(kept)), float(np.min(kept)), float(np.max(kept))


def curve_peak_confidence(reflectance: np.ndarray) -> dict[str, float]:
    y = np.asarray(reflectance, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size < 3:
        return {
            "peak_prominence": float("nan"),
            "peak_confidence": float("nan"),
            "background_level": float("nan"),
            "background_noise": float("nan"),
        }
    peak_value = float(np.max(finite))
    background = float(np.median(finite))
    mad = float(np.median(np.abs(finite - background)))
    background_noise = 1.4826 * mad
    prominence = peak_value - background
    confidence = prominence / (prominence + background_noise + 1e-6) if prominence > 0 else 0.0
    return {
        "peak_prominence": prominence,
        "peak_confidence": float(confidence),
        "background_level": background,
        "background_noise": background_noise,
    }


def interpolate_x_at_y(x0: float, y0: float, x1: float, y1: float, target_y: float) -> float:
    if y1 == y0:
        return (x0 + x1) / 2.0
    return x0 + (target_y - y0) * (x1 - x0) / (y1 - y0)


def compute_peak_metrics(wavelengths: tuple[float, ...], reflectance: np.ndarray) -> dict[str, float | str]:
    x = np.array(wavelengths, dtype=float)
    y = np.array(reflectance, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return {
            "peak_wavelength_nm": float("nan"),
            "peak_reflectance": float("nan"),
            "half_max_reflectance": float("nan"),
            "fwhm_nm": float("nan"),
            "cwl_nm": float("nan"),
            "peak_prominence": float("nan"),
            "peak_confidence": float("nan"),
            "background_level": float("nan"),
            "background_noise": float("nan"),
            "secondary_peak_wavelength_nm": float("nan"),
            "secondary_peak_reflectance": float("nan"),
            "secondary_peak_ratio": float("nan"),
            "minor_peak_count_10pct": 0,
            "metric_status": "not_enough_valid_points",
        }

    valid_indices = np.flatnonzero(finite)
    peak_pos = int(valid_indices[np.argmax(y[finite])])
    peak_value = float(y[peak_pos])
    if peak_value <= 0:
        return {
            "peak_wavelength_nm": float(x[peak_pos]),
            "peak_reflectance": peak_value,
            "half_max_reflectance": float("nan"),
            "fwhm_nm": float("nan"),
            "cwl_nm": float("nan"),
            "peak_prominence": float("nan"),
            "peak_confidence": 0.0,
            "background_level": float("nan"),
            "background_noise": float("nan"),
            "secondary_peak_wavelength_nm": float("nan"),
            "secondary_peak_reflectance": float("nan"),
            "secondary_peak_ratio": float("nan"),
            "minor_peak_count_10pct": 0,
            "metric_status": "non_positive_peak",
        }

    half_max = peak_value / 2.0
    left_cross = float("nan")
    right_cross = float("nan")

    for idx in range(peak_pos - 1, -1, -1):
        if not (np.isfinite(y[idx]) and np.isfinite(y[idx + 1])):
            continue
        if (y[idx] - half_max) * (y[idx + 1] - half_max) <= 0:
            left_cross = interpolate_x_at_y(float(x[idx]), float(y[idx]), float(x[idx + 1]), float(y[idx + 1]), half_max)
            break

    for idx in range(peak_pos, len(y) - 1):
        if not (np.isfinite(y[idx]) and np.isfinite(y[idx + 1])):
            continue
        if (y[idx] - half_max) * (y[idx + 1] - half_max) <= 0:
            right_cross = interpolate_x_at_y(float(x[idx]), float(y[idx]), float(x[idx + 1]), float(y[idx + 1]), half_max)
            break

    if not (np.isfinite(left_cross) and np.isfinite(right_cross) and right_cross > left_cross):
        status = "half_max_crossing_not_found"
        fwhm = cwl = float("nan")
    else:
        status = "ok"
        fwhm = right_cross - left_cross
        cwl = (left_cross + right_cross) / 2.0

    secondary_peak_wavelength = float("nan")
    secondary_peak_reflectance = float("nan")
    secondary_peak_ratio = float("nan")
    minor_peak_count = 0
    local_peak_indices: list[int] = []
    for idx in range(1, len(y) - 1):
        if idx == peak_pos or not (np.isfinite(y[idx - 1]) and np.isfinite(y[idx]) and np.isfinite(y[idx + 1])):
            continue
        if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1] and y[idx] > 0:
            local_peak_indices.append(idx)

    if local_peak_indices:
        local_peak_indices.sort(key=lambda idx: float(y[idx]), reverse=True)
        secondary_idx = local_peak_indices[0]
        secondary_peak_wavelength = float(x[secondary_idx])
        secondary_peak_reflectance = float(y[secondary_idx])
        secondary_peak_ratio = secondary_peak_reflectance / peak_value if peak_value > 0 else float("nan")
        minor_peak_count = sum(1 for idx in local_peak_indices if float(y[idx]) >= 0.1 * peak_value)

    confidence = curve_peak_confidence(y)
    if status == "ok":
        if np.isfinite(secondary_peak_ratio) and secondary_peak_ratio >= 0.9:
            status = "ambiguous_peak"
        elif confidence["peak_confidence"] < DEFAULT_MIN_PEAK_CONFIDENCE:
            status = "low_peak_confidence"
    return {
        "peak_wavelength_nm": float(x[peak_pos]),
        "peak_reflectance": peak_value,
        "half_max_reflectance": half_max,
        "fwhm_nm": float(fwhm),
        "cwl_nm": float(cwl),
        **confidence,
        "secondary_peak_wavelength_nm": secondary_peak_wavelength,
        "secondary_peak_reflectance": secondary_peak_reflectance,
        "secondary_peak_ratio": secondary_peak_ratio,
        "minor_peak_count_10pct": minor_peak_count,
        "metric_status": status,
    }


def compute_peak_metrics_with_fallback(
    wavelengths: tuple[float, ...],
    metric_reflectance: np.ndarray,
    display_reflectance: np.ndarray,
) -> dict[str, float | str]:
    metrics = compute_peak_metrics(wavelengths, metric_reflectance)
    if (
        metrics["metric_status"] == "half_max_crossing_not_found"
        and not np.isfinite(float(metrics["fwhm_nm"]))
    ):
        fallback = compute_peak_metrics(wavelengths, display_reflectance)
        if fallback["metric_status"] == "ok":
            fallback["metric_status"] = "fallback_raw_half_max"
            return fallback
    return metrics


def wavelength_limits(wavelengths: tuple[float, ...]) -> tuple[float, float]:
    start = float(wavelengths[0])
    end = float(wavelengths[-1])
    if start == end:
        return start - 0.5, end + 0.5
    return min(start, end), max(start, end)


def make_preview(cubes: dict[str, CubeData], band_index: int, options: ReflectanceOptions) -> np.ndarray:
    return compute_reflectance(
        get_band(cubes["sign"], band_index),
        get_band(cubes["ref"], band_index),
        get_band(cubes["dark"], band_index),
        options=options,
    ).values


def make_cwl_drift_map_cpu(cubes: dict[str, CubeData], options: ReflectanceOptions) -> np.ndarray:
    meta = cubes["sign"].meta
    wavelengths = np.array(meta.wavelengths, dtype=np.float32)
    if meta.interleave == "bsq":
        sign_cube = np.asarray(cubes["sign"].data)
        peak_indices = np.argmax(sign_cube, axis=0)
        peak_signal = np.take_along_axis(sign_cube, peak_indices[None, :, :], axis=0)[0]
        background_signal = np.median(sign_cube, axis=0)

        thresholds = np.empty(meta.bands, dtype=np.float32)
        for band_index in range(meta.bands):
            dark_band = get_band(cubes["dark"], band_index).astype(np.float32, copy=False)
            thresholds[band_index] = max(options.eps, options.snr * float(np.std(dark_band)))
        peak_threshold = thresholds[peak_indices]

        cwl_map = wavelengths[peak_indices]
        finite_peak = peak_signal[np.isfinite(peak_signal)]
        min_peak_signal = 0.0
        if finite_peak.size:
            min_peak_signal = DEFAULT_CWL_MIN_PEAK_FRACTION * float(np.max(finite_peak))
        prominence = peak_signal.astype(np.float32, copy=False) - background_signal.astype(np.float32, copy=False)
        min_prominence = DEFAULT_CWL_MIN_PROMINENCE_FRACTION * np.maximum(peak_signal.astype(np.float32, copy=False), 1.0)
        valid = (
            np.isfinite(peak_signal)
            & (peak_signal > peak_threshold)
            & (peak_signal >= min_peak_signal)
            & (prominence >= min_prominence)
        )
        if options.min_component_pixels > 1:
            valid = remove_small_components(valid, options.min_component_pixels)
        cwl_map = cwl_map.astype(np.float32, copy=False)
        cwl_map[~valid] = np.nan
        return keep_dominant_cwl_window(smooth_cwl_map(cwl_map))

    best_reflectance = np.full((meta.lines, meta.samples), -np.inf, dtype=np.float32)
    cwl_map = np.full((meta.lines, meta.samples), np.nan, dtype=np.float32)
    for band_index, wavelength in enumerate(wavelengths):
        reflectance = make_preview(cubes, band_index, options)
        valid = np.isfinite(reflectance) & (reflectance > best_reflectance)
        best_reflectance[valid] = reflectance[valid]
        cwl_map[valid] = wavelength
    cwl_map[~np.isfinite(best_reflectance)] = np.nan
    cwl_map[best_reflectance <= 0] = np.nan
    return keep_dominant_cwl_window(smooth_cwl_map(cwl_map))


def make_cwl_drift_map_gpu(cubes: dict[str, CubeData], options: ReflectanceOptions) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA torch is not available") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA torch is not available")
    device = torch.device("cuda")
    meta = cubes["sign"].meta
    wavelengths = np.array(meta.wavelengths, dtype=np.float32)
    best_reflectance = torch.full((meta.lines, meta.samples), -float("inf"), dtype=torch.float32, device=device)
    cwl_map = torch.full((meta.lines, meta.samples), float("nan"), dtype=torch.float32, device=device)
    for band_index, wavelength in enumerate(wavelengths):
        sign = torch.as_tensor(np.asarray(get_band(cubes["sign"], band_index)), dtype=torch.float32, device=device)
        ref = torch.as_tensor(np.asarray(get_band(cubes["ref"], band_index)), dtype=torch.float32, device=device)
        dark = torch.as_tensor(np.asarray(get_band(cubes["dark"], band_index)), dtype=torch.float32, device=device)
        dark_std = torch.std(dark)
        threshold = max(options.eps, options.snr * float(dark_std.detach().cpu()))
        signal = sign
        ref_signal = ref
        prezero = signal < 0
        low_snr = (~prezero) & (signal <= threshold)
        valid = (~prezero) & (~low_snr)
        reflectance = torch.zeros_like(signal)
        reflectance[valid] = signal[valid]
        better = valid & torch.isfinite(reflectance) & (reflectance > best_reflectance)
        best_reflectance[better] = reflectance[better]
        cwl_map[better] = float(wavelength)
    cwl_map[~torch.isfinite(best_reflectance)] = float("nan")
    cwl_map[best_reflectance <= 0] = float("nan")
    return cwl_map.detach().cpu().numpy()


def make_cwl_drift_result(cubes: dict[str, CubeData], options: ReflectanceOptions) -> CwlDriftResult:
    if options.spatial_sigma == 0:
        try:
            return CwlDriftResult(make_cwl_drift_map_gpu(cubes, options), "GPU torch CUDA")
        except Exception:
            pass
    return CwlDriftResult(make_cwl_drift_map_cpu(cubes, options), "CPU")


def cwl_drift_limits(cwl_map: np.ndarray, wavelengths: tuple[float, ...]) -> tuple[float, float]:
    finite = cwl_map[np.isfinite(cwl_map)]
    if finite.size == 0:
        return wavelength_limits(wavelengths)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        return vmin - 0.5, vmax + 0.5
    return vmin, vmax


def smooth_cwl_map(cwl_map: np.ndarray, size: int = DEFAULT_CWL_MEDIAN_FILTER_SIZE) -> np.ndarray:
    if size <= 1:
        return cwl_map
    finite = np.isfinite(cwl_map)
    if not np.any(finite):
        return cwl_map
    fill_value = float(np.nanmedian(cwl_map[finite]))
    filled = np.where(finite, cwl_map, fill_value)
    smoothed = median_filter(filled, size=size, mode="nearest")
    smoothed[~finite] = np.nan
    return smoothed.astype(np.float32, copy=False)


def keep_dominant_cwl_window(
    cwl_map: np.ndarray,
    window_nm: float = DEFAULT_CWL_DOMINANT_WINDOW_NM,
) -> np.ndarray:
    finite = cwl_map[np.isfinite(cwl_map)]
    if finite.size == 0 or window_nm <= 0:
        return cwl_map
    values, counts = np.unique(finite, return_counts=True)
    dominant = float(values[int(np.argmax(counts))])
    filtered = cwl_map.copy()
    filtered[np.isfinite(filtered) & (np.abs(filtered - dominant) > window_nm)] = np.nan
    return filtered


def compute_roi_record_data(
    cubes: dict[str, CubeData],
    options: ReflectanceOptions,
    wavelengths: tuple[float, ...],
    roi_id: int,
    roi: Roi,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | str], str]:
    rows = compute_roi_stats(cubes, roi, options, roi_id=roi_id)
    display_means = np.array([row["mean_reflectance"] for row in rows], dtype=float)
    metric_means = np.array([row["metric_mean_reflectance"] for row in rows], dtype=float)
    metrics = compute_peak_metrics_with_fallback(wavelengths, metric_means, display_means)
    color = ROI_COLORS[(roi_id - 1) % len(ROI_COLORS)]
    return rows, metrics, color


def plot_roi_reflectance_curve(
    ax,
    wavelengths: tuple[float, ...],
    display_means: np.ndarray,
    metric_means: np.ndarray,
    color: str,
    label: str | None = None,
    linewidth: float = 1.4,
    markersize: float = 3.0,
) -> None:
    x = np.array(wavelengths, dtype=float)
    ax.plot(
        x,
        display_means,
        marker="o",
        markersize=markersize,
        linewidth=linewidth,
        label=label,
        color=color,
    )


class RoiReflectanceApp:
    def __init__(
        self,
        cubes: dict[str, CubeData],
        options: ReflectanceOptions,
        output_path: Path,
        data_dir: Path,
        initial_roi: Roi | None,
    ) -> None:
        self.cubes = cubes
        self.options = options
        self.output_path = output_path
        self.data_dir = data_dir
        self.roi_records: list[RoiRecord] = []
        self.executor = ThreadPoolExecutor(max_workers=DEFAULT_AUTO_WORKERS)
        self.auto_future: Future[list[tuple[int, Roi, list[dict[str, float | int | str]], dict[str, float | str], str]]] | None = None
        self.cwl_future: Future[CwlDriftResult] | None = None
        self.cwl_timer = None
        self.cwl_job_id = 0
        self.auto_timer = None
        self.auto_job_id = 0
        self.wavelengths = cubes["sign"].meta.wavelengths
        self.x_limits = wavelength_limits(self.wavelengths)
        meta = cubes["sign"].meta
        self.cwl_drift_map = np.full((meta.lines, meta.samples), np.nan, dtype=np.float32)
        self.preview_limits = cwl_drift_limits(self.cwl_drift_map, self.wavelengths)
        self.next_roi_id = 1

        self.fig, (self.ax_image, self.ax_curve) = plt.subplots(1, 2, figsize=(19.5, 9))
        self.fig.subplots_adjust(bottom=0.24, wspace=0.28)
        self.fig.patch.set_facecolor(THEME_FIG_BG)
        try:
            self.fig.canvas.manager.set_window_title("ROI Signal Evaluation")
        except AttributeError:
            pass

        vmin, vmax = self.preview_limits
        cwl_cmap = plt.get_cmap("turbo").copy()
        cwl_cmap.set_bad(color="#eef2f6")
        self.image_artist = self.ax_image.imshow(self.cwl_drift_map, cmap=cwl_cmap, vmin=vmin, vmax=vmax)
        style_plot_axis(self.ax_image)
        self.ax_image.set_title(self.image_title())
        self.ax_image.set_xlabel("x")
        self.ax_image.set_ylabel("y")
        colorbar = self.fig.colorbar(self.image_artist, ax=self.ax_image, fraction=0.046, pad=0.04)
        colorbar.set_label("Peak wavelength (nm)", color=THEME_MUTED_TEXT)
        colorbar.ax.tick_params(colors=THEME_MUTED_TEXT)
        colorbar.outline.set_edgecolor(THEME_BORDER)

        style_plot_axis(self.ax_curve)
        self.ax_curve.set_title("ROI mean Sign signal")
        self.ax_curve.set_xlabel("Wavelength (nm)")
        self.ax_curve.set_ylabel("Sign signal")
        self.ax_curve.set_xlim(*self.x_limits)
        self.ax_curve.grid(True, color=THEME_GRID, alpha=0.8, linewidth=0.8)
        self.status = self.fig.text(
            0.02,
            0.03,
            "Draw rectangles to add ROIs. Curves are computed directly from Sign/Rec.",
            fontsize=9,
            color=THEME_MUTED_TEXT,
        )
        self.result_box = self.fig.text(
            0.02,
            0.13,
            self.result_box_text(),
            fontsize=14,
            fontweight="bold",
            color=THEME_TEXT,
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": THEME_PANEL_BG,
                "edgecolor": THEME_BORDER,
                "alpha": 0.92,
            },
            linespacing=1.35,
        )

        button_left = 0.53
        button_width = 0.079
        button_gap = 0.015
        button_bottom = 0.07
        button_height = 0.05

        load_ax = self.fig.add_axes((button_left, button_bottom, button_width, button_height))
        self.load_button = Button(load_ax, "Load Data")
        style_button(self.load_button)
        self.load_button.on_clicked(self.on_load_data)

        auto_ax = self.fig.add_axes((button_left + (button_width + button_gap), button_bottom, button_width, button_height))
        self.auto_button = Button(auto_ax, "Auto 5x5")
        style_button(self.auto_button)
        self.auto_button.on_clicked(self.on_auto_grid)

        report_ax = self.fig.add_axes((button_left + 2 * (button_width + button_gap), button_bottom, button_width, button_height))
        self.report_button = Button(report_ax, "Report PNG")
        style_button(self.report_button)
        self.report_button.on_clicked(self.on_export_report)

        export_ax = self.fig.add_axes((button_left + 3 * (button_width + button_gap), button_bottom, button_width, button_height))
        self.export_button = Button(export_ax, "Export CSV")
        style_button(self.export_button)
        self.export_button.on_clicked(self.on_export)

        clear_ax = self.fig.add_axes((button_left + 4 * (button_width + button_gap), button_bottom, button_width, button_height))
        self.clear_button = Button(clear_ax, "Clear")
        style_button(self.clear_button)
        self.clear_button.on_clicked(self.on_clear)

        for button in (
            self.load_button,
            self.auto_button,
            self.report_button,
            self.export_button,
            self.clear_button,
        ):
            button.color = THEME_BUTTON
            button.hovercolor = THEME_BUTTON_HOVER

        self.selector = RectangleSelector(
            self.ax_image,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="pixels",
            interactive=True,
        )

        if initial_roi is not None:
            self.add_roi(initial_roi)
        self.start_cwl_drift_job("Loading CWL drift map")

    def image_title(self) -> str:
        return "CWL drift map - peak wavelength per pixel"

    def result_box_text(self) -> str:
        cwl_values = [
            float(record.metrics["cwl_nm"])
            for record in self.roi_records
            if np.isfinite(float(record.metrics["cwl_nm"]))
        ]
        fwhm_values = [
            float(record.metrics["fwhm_nm"])
            for record in self.roi_records
            if np.isfinite(float(record.metrics["fwhm_nm"]))
        ]

        if cwl_values:
            cwl_text = f"CWL: min={min(cwl_values):.2f} nm, max={max(cwl_values):.2f} nm"
        else:
            cwl_text = "CWL: min=-- nm, max=-- nm"

        if fwhm_values:
            fwhm_text = f"FWHM: min={min(fwhm_values):.2f} nm, max={max(fwhm_values):.2f} nm"
        else:
            fwhm_text = "FWHM: min=-- nm, max=-- nm"

        return f"{cwl_text}\n{fwhm_text}"

    def update_result_box(self) -> None:
        self.result_box.set_text(self.result_box_text())

    def start_cwl_drift_job(self, message: str) -> None:
        self.cwl_job_id += 1
        job_id = self.cwl_job_id
        self.status.set_text(f"{message}... GUI remains responsive.")
        self.fig.canvas.draw_idle()
        self.cwl_future = self.executor.submit(make_cwl_drift_result, self.cubes, self.options)
        self.schedule_cwl_drift_poll(job_id)

    def schedule_cwl_drift_poll(self, job_id: int) -> None:
        timer = self.fig.canvas.new_timer(interval=200)
        timer.single_shot = True
        timer.add_callback(lambda: self.poll_cwl_drift(job_id))
        self.cwl_timer = timer
        timer.start()

    def poll_cwl_drift(self, job_id: int) -> None:
        if job_id != self.cwl_job_id:
            return
        if self.cwl_future is None:
            return
        if not self.cwl_future.done():
            self.schedule_cwl_drift_poll(job_id)
            return
        self.finish_cwl_drift(job_id, self.cwl_future)

    def finish_cwl_drift(self, job_id: int, future: Future[CwlDriftResult]) -> None:
        if job_id != self.cwl_job_id:
            return
        try:
            result = future.result()
        except Exception as exc:
            self.status.set_text(f"CWL drift map failed: {exc}")
            self.fig.canvas.draw_idle()
            return
        self.cwl_drift_map = result.cwl_map
        self.preview_limits = cwl_drift_limits(self.cwl_drift_map, self.wavelengths)
        self.image_artist.set_data(self.cwl_drift_map)
        self.image_artist.set_clim(*self.preview_limits)
        self.ax_image.set_title(self.image_title())
        self.status.set_text(
            f"CWL drift map ready ({result.compute_backend}). "
            f"Range={self.preview_limits[0]:g}-{self.preview_limits[1]:g} nm."
        )
        self.fig.canvas.draw_idle()

    def on_load_data(self, _event) -> None:
        root = Tk()
        root.withdraw()
        selected = filedialog.askdirectory(
            title="Select folder containing *-Rec, *-Ref, *-Dark",
            initialdir=str(self.data_dir),
        )
        root.destroy()
        if not selected:
            return

        try:
            new_cubes = load_dataset(Path(selected))
        except Exception as exc:
            self.status.set_text(f"Failed to load data: {exc}")
            self.fig.canvas.draw_idle()
            return

        self.cubes = new_cubes
        self.data_dir = Path(selected)
        self.wavelengths = new_cubes["sign"].meta.wavelengths
        self.x_limits = wavelength_limits(self.wavelengths)
        meta = new_cubes["sign"].meta
        self.cwl_drift_map = np.full((meta.lines, meta.samples), np.nan, dtype=np.float32)
        self.preview_limits = cwl_drift_limits(self.cwl_drift_map, self.wavelengths)
        self.on_clear(None)

        self.image_artist.set_data(self.cwl_drift_map)
        self.image_artist.set_clim(*self.preview_limits)
        self.ax_image.set_title(self.image_title())
        self.ax_curve.set_xlim(*self.x_limits)
        self.status.set_text(
            f"Loaded data: {self.data_dir} | bands={len(self.wavelengths)}, "
            f"range={self.wavelengths[0]:g}-{self.wavelengths[-1]:g} nm"
        )
        self.fig.canvas.draw_idle()
        self.start_cwl_drift_job("Loading CWL drift map")

    def on_select(self, click_event, release_event) -> None:
        if click_event.xdata is None or click_event.ydata is None:
            return
        if release_event.xdata is None or release_event.ydata is None:
            return
        meta = self.cubes["sign"].meta
        roi = roi_from_selector(
            click_event.xdata,
            click_event.ydata,
            release_event.xdata,
            release_event.ydata,
            samples=meta.samples,
            lines=meta.lines,
        )
        self.add_roi(roi)

    def on_auto_grid(self, _event) -> None:
        meta = self.cubes["sign"].meta
        self.on_clear(None)
        rois = auto_grid_rois(
            samples=meta.samples,
            lines=meta.lines,
            rows=DEFAULT_GRID_ROWS,
            cols=DEFAULT_GRID_COLS,
            area_fraction=DEFAULT_GRID_ROI_FRACTION,
        )
        self.auto_job_id += 1
        job_id = self.auto_job_id
        start_roi_id = self.next_roi_id
        roi_jobs = [(start_roi_id + index, roi) for index, roi in enumerate(rois)]
        self.next_roi_id += len(rois)
        self.status.set_text(
            f"Computing {len(rois)} automatic ROIs with {DEFAULT_AUTO_WORKERS} workers..."
        )
        self.fig.canvas.draw_idle()
        self.auto_future = self.executor.submit(self.compute_auto_grid_records, roi_jobs)
        self.schedule_auto_grid_poll(job_id)

    def schedule_auto_grid_poll(self, job_id: int) -> None:
        timer = self.fig.canvas.new_timer(interval=200)
        timer.single_shot = True
        timer.add_callback(lambda: self.poll_auto_grid(job_id))
        self.auto_timer = timer
        timer.start()

    def poll_auto_grid(self, job_id: int) -> None:
        if job_id != self.auto_job_id:
            return
        if self.auto_future is None:
            return
        if not self.auto_future.done():
            self.schedule_auto_grid_poll(job_id)
            return
        self.finish_auto_grid(job_id, self.auto_future)

    def compute_auto_grid_records(
        self,
        roi_jobs: list[tuple[int, Roi]],
    ) -> list[tuple[int, Roi, list[dict[str, float | int | str]], dict[str, float | str], str]]:
        results = []
        with ThreadPoolExecutor(max_workers=DEFAULT_AUTO_WORKERS) as pool:
            futures = [
                pool.submit(compute_roi_record_data, self.cubes, self.options, self.wavelengths, roi_id, roi)
                for roi_id, roi in roi_jobs
            ]
            for (roi_id, roi), future in zip(roi_jobs, futures):
                rows, metrics, color = future.result()
                results.append((roi_id, roi, rows, metrics, color))
        return results

    def finish_auto_grid(
        self,
        job_id: int,
        future: Future[list[tuple[int, Roi, list[dict[str, float | int | str]], dict[str, float | str], str]]],
    ) -> None:
        if job_id != self.auto_job_id:
            return
        try:
            results = future.result()
        except Exception as exc:
            self.status.set_text(f"Auto 5x5 failed: {exc}")
            self.fig.canvas.draw_idle()
            return
        for roi_id, roi, rows, metrics, color in results:
            self.render_roi_record(roi_id, roi, rows, metrics, color)
        self.finalize_roi_plot()
        self.update_result_box()
        self.status.set_text(
            f"Added {len(results)} automatic ROIs: {DEFAULT_GRID_ROWS}x{DEFAULT_GRID_COLS}, "
            f"each ROI area is {DEFAULT_GRID_ROI_FRACTION:.0%} of its grid block."
        )
        self.fig.canvas.draw_idle()

    def add_roi(self, roi: Roi) -> None:
        roi_id = self.next_roi_id
        self.next_roi_id += 1
        rows, metrics, color = compute_roi_record_data(self.cubes, self.options, self.wavelengths, roi_id, roi)
        self.render_roi_record(roi_id, roi, rows, metrics, color)
        self.finalize_roi_plot()
        self.update_result_box()
        display_means = np.array([row["mean_reflectance"] for row in rows], dtype=float)
        fwhm = metrics["fwhm_nm"]
        cwl = metrics["cwl_nm"]
        low_snr_count = sum(int(row["low_snr_count"]) for row in rows)
        low_signal_snr_count = sum(int(row["low_signal_snr_count"]) for row in rows)
        prezero_count = sum(int(row["prezero_count"]) for row in rows)
        artifact_count = sum(1 for row in rows if row["artifact_status"] != "normal")
        low_quality_count = sum(1 for row in rows if row["quality_status"] != "ok")
        self.status.set_text(
            f"Added ROI {roi_id}: x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height}. "
            f"Mean Sign range: {finite_range_text(display_means, precision=4)}. "
            f"CWL={float(cwl):.2f} nm, FWHM={float(fwhm):.2f} nm ({metrics['metric_status']}). "
            f"Confidence={float(metrics['peak_confidence']):.3f}; "
            f"Secondary peak ratio={float(metrics['secondary_peak_ratio']):.3f}; "
            f"Prezero pixels: {prezero_count}; Low-SNR excluded: {low_snr_count} "
            f"(sign={low_signal_snr_count}); "
            f"low-quality bands: {low_quality_count}; marked bands: {artifact_count}."
        )
        self.fig.canvas.draw_idle()

    def render_roi_record(
        self,
        roi_id: int,
        roi: Roi,
        rows: list[dict[str, float | int | str]],
        metrics: dict[str, float | str],
        color: str,
    ) -> None:
        display_means = np.array([row["mean_reflectance"] for row in rows], dtype=float)
        patch = Rectangle(
            (roi.x, roi.y),
            roi.width,
            roi.height,
            fill=False,
            edgecolor=color,
            linewidth=1.8,
        )
        self.ax_image.add_patch(patch)
        label = self.ax_image.text(
            roi.x,
            max(0, roi.y - 4),
            f"ROI {roi_id}",
            color=color,
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1},
        )
        self.roi_records.append(
            RoiRecord(roi_id=roi_id, roi=roi, rows=rows, metrics=metrics, color=color, patch=patch, label=label)
        )
        self.selector.extents = (roi.x, roi.x1, roi.y, roi.y1)
        fwhm = metrics["fwhm_nm"]
        cwl = metrics["cwl_nm"]
        if np.isfinite(float(fwhm)) and np.isfinite(float(cwl)):
            curve_label = f"ROI {roi_id} CWL={float(cwl):.1f} FWHM={float(fwhm):.1f}"
        else:
            curve_label = f"ROI {roi_id}"
        metric_means = np.array([row["metric_mean_reflectance"] for row in rows], dtype=float)
        plot_roi_reflectance_curve(
            self.ax_curve,
            self.wavelengths,
            display_means,
            metric_means,
            color,
            label=curve_label,
        )

    def finalize_roi_plot(self) -> None:
        self.ax_curve.relim()
        self.ax_curve.autoscale_view()
        self.ax_curve.set_xlim(*self.x_limits)
        handles, labels = self.ax_curve.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.ax_curve.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
        self.ax_curve.set_title(f"ROI mean Sign signal ({len(self.roi_records)} ROI)")

    def on_export(self, _event) -> None:
        if not self.roi_records:
            self.status.set_text("No ROI selected. Draw at least one rectangle before exporting.")
            self.fig.canvas.draw_idle()
            return
        rows = rows_with_metrics(self.roi_records)
        write_stats_csv(rows, self.output_path)
        self.status.set_text(f"Saved {len(self.roi_records)} ROI Sign signal CSV: {self.output_path}")
        self.fig.canvas.draw_idle()

    def on_export_report(self, _event) -> None:
        if not self.roi_records:
            self.status.set_text("No ROI selected. Draw at least one rectangle before exporting report.")
            self.fig.canvas.draw_idle()
            return

        report_path = self.output_path.with_suffix(".png")
        fig, (ax_image, ax_curve, ax_quality) = plt.subplots(1, 3, figsize=(18, 5.5))
        cwl_cmap = plt.get_cmap("turbo").copy()
        cwl_cmap.set_bad(color="#eef2f6")
        image = ax_image.imshow(
            self.cwl_drift_map,
            cmap=cwl_cmap,
            vmin=self.preview_limits[0],
            vmax=self.preview_limits[1],
        )
        fig.colorbar(image, ax=ax_image, fraction=0.046, pad=0.04, label="Peak wavelength (nm)")
        ax_image.set_title("CWL drift map")
        ax_image.set_xlabel("x")
        ax_image.set_ylabel("y")

        for record in self.roi_records:
            roi = record.roi
            ax_image.add_patch(
                Rectangle((roi.x, roi.y), roi.width, roi.height, fill=False, edgecolor=record.color, linewidth=1.6)
            )
            ax_image.text(roi.x, max(0, roi.y - 4), f"ROI {record.roi_id}", color=record.color, fontsize=8)

            means = np.array([row["mean_reflectance"] for row in record.rows], dtype=float)
            metric_means = np.array([row["metric_mean_reflectance"] for row in record.rows], dtype=float)
            division_fraction = np.array([row["division_fraction"] for row in record.rows], dtype=float)
            plot_roi_reflectance_curve(
                ax_curve,
                self.wavelengths,
                means,
                metric_means,
                record.color,
                linewidth=1.2,
                markersize=3.0,
            )
            ax_quality.plot(self.wavelengths, division_fraction, marker=".", linewidth=1.0, color=record.color)

            metrics = record.metrics
            ax_curve.axvline(float(metrics["peak_wavelength_nm"]), color=record.color, linestyle=":", linewidth=1.0)

        ax_curve.set_title("ROI Sign signal")
        ax_curve.set_xlabel("Wavelength (nm)")
        ax_curve.set_ylabel("Sign signal")
        ax_curve.set_xlim(*self.x_limits)
        ax_curve.grid(True, alpha=0.3)
        ax_quality.set_title("Division pixel fraction")
        ax_quality.set_xlabel("Wavelength (nm)")
        ax_quality.set_ylabel("Fraction")
        ax_quality.set_xlim(*self.x_limits)
        ax_quality.set_ylim(-0.02, 1.02)
        ax_quality.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(report_path, dpi=160)
        plt.close(fig)
        self.status.set_text(f"Saved ROI report PNG: {report_path}")
        self.fig.canvas.draw_idle()

    def on_clear(self, _event) -> None:
        self.auto_job_id += 1
        for record in self.roi_records:
            if record.patch is not None:
                record.patch.remove()
            if record.label is not None:
                record.label.remove()
        for line in list(self.ax_curve.lines):
            line.remove()
        legend = self.ax_curve.get_legend()
        if legend is not None:
            legend.remove()
        self.roi_records.clear()
        self.next_roi_id = 1
        self.ax_curve.relim()
        self.ax_curve.autoscale_view()
        self.ax_curve.set_xlim(*self.x_limits)
        self.ax_curve.set_title("ROI mean Sign signal")
        self.update_result_box()
        self.status.set_text("Cleared ROIs. Draw rectangles on the image to add new ROIs.")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def load_dataset(data_dir: Path) -> dict[str, CubeData]:
    folders = find_cube_folders(data_dir)
    cubes = {
        "sign": load_cube("Sign/Rec", folders["sign"]),
        "ref": load_cube("Ref", folders["ref"]),
        "dark": load_cube("Dark", folders["dark"]),
    }
    assert_matching_cubes(cubes.values())
    return cubes


def data_dir_from_cubes(cubes: dict[str, CubeData]) -> Path:
    return cubes["sign"].meta.hdr_path.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ROI Sign/Rec signal from Sign/Rec, Ref, and Dark ENVI raw cubes."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing *-Rec, *-Ref, *-Dark")
    parser.add_argument("--output", type=Path, default=None, help="CSV path used by --no-gui or the Export CSV button")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Absolute minimum allowed Rec/Ref signal")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR, help="Minimum Rec/Ref signal in multiples of dark std")
    parser.add_argument(
        "--spatial-sigma",
        type=float,
        default=DEFAULT_SPATIAL_SIGMA,
        help="Spatial Gaussian sigma applied to Rec and Ref before SNR gating; 0 disables it",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_MIN_VALID_FRACTION,
        help="Quality flag threshold for ROI pixels that pass Signal and Ref SNR gates",
    )
    parser.add_argument(
        "--min-division-pixels",
        type=int,
        default=DEFAULT_MIN_DIVISION_PIXELS,
        help="Minimum number of ROI pixels that must pass Signal and Ref SNR gates",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=DEFAULT_MIN_COMPONENT_PIXELS,
        help="Minimum connected valid-pixel component size kept before division",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("mark", "none"),
        default=DEFAULT_ARTIFACT_MODE,
        help="Whether to visually/CSV mark the configured wavelength range without changing values",
    )
    parser.add_argument(
        "--artifact-range",
        type=float,
        nargs=2,
        default=(DEFAULT_ARTIFACT_MIN_NM, DEFAULT_ARTIFACT_MAX_NM),
        metavar=("MIN_NM", "MAX_NM"),
        help="Wavelength range treated as an artifact band",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        action="append",
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Initial or batch ROI in pixel coordinates; can be repeated",
    )
    parser.add_argument(
        "--auto-grid",
        action="store_true",
        help="Use the default 5x5 automatic ROI grid; each ROI covers 30 percent of its block",
    )
    parser.add_argument("--no-gui", action="store_true", help="Compute --roi and write CSV without opening the GUI")
    return parser.parse_args()


def main() -> None:
    configure_gui_fonts()
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    output_path = args.output or default_output_path()
    cubes = load_dataset(data_dir)
    meta = cubes["sign"].meta
    rois = []
    for roi_values in args.roi or []:
        rois.append(clamp_roi(Roi(*roi_values), samples=meta.samples, lines=meta.lines))
    if args.auto_grid:
        rois.extend(auto_grid_rois(samples=meta.samples, lines=meta.lines))

    artifact_min_nm = min(args.artifact_range)
    artifact_max_nm = max(args.artifact_range)
    options = ReflectanceOptions(
        eps=args.eps,
        snr=args.snr,
        spatial_sigma=max(0.0, args.spatial_sigma),
        min_valid_fraction=args.min_valid_fraction,
        min_division_pixels=max(1, args.min_division_pixels),
        min_component_pixels=max(1, args.min_component_pixels),
        artifact_min_nm=artifact_min_nm,
        artifact_max_nm=artifact_max_nm,
        artifact_mode=args.artifact_mode,
    )

    print(f"Loaded data: {data_dir}")
    print(f"Shape: bands={meta.bands}, lines={meta.lines}, samples={meta.samples}")
    print(f"Formula: Sign/Rec")
    print(
        f"Processing: Sign/Rec > max({options.eps:g}, {options.snr:g} * dark_std), "
        f"spatial sigma={options.spatial_sigma:g}, "
        f"quality division_fraction >= {options.min_valid_fraction:g} and "
        f"division pixels >= {options.min_division_pixels}, "
        f"component pixels >= {options.min_component_pixels}, "
        f"artifact {options.artifact_min_nm:g}-{options.artifact_max_nm:g} nm={options.artifact_mode}"
    )

    if args.no_gui:
        if not rois:
            raise SystemExit("--no-gui requires at least one --roi X Y WIDTH HEIGHT")
        records = []
        for index, roi in enumerate(rois, start=1):
            rows = compute_roi_stats(cubes, roi, options, roi_id=index)
            display_means = np.array([row["mean_reflectance"] for row in rows], dtype=float)
            metric_means = np.array([row["metric_mean_reflectance"] for row in rows], dtype=float)
            metrics = compute_peak_metrics_with_fallback(meta.wavelengths, metric_means, display_means)
            records.append(
                RoiRecord(
                    roi_id=index,
                    roi=roi,
                    rows=rows,
                    metrics=metrics,
                    color="",
                    patch=None,
                    label=None,
                )
            )
        rows = rows_with_metrics(records)
        write_stats_csv(rows, output_path)
        means = np.array([row["mean_reflectance"] for row in rows], dtype=float)
        for record in records:
            print(
                f"ROI {record.roi_id}: x={record.roi.x}, y={record.roi.y}, "
                f"width={record.roi.width}, height={record.roi.height}, "
                f"CWL={record.metrics['cwl_nm']}, FWHM={record.metrics['fwhm_nm']}"
            )
        print(f"Mean Sign signal range: {finite_range_text(means, precision=6)}")
        print(f"Saved CSV: {output_path}")
        return

    initial_roi = rois[0] if rois else None
    app = RoiReflectanceApp(
        cubes,
        options=options,
        output_path=output_path,
        data_dir=data_dir,
        initial_roi=initial_roi,
    )
    app.show()


if __name__ == "__main__":
    main()
