import math
from attrs import field
import numpy as np
import matplotlib
import numbers
import natsort
import pandas as pd
from collections import defaultdict
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors as rcolors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from glassesTools import eyetracker

import naming


def cart2pol(x: float, y: float) -> tuple[float,float]:
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho: float, phi: float) -> tuple[float,float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def get_eccentricity_ring(rho, phi, viewing_distance, rings):
    # Convert rho to degrees
    eccentricity = np.arctan(float(rho) / float(viewing_distance)) * 180 / np.pi
    if float(phi) < 0:
        rings = rings[1]
    else:
        rings = rings[0]
    ring = -1
    for kk, r in enumerate(rings):
        if round(eccentricity) in r:
            ring = kk
            break
    if ring==-1:
        RuntimeError(f'No ring found for target distance {eccentricity:.2f}, direction {np.degrees(phi):.2f}')

    return eccentricity, ring


def spans_from_bool(is_bright: np.ndarray, x0: float = 0.0, dx: float = 1.0):
    """
    Convert a boolean array into spans [(start, end, value), ...] where
    value is True for bright and False for dark. Each element spans [x0 + i*dx, x0 + (i+1)*dx).
    """
    is_bright = np.asarray(is_bright, dtype=bool)
    if is_bright.size == 0:
        return []

    # Find boundaries where value changes
    changes = np.flatnonzero(np.diff(is_bright))
    # Start indices for runs
    starts = np.r_[0, changes + 1]
    # End indices (exclusive)
    ends = np.r_[changes + 1, len(is_bright)]

    spans = []
    for s, e in zip(starts, ends):
        x_start = x0 + s * dx
        x_end   = x0 + e * dx
        spans.append((x_start, x_end, bool(is_bright[s])))
    return spans

def local_robust_range(signal, window_size=100, lower=5, upper=95):
    """
    Compute locally adaptive robust range using percentiles in a sliding window.

    Parameters:
        signal (array-like): Input signal.
        window_size (int): Size of the sliding window.
        lower (float): Lower percentile.
        upper (float): Upper percentile.

    Returns:
        ranges (np.ndarray): Array of local robust ranges.
    """
    signal = np.asarray(signal)
    n = len(signal)
    ranges = np.zeros(n)

    half_win = window_size // 2
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win)
        window = signal[start:end]
        low_val = np.percentile(window[~np.isnan(window)], lower)
        high_val = np.percentile(window[~np.isnan(window)], upper)
        ranges[i] = high_val - low_val

    return ranges


def get_et_info_from_recordings(data_dir, et_override_table):
    et_info = None
    for f in [f'{pr}eye_tracker_info.tsv' for pr in [naming.station1_prefix, naming.station2_prefix]]:
        this_info = pd.read_csv(data_dir / f, delimiter='\t')
        if et_info is None:
            et_info = this_info
        else:
            et_info = pd.concat([et_info, this_info], ignore_index=True)
    et_info['device'] = et_info['device'].apply(lambda d: eyetracker.EyeTracker(d))
    et_info = (et_info
        .groupby('device', sort=False)
        .agg(
            firmware_version=('firmware_version', unique_value_or_natsorted_list),
            recording_software_version=('recording_software_version', unique_value_or_natsorted_list)
        )
        .to_dict(orient='index')
    )
    # apply overrides
    for et in et_info:
        et_data = et_override_table.get(et, {})
        for key in et_data:
            et_info[et][key] = et_data[key]
    et_info = pd.DataFrame.from_dict(et_info, orient='index')
    # order columns
    cols = ['name', 'weight', 'sample_rate', 'firmware_version', 'recording_software_version']
    return et_info[cols+[c for c in et_info.columns if c not in cols]]

def unique_value_or_natsorted_list(series: pd.Series):
    # Drop NA-like entries, keep original types if you want; versions are typically strings anyway
    values = [v for v in series.dropna().unique().tolist() if pd.notna(v)]
    if not values:
        return None  # or return [] if you prefer an empty list
    if len(values) == 1:
        return values[0]
    # Natural sort (compares "2.9.8" < "2.10.0")
    return natsort.natsorted(values)


styles = getSampleStyleSheet()
APA_TableCaption = ParagraphStyle(
    "APA_TableCaption",
    parent=styles["Normal"],
    fontName="Times-Bold",
    fontSize=11,
    leading=14,
    alignment=TA_LEFT,
    spaceAfter=6
)
APA_FigureCaption = ParagraphStyle(
    "APA_FigureCaption",
    parent=styles["Normal"],
    fontName="Times-Roman",
    fontSize=10,
    leading=12,
    alignment=TA_LEFT,
    spaceBefore=6,
    spaceAfter=12
)
CenteredHeading1 = ParagraphStyle(
    "CenteredHeading1",
    parent=styles["Heading1"],
    alignment=TA_CENTER
)
LeftHeading = ParagraphStyle(
    "LeftHeading",
    parent=styles["Heading2"],
    alignment=TA_LEFT
)

class BookmarkedDocTemplate(SimpleDocTemplate):
    def __init__(self, filename, **kw):
        self.allowSplitting = 0
        super().__init__(filename, **kw)

    def afterFlowable(self, flowable):
        """Detects specific Paragraphs and adds bookmarks"""
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style == 'Heading3': # NB we abuse Heading3 for section headings
                text = flowable.getPlainText()
                key = f'h3_{self.page}_{text[:10]}'
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=0)

def make_apa_table(
    data,
    colNames=None,
    colWidths=None,
    font="Times-Roman",
    fontsize=10,
    header_font="Times-Bold",
    header_fontsize=10,
    decimals: int|defaultdict=2,
    colors=None,
    colors_limits=None,
    zebra=False,
    first_col_levels=None,
    missing_value_str="—",
    body_cell_padding=(6,4)      # h,v
):
    # Split header and body
    if isinstance(data, pd.DataFrame):
        data = [data.columns.tolist()] + data.values.tolist()
    elif isinstance(data, pd.Series):
        data = [data.index.tolist()] + [data.tolist()]
    header = data[0]
    body_raw = data[1:]
    header_lbls = header.copy()
    if colNames:
        header_lbls = [colNames.get(h, h) for h in header_lbls]

    # Detect numeric columns (a column is numeric if all non-empty values are numbers)
    numeric_cols = []
    n_cols = len(header)
    for c in range(n_cols):
        col_vals = [row[c] for row in body_raw if row[c] not in (None, "")]
        if col_vals and all(isinstance(v, numbers.Real) for v in col_vals):
            numeric_cols.append(c)

    # Decimal-aligned numeric cell
    def decimal_cell(val, n_decimals):
        s = f"{float(val):.{n_decimals}f}"
        if n_decimals<1:
            return s

        # split on dot
        int_part, frac_part = s.split(".", 1)
        sep_display = '.'
        sep_w = stringWidth(sep_display, font, fontsize) + 0.4

        inner = Table([[int_part, sep_display, frac_part]],
                        colWidths=[None, sep_w, None])
        inner.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), font),
            ("FONTSIZE", (0,0), (-1,-1), fontsize),
            ("ALIGN", (0,0), (0,0), "RIGHT"),   # integer part
            ("ALIGN", (1,0), (1,0), "CENTER"),  # separator
            ("ALIGN", (2,0), (2,0), "LEFT"),    # fraction
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING",   (0,0), (-1,-1), 0),
            ("BOTTOMPADDING",(0,0), (-1,-1), 0),
        ]))
        return inner

    # Remove duplicates in first column
    level_separators = []
    if first_col_levels is not None:
        seen = set()
        to_print = []
        pi = 0
        for r_idx, row in enumerate(body_raw, start=1):   # +1 to account for header
            key = row[0]
            if not key in seen:
                if r_idx>1:
                    level_separators.append(r_idx)
                seen.add(key)
                pi = 0
                to_print = first_col_levels.get(key, key)
                if not isinstance(to_print, list) and not isinstance(to_print, tuple):
                    to_print = [to_print]
            # replace with level info if available, else just blank out
            if pi<len(to_print):
                body_raw[r_idx-1][0] = to_print[pi]
                pi += 1
            else:
                body_raw[r_idx-1][0] = ""

    # Build display rows
    display_rows = []
    for row in body_raw:
        out = []
        for c_idx, cell in enumerate(row):
            if c_idx in numeric_cols and isinstance(cell, numbers.Real) and not math.isnan(cell):
                out.append(decimal_cell(cell, decimals[header[c_idx]] if isinstance(decimals, defaultdict) else decimals))
            elif isinstance(cell, list):
                out.append("\n".join(str(x) for x in cell))
            else:
                out.append(missing_value_str if cell is None or (isinstance(cell, float) and math.isnan(cell)) else str(cell))
        display_rows.append(out)

    table_data = [header_lbls] + display_rows

    # Create table
    tbl = Table(table_data, colWidths=colWidths, repeatRows=1)

    style_cmds = [
        # APA horizontal rules
        ('LINEABOVE',  (0,0), (-1,0), 1.0, rcolors.black),   # above header
        ('LINEBELOW',  (0,0), (-1,0), 0.6, rcolors.Color(0.6, 0.6, 0.6)),   # below header
        ('LINEBELOW',  (0,-1), (-1,-1), 1.0, rcolors.black), # bottom rule

        # Header formatting
        ('FONTNAME', (0,0), (-1,0), header_font),
        ('FONTSIZE', (0,0), (-1,0), header_fontsize),
        ('ALIGN',    (0,0), (-1,0), 'CENTER'),
        ('VALIGN',   (0, 0), (-1, 0), 'MIDDLE'),

        # Body formatting
        ('FONTNAME', (0,1), (-1,-1), font),
        ('FONTSIZE', (0,1), (-1,-1), fontsize),

        # First column left, others centered
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('VALIGN', (0, 1), (-1, -1), 'TOP'),

        # Cell padding
        ('LEFTPADDING',   (0,0), (-1,0), 6),
        ('RIGHTPADDING',  (0,0), (-1,0), 6),
        ('TOPPADDING',    (0,0), (-1,0), 4),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('LEFTPADDING',   (0,1), (-1,-1), body_cell_padding[0]),
        ('RIGHTPADDING',  (0,1), (-1,-1), body_cell_padding[0]),
        ('TOPPADDING',    (0,1), (-1,-1), body_cell_padding[1]),
        ('BOTTOMPADDING', (0,1), (-1,-1), body_cell_padding[1]),
    ] + [('LINEABOVE',  (0,r), (-1,r), 0.6, rcolors.Color(0.6, 0.6, 0.6)) for r in level_separators]

    # Zebra striping (optional)
    if zebra:
        for r in range(2, len(table_data), 2):
            style_cmds.append(
                ('BACKGROUND', (0, r), (-1, r), rcolors.whitesmoke)
            )

    # colorize cells based on thresholds (optional)
    if colors and colors_limits:
        for r_idx, row in enumerate(body_raw, start=1):   # +1 to account for header
            for c_idx, val in enumerate(row):
                h=header[c_idx]
                if h in colors and h in colors_limits and isinstance(val, numbers.Real):
                    color = get_color(abs(val), colors[h], *colors_limits[h])
                    style_cmds.append(
                        ('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), rcolors.Color(*color))
                    )

    tbl.setStyle(TableStyle(style_cmds))
    return tbl

def auto_image(path, max_width_cm=16, scale=1.):
    max_width = max_width_cm * cm
    img = Image(path)
    w, h = img.imageWidth, img.imageHeight
    scale_fac = max_width / float(w) * scale
    img.drawWidth = w * scale_fac
    img.drawHeight = h * scale_fac
    return img


def get_color(val, cmap, vmin, vmax):
    norm = matplotlib.colors.Normalize(vmin, vmax)
    return matplotlib.colormaps.get_cmap(cmap)(norm(val))

def custom_colormap(colors_255, cmap_name='CustomRdYlGn'):
    # Define RGB colors normalized to [0, 1]
    colors = [[comp/255 for comp in clr] for clr in colors_255]

    # Create a colormap from the defined colors
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name,
        colors
    )

def get_color_limits(data, colors_limits, fields, percentiles=None, need_abs=None):
    if percentiles is None:
        return colors_limits

    for field in fields:
        data_field = data[field].to_numpy()
        if need_abs and field in need_abs:
            data_field = np.abs(data_field)
        vmin = np.percentile(data_field[~np.isnan(data_field)], percentiles[0])
        vmax = np.percentile(data_field[~np.isnan(data_field)], percentiles[1])
        colors_limits[field] = [vmin, vmax]
    return colors_limits