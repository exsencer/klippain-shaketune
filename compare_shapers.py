#!/usr/bin/env python3
"""
Compare Klipper's default shaper fitting against Shake&Tune's dynamic damping ratio fitting.

Klipper default: assumes damping_ratio=0.1, pessimizes scoring over fixed [0.075, 0.1, 0.15].
S&T modified:   estimates zeta from PSD (corrected half-power), pessimizes over [zeta*0.75, zeta, zeta*1.25].

Usage:
    python3 compare_shapers.py measurement.stdata -k ~/klipper -o comparison.png [--scv 5.0]
    python3 compare_shapers.py measurement.csv    -k ~/klipper -o comparison.png [--scv 5.0]
"""

import argparse
import inspect
import json
import os
import sys
from importlib import import_module
from io import TextIOWrapper
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


KLIPPER_DEFAULT_DR = 0.1
KLIPPER_TEST_DRS = [0.075, 0.1, 0.15]


def load_klipper(klipper_dir):
    kdir = os.path.expanduser(klipper_dir)
    sys.path.insert(0, os.path.join(kdir, 'klippy'))
    os.environ['SHAKETUNE_IN_CLI'] = '1'
    sc_mod = import_module('.shaper_calibrate', 'extras')
    sys.modules['shaper_calibrate'] = sc_mod
    sys.modules['shaper_defs'] = import_module('.shaper_defs', 'extras')
    return sc_mod


def normalize_fbs_result(result):
    """Handle all Klipper find_best_shaper return formats across versions."""
    if isinstance(result, list):
        return result[0], result[1:]
    elif isinstance(result, tuple) and len(result) == 2:
        return result
    elif hasattr(result, 'name'):
        return result, []
    return result, []


def process_data(sc, data, name):
    sig = inspect.signature(sc.process_accelerometer_data)
    if 'name' in sig.parameters:
        return sc.process_accelerometer_data(name, data)
    return sc.process_accelerometer_data(data)


def run_fit(sc, calib_data, damping_ratio, test_drs, scv, max_smoothing, max_freq):
    try:
        result = sc.find_best_shaper(
            calib_data,
            shapers=None,
            damping_ratio=damping_ratio,
            scv=scv,
            shaper_freqs=None,
            max_smoothing=max_smoothing,
            test_damping_ratios=test_drs,
            max_freq=max_freq,
            logger=None,
        )
    except TypeError:
        # Older Klipper without damping_ratio / test_damping_ratios support
        result = sc.find_best_shaper(calib_data, max_smoothing, None)
    return normalize_fbs_result(result)


def get_shaper_vals(shaper, freqs):
    """Return the shaper's vibration profile resampled onto freqs."""
    if hasattr(shaper, 'freq_bins') and shaper.freq_bins is not None:
        # New Klipper (Oct 2025+): shaper carries its own freq_bins
        return np.interp(freqs, shaper.freq_bins, shaper.vals)
    # Older Klipper: vals already matches the freq_bins-trimmed array
    return shaper.vals


def load_stdata(path):
    """Load the first measurement from a Shake&Tune .stdata file as a numpy (N,4) array."""
    from zstandard import ZstdDecompressor
    measurements = []
    with open(path, 'rb') as f:
        with ZstdDecompressor().stream_reader(f) as decomp:
            for line in TextIOWrapper(decomp, encoding='utf-8'):
                if line.strip():
                    measurements.append(json.loads(line))
    if not measurements:
        raise SystemExit(f'Error: {path.name} contains no measurements.')
    return measurements


def load_input_file(path):
    """Return (name, raw_array) from either a .stdata or a raw accelerometer .csv."""
    if path.suffix == '.stdata':
        measurements = load_stdata(path)
        if len(measurements) > 1:
            print(f'Note: {path.name} contains {len(measurements)} measurements; using the first one.')
        m = measurements[0]
        return m['name'], np.array(m['samples'])

    # CSV path — validate header first
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#freq,psd_x,psd_y,psd_z'):
                raise SystemExit(f'Error: {path.name} is a processed PSD file. Use a raw accelerometer CSV.')
            if line.startswith('#time,accel_x,accel_y,accel_z'):
                break
        else:
            raise SystemExit(f'Error: {path.name} does not look like a Klipper accelerometer CSV.')
    raw = np.loadtxt(path, comments='#', delimiter=',')
    if raw.ndim == 1 or raw.shape[1] != 4:
        raise SystemExit(f'Error: expected 4 columns (time, x, y, z) in {path.name}.')
    return path.stem, raw


def build_shaper_table(kl_shapers, single_shapers, bracket_shapers, kl_choice_name, single_choice_name, bracket_choice_name):
    kl_map = {s.name: s for s in kl_shapers}
    single_map = {s.name: s for s in single_shapers}
    bracket_map = {s.name: s for s in bracket_shapers}
    names = [s.name for s in kl_shapers]

    col_labels = [
        'Shaper',
        'KL freq\n(Hz)', 'KL vibrs\n(%)', 'KL accel\n(mm/s²)',
        'Single ζ freq\n(Hz)', 'Single ζ vibrs\n(%)', 'Single ζ accel\n(mm/s²)',
        'Bracket ζ freq\n(Hz)', 'Bracket ζ vibrs\n(%)', 'Bracket ζ accel\n(mm/s²)',
    ]
    rows = []
    # choices: set of (row_idx, method) where method is 'kl', 'single', 'bracket'
    choices = []
    for name in names:
        kl = kl_map.get(name)
        sg = single_map.get(name)
        br = bracket_map.get(name)
        if kl is None or sg is None or br is None:
            continue
        rows.append([
            name.upper(),
            f'{kl.freq:.1f}', f'{kl.vibrs * 100:.1f}', f'{kl.max_accel:.0f}',
            f'{sg.freq:.1f}', f'{sg.vibrs * 100:.1f}', f'{sg.max_accel:.0f}',
            f'{br.freq:.1f}', f'{br.vibrs * 100:.1f}', f'{br.max_accel:.0f}',
        ])
        row_idx = len(rows) - 1
        if name == kl_choice_name:
            choices.append((row_idx, 'kl'))
        if name == single_choice_name:
            choices.append((row_idx, 'single'))
        if name == bracket_choice_name:
            choices.append((row_idx, 'bracket'))

    return col_labels, rows, choices


def main():
    parser = argparse.ArgumentParser(
        description='Compare Klipper default vs Shake&Tune dynamic-DR shaper fitting'
    )
    parser.add_argument('file', help='Shake&Tune .stdata file or raw Klipper accelerometer .csv')
    parser.add_argument('-k', '--klipper-dir', default='~/klipper', help='Klipper installation directory')
    parser.add_argument('-o', '--output', default='shaper_comparison.png', help='Output graph path')
    parser.add_argument('--scv', type=float, default=5.0, help='Square corner velocity (mm/s)')
    parser.add_argument('--max-smoothing', type=float, default=None, help='Maximum allowed smoothing')
    parser.add_argument('--max-freq', type=float, default=200.0, help='Max frequency to analyse (Hz)')
    args = parser.parse_args()

    # Locate this script's repo so we can import S&T helpers
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from shaketune.helpers.common_func import compute_mechanical_parameters

    # Load Klipper shaper_calibrate
    sc_mod = load_klipper(args.klipper_dir)
    sc = sc_mod.ShaperCalibrate(printer=None)

    # Load input file (.stdata or .csv)
    file_path = Path(args.file)
    meas_name, raw = load_input_file(file_path)
    calib_data = process_data(sc, raw, meas_name)
    calib_data.normalize_to_frequencies()

    # Estimate damping ratio from the full (untrimmed) PSD — matches S&T's shaper_computation.py
    fr, zeta, _, _ = compute_mechanical_parameters(calib_data.psd_sum, calib_data.freq_bins)
    zeta = zeta if zeta is not None else KLIPPER_DEFAULT_DR
    zeta_bracket = [zeta * 0.75, zeta, zeta * 1.25]

    # Local trimmed arrays used only for plotting
    mask = calib_data.freq_bins <= args.max_freq
    plot_freqs = calib_data.freq_bins[mask]
    plot_psd = calib_data.psd_sum[mask]

    print(f'Resonant frequency : {fr:.2f} Hz')
    print(f'Klipper assumed DR : {KLIPPER_DEFAULT_DR}  (bracket: {KLIPPER_TEST_DRS})')
    print(f'S&T estimated DR   : {zeta:.4f}  (bracket: [{zeta_bracket[0]:.4f}, {zeta_bracket[1]:.4f}, {zeta_bracket[2]:.4f}])')

    # --- Run all three fittings ---
    kl_choice, kl_shapers = run_fit(
        sc, calib_data,
        damping_ratio=KLIPPER_DEFAULT_DR,
        test_drs=KLIPPER_TEST_DRS,
        scv=args.scv,
        max_smoothing=args.max_smoothing,
        max_freq=args.max_freq,
    )
    single_choice, single_shapers = run_fit(
        sc, calib_data,
        damping_ratio=zeta,
        test_drs=[zeta],
        scv=args.scv,
        max_smoothing=args.max_smoothing,
        max_freq=args.max_freq,
    )
    bracket_choice, bracket_shapers = run_fit(
        sc, calib_data,
        damping_ratio=zeta,
        test_drs=zeta_bracket,
        scv=args.scv,
        max_smoothing=args.max_smoothing,
        max_freq=args.max_freq,
    )

    print()
    print(f'Klipper  (DR=0.100, bracket {KLIPPER_TEST_DRS})  : {kl_choice.name.upper()} @ {kl_choice.freq:.1f} Hz'
          f'  vibrs={kl_choice.vibrs * 100:.1f}%  max_accel={kl_choice.max_accel:.0f} mm/s²')
    print(f'S&T single ζ={zeta:.4f}                          : {single_choice.name.upper()} @ {single_choice.freq:.1f} Hz'
          f'  vibrs={single_choice.vibrs * 100:.1f}%  max_accel={single_choice.max_accel:.0f} mm/s²')
    print(f'S&T bracket [{zeta_bracket[0]:.4f}, {zeta_bracket[1]:.4f}, {zeta_bracket[2]:.4f}]  : {bracket_choice.name.upper()} @ {bracket_choice.freq:.1f} Hz'
          f'  vibrs={bracket_choice.vibrs * 100:.1f}%  max_accel={bracket_choice.max_accel:.0f} mm/s²')

    # --- Graph ---
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#f5f5f5')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.45)

    # Top: PSD + attenuated spectra for all three recommendations
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('white')
    ax.plot(plot_freqs, plot_psd, color='#4c72b0', linewidth=1.2, label='Raw PSD', alpha=0.5)

    kl_map = {s.name: s for s in kl_shapers}
    single_map = {s.name: s for s in single_shapers}
    bracket_map = {s.name: s for s in bracket_shapers}

    curves = [
        (kl_map.get(kl_choice.name, kl_shapers[0]),
         '#dd4444',
         f'Klipper  {kl_choice.name.upper()} @ {kl_choice.freq:.1f} Hz  DR=0.100  accel={kl_choice.max_accel:.0f}'),
        (single_map.get(single_choice.name, single_shapers[0]),
         '#e07b00',
         f'Single ζ  {single_choice.name.upper()} @ {single_choice.freq:.1f} Hz  ζ={zeta:.3f}  accel={single_choice.max_accel:.0f}'),
        (bracket_map.get(bracket_choice.name, bracket_shapers[0]),
         '#2aa04a',
         f'Bracket ζ  {bracket_choice.name.upper()} @ {bracket_choice.freq:.1f} Hz  ζ={zeta:.3f}±25%  accel={bracket_choice.max_accel:.0f}'),
    ]
    for shaper_obj, color, label in curves:
        vals = get_shaper_vals(shaper_obj, plot_freqs)
        ax.plot(plot_freqs, plot_psd * vals, color=color, linewidth=1.8, label=label)

    ax.axvline(fr, color='#888', linewidth=1.0, linestyle=':', alpha=0.8, label=f'fr = {fr:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_xlim(0, args.max_freq)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(
        f'Shaper Fitting Comparison — {file_path.name}\n'
        f'fr = {fr:.1f} Hz  |  Klipper DR = 0.100  |  S&T estimated DR = {zeta:.4f}',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    # Bottom: comparison table (all three methods)
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    col_labels, rows, choices = build_shaper_table(
        kl_shapers, single_shapers, bracket_shapers,
        kl_choice.name, single_choice.name, bracket_choice.name,
    )

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    # Column ranges for each method (0-indexed after the Shaper name col)
    METHOD_COLS = {'kl': range(1, 4), 'single': range(4, 7), 'bracket': range(7, 10)}
    METHOD_COLORS = {'kl': '#ffd6d6', 'single': '#fff0d6', 'bracket': '#d6f5d6'}

    # Build a per-row set of highlighted methods
    row_methods = {}
    for row_idx, method in choices:
        row_methods.setdefault(row_idx, set()).add(method)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#dde4ed')
            cell.set_text_props(fontweight='bold', fontsize=7)
        else:
            data_row = row - 1
            methods = row_methods.get(data_row, set())
            # Find which method owns this column and highlight if it's a chosen method
            cell_color = 'white'
            for method, col_range in METHOD_COLS.items():
                if col in col_range and method in methods:
                    cell_color = METHOD_COLORS[method]
                    break
            cell.set_facecolor(cell_color)
        cell.set_edgecolor('#cccccc')

    legend_text = (
        '  Column shading:  red = Klipper choice   orange = Single ζ choice   green = Bracket ζ choice'
    )
    ax_tbl.set_title(legend_text, fontsize=8, color='#555', pad=4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
