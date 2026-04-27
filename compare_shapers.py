#!/usr/bin/env python3
"""
Compare Klipper's default shaper fitting against Shake&Tune's dynamic damping ratio fitting.

Klipper default: assumes damping_ratio=0.1, pessimizes scoring over fixed [0.075, 0.1, 0.15].
S&T modified:   estimates zeta from PSD (corrected half-power), pessimizes over [zeta*0.75, zeta, zeta*1.25].

Usage:
    python3 compare_shapers.py measurement.csv -k ~/klipper -o comparison.png [--scv 5.0]
"""

import argparse
import inspect
import os
import sys
from importlib import import_module
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


def build_shaper_table(kl_shapers, st_shapers, kl_choice_name, st_choice_name):
    kl_map = {s.name: s for s in kl_shapers}
    st_map = {s.name: s for s in st_shapers}
    names = [s.name for s in kl_shapers]

    col_labels = [
        'Shaper',
        'KL freq\n(Hz)', 'KL vibrs\n(%)', 'KL max_accel\n(mm/s²)',
        'S&T freq\n(Hz)', 'S&T vibrs\n(%)', 'S&T max_accel\n(mm/s²)',
        'Δ max_accel\n(mm/s²)',
    ]
    rows = []
    highlight = []  # (row_idx, reason)  reason: 'kl', 'st', 'both'
    for i, name in enumerate(names):
        kl = kl_map.get(name)
        st = st_map.get(name)
        if kl is None or st is None:
            continue
        delta = st.max_accel - kl.max_accel
        rows.append([
            name.upper(),
            f'{kl.freq:.1f}', f'{kl.vibrs * 100:.1f}', f'{kl.max_accel:.0f}',
            f'{st.freq:.1f}', f'{st.vibrs * 100:.1f}', f'{st.max_accel:.0f}',
            f'{delta:+.0f}',
        ])
        if name == kl_choice_name and name == st_choice_name:
            highlight.append((len(rows) - 1, 'both'))
        elif name == kl_choice_name:
            highlight.append((len(rows) - 1, 'kl'))
        elif name == st_choice_name:
            highlight.append((len(rows) - 1, 'st'))

    return col_labels, rows, highlight


def main():
    parser = argparse.ArgumentParser(
        description='Compare Klipper default vs Shake&Tune dynamic-DR shaper fitting'
    )
    parser.add_argument('csv', help='Raw Klipper accelerometer CSV file')
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

    # Load raw CSV — validate it's a raw accelerometer file, not a Klipper-processed PSD
    csv_path = Path(args.csv)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#freq,psd_x,psd_y,psd_z'):
                raise SystemExit(f'Error: {csv_path.name} is a processed PSD file. Use a raw accelerometer CSV.')
            if line.startswith('#time,accel_x,accel_y,accel_z'):
                break
        else:
            raise SystemExit(f'Error: {csv_path.name} does not look like a Klipper accelerometer CSV.')

    raw = np.loadtxt(csv_path, comments='#', delimiter=',')
    if raw.ndim == 1 or raw.shape[1] != 4:
        raise SystemExit(f'Error: expected 4 columns (time, x, y, z) in {csv_path.name}.')
    calib_data = process_data(sc, raw, csv_path.stem)
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

    # --- Run both fittings ---
    kl_choice, kl_shapers = run_fit(
        sc, calib_data,
        damping_ratio=KLIPPER_DEFAULT_DR,
        test_drs=KLIPPER_TEST_DRS,
        scv=args.scv,
        max_smoothing=args.max_smoothing,
        max_freq=args.max_freq,
    )
    st_choice, st_shapers = run_fit(
        sc, calib_data,
        damping_ratio=zeta,
        test_drs=zeta_bracket,
        scv=args.scv,
        max_smoothing=args.max_smoothing,
        max_freq=args.max_freq,
    )

    print()
    print(f'Klipper recommends : {kl_choice.name.upper()} @ {kl_choice.freq:.1f} Hz'
          f'  vibrs={kl_choice.vibrs * 100:.1f}%  max_accel={kl_choice.max_accel:.0f} mm/s²')
    print(f'S&T recommends     : {st_choice.name.upper()} @ {st_choice.freq:.1f} Hz'
          f'  vibrs={st_choice.vibrs * 100:.1f}%  max_accel={st_choice.max_accel:.0f} mm/s²')

    # --- Graph ---
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor('#f5f5f5')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.45)

    # Top: PSD + attenuated spectra for each recommendation
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('white')
    ax.plot(plot_freqs, plot_psd, color='#4c72b0', linewidth=1.2, label='Raw PSD', alpha=0.5)

    kl_map = {s.name: s for s in kl_shapers}
    st_map = {s.name: s for s in st_shapers}

    kl_shaper_obj = kl_map.get(kl_choice.name, kl_shapers[0])
    st_shaper_obj = st_map.get(st_choice.name, st_shapers[0])
    kl_vals = get_shaper_vals(kl_shaper_obj, plot_freqs)
    st_vals = get_shaper_vals(st_shaper_obj, plot_freqs)

    same_choice = kl_choice.name == st_choice.name

    ax.plot(
        plot_freqs, plot_psd * kl_vals,
        color='#dd4444', linewidth=1.8,
        label=f'Klipper ({kl_choice.name.upper()} @ {kl_choice.freq:.1f} Hz, DR=0.100, max_accel={kl_choice.max_accel:.0f})',
    )
    if same_choice:
        ax.plot(
            plot_freqs, plot_psd * st_vals,
            color='#2aa04a', linewidth=1.8, linestyle='--',
            label=f'S&T ({st_choice.name.upper()} @ {st_choice.freq:.1f} Hz, DR={zeta:.3f}, max_accel={st_choice.max_accel:.0f}) [same type]',
        )
    else:
        ax.plot(
            plot_freqs, plot_psd * st_vals,
            color='#2aa04a', linewidth=1.8,
            label=f'S&T ({st_choice.name.upper()} @ {st_choice.freq:.1f} Hz, DR={zeta:.3f}, max_accel={st_choice.max_accel:.0f})',
        )

    ax.axvline(fr, color='#888', linewidth=1.0, linestyle=':', alpha=0.8, label=f'fr = {fr:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_xlim(0, args.max_freq)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(
        f'Shaper Fitting Comparison — {csv_path.name}\n'
        f'fr = {fr:.1f} Hz  |  Klipper DR = 0.100  |  S&T estimated DR = {zeta:.4f}',
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    # Bottom: comparison table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    col_labels, rows, highlight_rows = build_shaper_table(
        kl_shapers, st_shapers, kl_choice.name, st_choice.name
    )

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    highlight_colors = {'kl': '#ffd6d6', 'st': '#d6f5d6', 'both': '#d6eaf8'}
    highlight_map = {r: reason for r, reason in highlight_rows}

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#dde4ed')
            cell.set_text_props(fontweight='bold', fontsize=8)
        else:
            data_row = row - 1
            reason = highlight_map.get(data_row)
            cell.set_facecolor(highlight_colors[reason] if reason else 'white')

            # Bold the Δ column if it's a highlighted row
            if col == len(col_labels) - 1 and reason:
                cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#cccccc')

    legend_text = (
        '  Shading:  red = Klipper choice   green = S&T choice   blue = both agree  '
        '  Δ max_accel = S&T minus Klipper'
    )
    ax_tbl.set_title(legend_text, fontsize=8, color='#555', pad=4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
