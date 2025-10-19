#!/usr/bin/env python3
"""Run multiple hospital workloads and compare edge-computing metrics."""

import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
from run_simulation import run_simulation

# Default workload scenarios (vary wards and patient counts around the 15-patient average)
DEFAULT_WORKLOADS = [
    {"name": "low_load", "num_wards": 2, "patients_per_ward": 12},
    {"name": "baseline", "num_wards": 4, "patients_per_ward": 15},
    {"name": "peak_load", "num_wards": 6, "patients_per_ward": 18},
]

EDGE_METRICS = {
    "latency_ms": {
        "label": "Latency (ms)",
        "filename": "latency_comparison.png",
        "formatter": lambda v: f"{v:.1f} ms"
    },
    "throughput": {
        "label": "Throughput (tasks/sec)",
        "filename": "throughput_comparison.png",
        "formatter": lambda v: f"{v:.2f}"
    },
    "energy_per_task": {
        "label": "Energy Efficiency (J/task)",
        "filename": "energy_comparison.png",
        "formatter": lambda v: f"{v:.2f} J"
    },
    "resource_util_pct": {
        "label": "Resource Utilization (%)",
        "filename": "resource_comparison.png",
        "formatter": lambda v: f"{v:.1f}%"
    },
    "medical_accuracy_pct": {
        "label": "Medical Accuracy (%)",
        "title": "medical accuracy (offloading)",
        "filename": "medical_accuracy_offloading.png",
        "formatter": lambda v: f"{v:.1f}%"
    },
}


def estimate_energy(row):
    """Approximate per-task energy based on processing location."""
    latency_ms = row.get('total_latency_ms')
    if pd.isna(latency_ms):
        latency_ms = row.get('latency_ms', 0.0)
    latency_sec = max(0.0, latency_ms / 1000.0)
    data_size_mb = config.HEALTH_DATA_PACKET_SIZE_KB / 1024.0

    if row.get('location', '').lower() == 'edge':
        wearable_tx = data_size_mb * config.ENERGY_PER_MB_WIRELESS
        edge_proc_time = latency_sec - (config.LATENCY_WEARABLE_TO_EDGE / 1000.0)
        edge_proc_time = max(0.0, edge_proc_time)
        edge_energy = config.POWER_EDGE_SERVER_BUSY * edge_proc_time
        wearable_energy = config.POWER_WEARABLE_BUSY * latency_sec
        return wearable_tx + edge_energy + wearable_energy

    wearable_tx = data_size_mb * config.ENERGY_PER_MB_WIRELESS
    edge_cloud_tx = data_size_mb * config.ENERGY_PER_MB_WIRED
    edge_relay_time = config.LATENCY_WEARABLE_TO_EDGE / 1000.0
    edge_relay_energy = config.POWER_EDGE_SERVER_IDLE * edge_relay_time
    cloud_proc_time = latency_sec - edge_relay_time - (config.LATENCY_EDGE_TO_CLOUD / 1000.0)
    cloud_proc_time = max(0.0, cloud_proc_time)
    cloud_energy = config.POWER_CLOUD_INSTANCE_BUSY * cloud_proc_time
    wearable_energy = config.POWER_WEARABLE_BUSY * latency_sec
    return wearable_tx + edge_cloud_tx + edge_relay_energy + cloud_energy + wearable_energy


def prepare_output_dirs(base_dir, workload_name):
    workload_root = base_dir / workload_name
    if workload_root.exists():
        shutil.rmtree(workload_root)
    data_dir = workload_root / 'data'
    charts_dir = workload_root / 'charts'
    data_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, charts_dir


def summarize_run(data_dir, decisions_path, duration, workload_meta):
    performance_path = data_dir / 'performance.csv'
    if not performance_path.exists():
        raise FileNotFoundError(f"Missing performance data at {performance_path}")
    perf_df = pd.read_csv(performance_path)
    perf_df['energy_joules'] = perf_df.apply(estimate_energy, axis=1)

    decisions_df = pd.read_csv(decisions_path) if decisions_path.exists() else pd.DataFrame()
    dqn_decisions_path = data_dir / 'dqn_decisions.csv'
    dqn_decisions_df = pd.read_csv(dqn_decisions_path) if dqn_decisions_path.exists() else pd.DataFrame()

    latency_col = 'total_latency_ms' if 'total_latency_ms' in perf_df.columns else 'latency_ms'
    avg_latency = perf_df[latency_col].mean()
    total_tasks = len(perf_df)
    throughput = total_tasks / duration if duration > 0 else 0.0
    energy_per_task = perf_df['energy_joules'].mean()

    resource_util = np.nan
    if not decisions_df.empty and 'cpu_util' in decisions_df.columns:
        resource_util = decisions_df['cpu_util'].mean() * 100.0

    medical_accuracy = np.nan
    if not dqn_decisions_df.empty and 'correct' in dqn_decisions_df.columns:
        correct_series = dqn_decisions_df['correct']
        if correct_series.dtype == bool:
            medical_accuracy = correct_series.mean() * 100.0
        else:
            medical_accuracy = correct_series.astype(str).str.lower().isin(['true', '1', 'yes']).mean() * 100.0

    return {
        'workload': workload_meta['name'],
        'num_wards': workload_meta['num_wards'],
        'patients_per_ward': workload_meta['patients_per_ward'],
        'total_patients': workload_meta['num_wards'] * workload_meta['patients_per_ward'],
        'latency_ms': avg_latency,
        'throughput': throughput,
        'energy_per_task': energy_per_task,
        'resource_util_pct': resource_util,
        'medical_accuracy_pct': medical_accuracy,
        'tasks_processed': total_tasks,
    }


def plot_metric(summary_df, metric_key, output_dir):
    settings = EDGE_METRICS[metric_key]
    values = summary_df[metric_key].values
    labels = [
        f"{row.total_patients} pts\n{row.num_wards} wards"
        for row in summary_df.itertuples()
    ]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(values)), values, color='#4E79A7', alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                settings['formatter'](val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Workload (patients / wards)', fontweight='bold')
    ax.set_ylabel(settings['label'], fontweight='bold')
    plot_title = settings.get('title') or f"{settings['label']} Across Workloads"
    ax.set_title(plot_title, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    output_path = output_dir / settings['filename']
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_workloads(strategy, duration, workloads, output_root):
    base_dir = output_root
    base_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for workload in workloads:
        print(f"\n=== Running workload '{workload['name']}' ({workload['num_wards']} wards, "
              f"{workload['patients_per_ward']} patients/ward) ===")
        data_dir, charts_dir = prepare_output_dirs(base_dir, workload['name'])

        run_info = run_simulation(
            strategy=strategy,
            duration=duration,
            num_wards=workload['num_wards'],
            patients_per_ward=workload['patients_per_ward'],
            data_output_dir=data_dir,
            charts_output_dir=charts_dir
        )

        decisions_path = Path(run_info['data_dir']) / 'decisions.csv'
        summary = summarize_run(Path(run_info['data_dir']), decisions_path, run_info['duration'], workload)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values('total_patients').reset_index(drop=True)
    summary_csv = base_dir / 'workload_comparison_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved workload summary metrics to {summary_csv}")

    comparison_dir = base_dir / 'comparison_charts'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for metric in EDGE_METRICS:
        plot_metric(summary_df, metric, comparison_dir)

    print(f"Comparison charts written to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple workloads and compare edge metrics.')
    parser.add_argument('--strategy', type=str, default='dqn',
                        choices=['dqn', 'drl', 'always_edge', 'always_cloud', 'random'],
                        help='Simulation strategy to use for all workloads (default: dqn)')
    parser.add_argument('--duration', type=int, default=config.SIMULATION_DURATION,
                        help=f'Simulation duration in seconds (default: {config.SIMULATION_DURATION})')
    parser.add_argument('--output-root', type=str,
                        help='Optional directory to store workload run artifacts')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / config.RESULTS_DIR / 'workload_study'
    output_root = Path(args.output_root) if args.output_root else default_root

    run_workloads(args.strategy, args.duration, DEFAULT_WORKLOADS, output_root)


if __name__ == '__main__':
    main()
