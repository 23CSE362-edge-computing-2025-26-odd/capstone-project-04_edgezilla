# Scenario 1: Low Occupancy – 20% Wards Filled

## Scenario Description
In this scenario, only 20% of hospital wards are occupied. The edge computing system monitors a small number of patients, resulting in minimal load on wearables, edge nodes, and cloud servers.

---

## Specifications

### Wearable Devices
- Model: Raspberry Pi Zero W with vitals sensors
- CPU: 1 Core, 1 GHz
- RAM: 512 MB
- Storage: 16 GB SD
- Tasks: Handles ~2 patient data streams concurrently

### Edge Node
- Model: Raspberry Pi 4
- CPU: 4 Cores, 1.5 GHz
- RAM: 4 GB
- Storage: 64 GB SSD
- Tasks: ~20 preprocessing tasks concurrently

### Cloud Node
- CPU: 16 Cores, 2.5 GHz
- RAM: 64 GB
- Storage: 1 TB SSD
- Tasks: ~200 inference tasks concurrently

---

## Expected Performance Metrics
| Metric                  | Value             |
|-------------------------|-----------------|
| Latency (ms)            | 100              |
| Throughput (tasks/sec)  | 15               |
| CPU Utilization (Edge)  | 30%              |
| CPU Utilization (Cloud) | 20%              |
| Energy Consumption (W)  | 4                |

---

**Observation**: Low occupancy allows the edge nodes to process data quickly with minimal latency and energy usage.
