# Scenario 2: Moderate Occupancy – 50% Wards Filled

## Scenario Description
50% of hospital wards are occupied. The system handles moderate data traffic from wearables, resulting in higher edge node utilization.

---

## Specifications

### Wearable Devices
- 5 Raspberry Pi Zero W units per ward
- Each handles ~2 patient streams

### Edge Nodes
- 3 Raspberry Pi 4 nodes
- CPU: 4 Cores, 1.5 GHz
- RAM: 4 GB
- Tasks: ~20 preprocessing tasks per node

### Cloud Node
- CPU: 16 Cores, 2.5 GHz
- RAM: 64 GB
- Tasks: ~200 ML inference tasks concurrently

---

## Expected Performance Metrics
| Metric                  | Value             |
|-------------------------|-----------------|
| Latency (ms)            | 130              |
| Throughput (tasks/sec)  | 45               |
| CPU Utilization (Edge)  | 55%              |
| CPU Utilization (Cloud) | 35%              |
| Energy Consumption (W)  | 12               |

---

**Observation**: Moderate occupancy increases latency slightly, but the edge cluster efficiently manages the workload.
