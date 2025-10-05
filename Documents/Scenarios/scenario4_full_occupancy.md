# Scenario 4: Full Occupancy – 100% Wards Filled

## Scenario Description
All wards are occupied. Edge nodes are fully loaded, and heavy reliance on cloud servers is required for real-time processing.

---

## Specifications

### Wearable Devices
- 10 Raspberry Pi Zero W units per ward
- Each handles ~2 patient streams

### Edge Nodes
- 6 Raspberry Pi 4 nodes
- CPU: 4 Cores, 1.5 GHz
- RAM: 4 GB
- Tasks: ~35 preprocessing tasks per node

### Cloud Node
- CPU: 16 Cores, 2.5 GHz
- RAM: 64 GB
- Tasks: ~200 ML inference tasks concurrently

---

## Expected Performance Metrics
| Metric                  | Value             |
|-------------------------|-----------------|
| Latency (ms)            | 220              |
| Throughput (tasks/sec)  | 150              |
| CPU Utilization (Edge)  | 85%              |
| CPU Utilization (Cloud) | 60%              |
| Energy Consumption (W)  | 28               |

---

**Observation**: Full occupancy scenario stresses both edge and cloud resources, highlighting the importance of efficient task scheduling and load balancing.
