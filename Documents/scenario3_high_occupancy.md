# Scenario 3: High Occupancy – 80% Wards Filled

## Scenario Description
80% of wards are occupied. High data volume from multiple wearables increases the workload on edge nodes, requiring some tasks to be offloaded to the cloud.

---

## Specifications

### Wearable Devices
- 8 Raspberry Pi Zero W units per ward
- Each handles ~2 patient streams

### Edge Nodes
- 5 Raspberry Pi 4 nodes
- CPU: 4 Cores, 1.5 GHz
- RAM: 4 GB
- Tasks: ~30 preprocessing tasks per node

### Cloud Node
- CPU: 16 Cores, 2.5 GHz
- RAM: 64 GB
- Tasks: ~200 ML inference tasks concurrently

---

## Expected Performance Metrics
| Metric                  | Value             |
|-------------------------|-----------------|
| Latency (ms)            | 180              |
| Throughput (tasks/sec)  | 100              |
| CPU Utilization (Edge)  | 70%              |
| CPU Utilization (Cloud) | 50%              |
| Energy Consumption (W)  | 20               |

---

**Observation**: High occupancy pushes edge nodes near full utilization, increasing latency and energy consumption.
