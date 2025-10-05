# Scenario 2: High Occupancy (80% Wards Active)

## Description
This scenario simulates a hospital where 80% of wards are filled. Most patients have wearables generating health data. The workload is high but manageable at the edge with efficient task scheduling.

## Resource Specifications

### Wearables
- CPU Cores: 1  
- MIPS: ~500  
- RAM: 256 MB  
- Bandwidth: 0.1–0.4 Mbps per device  
- Task Capacity: ~2 tasks/sec  

### Edge Layer
- CPU Cores: 4  
- MIPS: ~2000 per core  
- RAM: 3 GB  
- Bandwidth: 10–80 Mbps (aggregated)  
- Task Capacity: ~40 tasks/sec per core  

### Cloud Layer
- CPU Cores: 12  
- MIPS: ~8000 per core  
- RAM: 48 GB  
- Bandwidth: 500 Mbps+  
- Task Capacity: Elastic  

## Baseline Comparison
- Edge still manages the majority of tasks.  
- Load balancing is necessary to prevent delays.  
- Cloud usage reduces compared to Scenario 1.  

## Evaluation Metrics
- Latency distribution across edge and cloud.  
- Task rejection/overflow ratio.  
- Throughput vs available cores.  

## Visualization Plan
- Stacked bar: Tasks handled by edge vs cloud.  
- Line chart: Latency comparison with Scenario 1.  
- Histogram: Bandwidth utilization.  
