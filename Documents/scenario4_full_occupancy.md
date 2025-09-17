# Scenario 4: Emergency Surge (Sudden Patient Spike)

## Description
This scenario models an unexpected emergency where patient inflow spikes rapidly. New wearables are connected, drastically increasing data traffic in a short time. Edge servers are stressed beyond normal capacity.

## Resource Specifications

### Wearables
- CPU Cores: 1  
- MIPS: ~500  
- RAM: 256 MB  
- Bandwidth: 0.2–0.6 Mbps per device  
- Task Capacity: ~2 tasks/sec  

### Edge Layer
- CPU Cores: 4–6  
- MIPS: ~2500 per core  
- RAM: 4–6 GB  
- Bandwidth: 50–100 Mbps  
- Task Capacity: ~60 tasks/sec per core (burst mode)  

### Cloud Layer
- CPU Cores: 20+  
- MIPS: ~12000 per core  
- RAM: 128 GB  
- Bandwidth: 2 Gbps+  
- Task Capacity: Rapid scaling under surge  

## Baseline Comparison
- Edge queues overflow quickly.  
- Cloud absorbs excess load but increases latency.  
- Need for adaptive scheduling and dynamic offloading.  

## Evaluation Metrics
- Task failure rate under surge.  
- Average latency during peak.  
- Bandwidth spikes observed.  
- Cloud vs edge scaling efficiency.  

## Visualization Plan
- Line graph: Surge vs latency.  
- Bar chart: Failed vs completed tasks.  
- Heatmap: Bandwidth utilization over time.  
