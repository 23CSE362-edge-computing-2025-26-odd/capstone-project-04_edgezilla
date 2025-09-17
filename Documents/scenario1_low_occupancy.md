# Scenario 3: Moderate Occupancy (50% Wards Active)

## Description
Here, only half of the hospital wards are active. The patient monitoring load is moderate, allowing the edge to comfortably process most of the data with minimal dependency on the cloud.

## Resource Specifications

### Wearables
- CPU Cores: 1  
- MIPS: ~500  
- RAM: 128–256 MB  
- Bandwidth: 0.1–0.3 Mbps per device  
- Task Capacity: ~1–2 tasks/sec  

### Edge Layer
- CPU Cores: 2–4  
- MIPS: ~2000 per core  
- RAM: 2 GB  
- Bandwidth: 10–50 Mbps (aggregated)  
- Task Capacity: ~30 tasks/sec per core  

### Cloud Layer
- CPU Cores: 8  
- MIPS: ~6000 per core  
- RAM: 32 GB  
- Bandwidth: 300 Mbps+  
- Task Capacity: Scales when required  

## Baseline Comparison
- Most tasks handled at edge.  
- Lower latency than Scenarios 1 & 2.  
- Cloud mainly used for backup and heavy analytics.  

## Evaluation Metrics
- Average latency improvements.  
- Resource utilization at edge.  
- Task completion with minimal offloading.  

## Visualization Plan
- Line graph: Latency improvement trend.  
- Pie chart: % tasks processed at edge vs cloud.  
- Bar chart: Bandwidth savings.  
