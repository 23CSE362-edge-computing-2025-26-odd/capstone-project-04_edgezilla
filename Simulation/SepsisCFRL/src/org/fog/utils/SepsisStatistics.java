package org.fog.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.cloudbus.cloudsim.core.CloudSim;

/**
 * Statistics collector for Sepsis Detection simulation.
 * Tracks loop latencies, tuple counts, and other metrics.
 */
public class SepsisStatistics {
    
    private static SepsisStatistics instance = null;
    
    // Loop latency tracking
    private List<Double> edgeLoopLatencies = new ArrayList<>();
    private List<Double> cloudLoopLatencies = new ArrayList<>();
    
    // Alert counts
    private int edgeAlertCount = 0;
    private int cloudAlertCount = 0;
    
    // Tuple processing counts
    private Map<String, Integer> tupleProcessingCounts = new HashMap<>();
    
    // Simulation timing
    private double simulationStartTime = 0;
    private double simulationEndTime = 0;
    
    // Energy consumption tracking
    private Map<String, Double> deviceEnergyConsumption = new HashMap<>();
    
    private SepsisStatistics() {
        // Private constructor for singleton
    }
    
    public static SepsisStatistics getInstance() {
        if (instance == null) {
            instance = new SepsisStatistics();
        }
        return instance;
    }
    
    public void reset() {
        edgeLoopLatencies.clear();
        cloudLoopLatencies.clear();
        edgeAlertCount = 0;
        cloudAlertCount = 0;
        tupleProcessingCounts.clear();
        deviceEnergyConsumption.clear();
        simulationStartTime = 0;
        simulationEndTime = 0;
    }
    
    public void recordSimulationStart() {
        simulationStartTime = CloudSim.clock();
    }
    
    public void recordSimulationEnd() {
        simulationEndTime = CloudSim.clock();
    }
    
    public void recordEdgeLoopLatency(double latency) {
        edgeLoopLatencies.add(latency);
    }
    
    public void recordCloudLoopLatency(double latency) {
        cloudLoopLatencies.add(latency);
    }
    
    public void incrementEdgeAlert() {
        edgeAlertCount++;
    }
    
    public void incrementCloudAlert() {
        cloudAlertCount++;
    }
    
    public void recordTupleProcessing(String tupleType) {
        tupleProcessingCounts.put(tupleType, tupleProcessingCounts.getOrDefault(tupleType, 0) + 1);
    }
    
    public void recordDeviceEnergyConsumption(String deviceName, double energy) {
        deviceEnergyConsumption.put(deviceName, energy);
    }
    
    // Getters for calculated metrics
    public double getSimulationDuration() {
        return simulationEndTime - simulationStartTime;
    }
    
    public double getAverageEdgeLoopLatency() {
        if (edgeLoopLatencies.isEmpty()) return 0.0;
        return edgeLoopLatencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public double getAverageCloudLoopLatency() {
        if (cloudLoopLatencies.isEmpty()) return 0.0;
        return cloudLoopLatencies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    public int getEdgeAlertCount() {
        return edgeAlertCount;
    }
    
    public int getCloudAlertCount() {
        return cloudAlertCount;
    }
    
    public int getTupleProcessingCount(String tupleType) {
        return tupleProcessingCounts.getOrDefault(tupleType, 0);
    }
    
    public double getTotalEnergyConsumption() {
        return deviceEnergyConsumption.values().stream().mapToDouble(Double::doubleValue).sum();
    }
    
    public Map<String, Double> getDeviceEnergyConsumption() {
        return new HashMap<>(deviceEnergyConsumption);
    }
    
    public int getTotalEdgeLoops() {
        return edgeLoopLatencies.size();
    }
    
    public int getTotalCloudLoops() {
        return cloudLoopLatencies.size();
    }
} 