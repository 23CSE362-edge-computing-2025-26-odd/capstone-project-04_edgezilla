package org.fog.application;

import org.fog.application.selectivity.SelectivityModel;

/**
 * Dynamic selectivity model that allows runtime updates
 * to support DQN-based offloading decisions.
 */
public class DynamicDQNSelectivity implements SelectivityModel {
    
    private double selectivity;
    private static DynamicDQNSelectivity edgeInstance = null;
    private static DynamicDQNSelectivity cloudInstance = null;
    
    public DynamicDQNSelectivity(double initialSelectivity) {
        this.selectivity = initialSelectivity;
    }
    
    public void updateSelectivity(double newSelectivity) {
        this.selectivity = newSelectivity;
    }
    
    public double getSelectivity() {
        return selectivity;
    }
    
    @Override
    public boolean canSelect() {
        return Math.random() < selectivity;
    }
    
    @Override
    public double getMeanRate() {
        return selectivity;
    }
    
    @Override
    public double getMaxRate() {
        return 1.0;
    }
    
    // Static methods for singleton instances
    public static DynamicDQNSelectivity getEdgeInstance() {
        if (edgeInstance == null) {
            edgeInstance = new DynamicDQNSelectivity(0.7); // Default edge probability
        }
        return edgeInstance;
    }
    
    public static DynamicDQNSelectivity getCloudInstance() {
        if (cloudInstance == null) {
            cloudInstance = new DynamicDQNSelectivity(0.3); // Default cloud probability
        }
        return cloudInstance;
    }
    
    // Method to update selectivities based on DQN decision
    public static void updateSelectivitiesFromDQN(int decision) {
        if (decision == 0) {
            // Edge processing chosen
            getEdgeInstance().updateSelectivity(1.0);
            getCloudInstance().updateSelectivity(0.0);
        } else {
            // Cloud processing chosen
            getEdgeInstance().updateSelectivity(0.0);
            getCloudInstance().updateSelectivity(1.0);
        }
    }
} 