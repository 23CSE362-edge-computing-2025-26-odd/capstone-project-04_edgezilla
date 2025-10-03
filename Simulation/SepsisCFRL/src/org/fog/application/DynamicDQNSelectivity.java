package org.fog.application;

import java.util.HashMap;
import java.util.Map;
import org.fog.application.selectivity.SelectivityModel;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.cloudbus.cloudsim.Log;

public class DynamicDQNSelectivity implements SelectivityModel {

    // A map to store the decision for a tuple, so we only ask the DQN once per tuple.
    private static Map<Integer, Integer> tupleDecisionMap = new HashMap<>();
    private static final int MAX_CACHE_SIZE = 1000; // Prevent unbounded growth

    private String destinationPath; // This will be either "EDGE" or "CLOUD"
    private Tuple currentTuple;
    private FogDevice currentDevice;

    public DynamicDQNSelectivity(String path) {
        this.destinationPath = path;
    }

    public void setCurrentTupleAndDevice(Tuple tuple, FogDevice device) {
        this.currentTuple = tuple;
        this.currentDevice = device;
        
        // Clean up cache if it grows too large
        if (tupleDecisionMap.size() > MAX_CACHE_SIZE) {
            tupleDecisionMap.clear();
            Log.printLine("DynamicDQNSelectivity: Cleared decision cache due to size limit");
        }
    }

    /**
     * This is the core method that iFogSim calls.
     * It triggers the DQN decision and returns true if the decision matches this instance's path.
     */
    @Override
    public boolean canSelect() {
        if (currentTuple == null || currentDevice == null) {
            Log.printLine("DynamicDQNSelectivity: Warning - null tuple or device");
            return false;
        }

        int decision;
        int tupleId = currentTuple.getActualTupleId();

        // Check if a decision for this tuple has already been made and cached
        if (tupleDecisionMap.containsKey(tupleId)) {
            decision = tupleDecisionMap.get(tupleId);
            Log.printLine("DynamicDQNSelectivity: Using cached decision " + decision + " for tuple " + tupleId);
        } else {
            // If not, make a new decision by calling your main DQN logic
            try {
                decision = DQNOffloadDecisionModule.makeOffloadingDecision(currentTuple, currentDevice);
                tupleDecisionMap.put(tupleId, decision);
                Log.printLine("DynamicDQNSelectivity: Made new decision " + decision + " for tuple " + tupleId);
            } catch (Exception e) {
                Log.printLine("DynamicDQNSelectivity: Error making decision - " + e.getMessage());
                return false;
            }
        }

        // If this instance is for the EDGE path, return true only if the decision was 0.
        if ("EDGE".equals(this.destinationPath)) {
            return decision == 0; // 0 means process on Edge
        }
        
        // If this instance is for the CLOUD path, return true only if the decision was 1.
        if ("CLOUD".equals(this.destinationPath)) {
            return decision == 1; // 1 means offload to Cloud
        }

        return false;
    }

    // These methods are required by the interface but are not used for our logic.
    @Override
    public double getMeanRate() {
        return 1.0;
    }
    
    @Override
    public double getMaxRate() {
        return 1.0;
    }
}