package org.fog.application;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import org.fog.utils.SepsisStatistics;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

/**
 * DQN-based Offload Decision Module that communicates with Edge and Cloud DQN servers
 * to make intelligent offloading decisions instead of using static selectivity values.
 */
public class DQNOffloadDecisionModule {
    
    // DQN Server endpoints
    private static final String EDGE_DQN_SERVER = "http://localhost:5000";
    private static final String CLOUD_DQN_SERVER = "http://localhost:5001";
    
    // State tracking for each edge device
    private static Map<String, EdgeState> edgeStates = new ConcurrentHashMap<>();
    private static Map<String, Double> lastRewards = new ConcurrentHashMap<>();
    
    // Decision context tracking for each tuple
    private static Map<Integer, Map<String, Object>> tupleDecisionContexts = new ConcurrentHashMap<>();
    
    // Cloud aggregation interval (every 5 decisions)
    private static int cloudAggregationCounter = 0;
    private static final int CLOUD_AGGREGATION_INTERVAL = 5;
    
    /**
     * Inner class to track edge device state
     */
    public static class EdgeState {
        public double cpuUtilization;
        public double memoryUtilization;
        public double queueLength;
        public double latency;
        public double throughput;
        public long lastUpdate;
        
        public EdgeState() {
            this.cpuUtilization = 0.0;
            this.memoryUtilization = 0.0;
            this.queueLength = 0.0;
            this.latency = 0.0;
            this.throughput = 0.0;
            this.lastUpdate = System.currentTimeMillis();
        }
        
        public double[] toStateVector() {
            return new double[]{cpuUtilization, memoryUtilization, queueLength, latency, throughput, CloudSim.clock()};
        }
    }
    
    /**
     * Makes an offloading decision for a tuple using DQN
     * @param tuple The tuple to make decision for
     * @param fogDevice The fog device processing the tuple
     * @return 0 for edge processing, 1 for cloud offloading
     */
    public static int makeOffloadingDecision(Tuple tuple, FogDevice fogDevice) {
        String edgeId = fogDevice.getName();
        
        // Update edge state
        updateEdgeState(edgeId, fogDevice);
        
        // Get current state
        EdgeState state = edgeStates.get(edgeId);
        if (state == null) {
            state = new EdgeState();
            edgeStates.put(edgeId, state);
        }
        
        try {
            // Get Q-values from Edge DQN
            double[] qValues = getQValuesFromEdgeDQN(edgeId, state.toStateVector());
            
            // Choose action (0 = edge, 1 = cloud)
            int action = (qValues[0] > qValues[1]) ? 0 : 1;
            
            Log.printLine("DQN Decision for " + edgeId + ": action=" + action + 
                         " Q-values=[" + qValues[0] + ", " + qValues[1] + "]");
            
            // Update dynamic selectivities based on DQN decision
            updateSelectivitiesForDecision(action);
            
            // Store transition for learning (will be completed after execution)
            storeDecisionForLearning(edgeId, state.toStateVector(), action, tuple);
            
            // Track statistics based on decision
            SepsisStatistics.getInstance().recordTupleProcessing("T_PREPROCESSED");
            if (action == 0) {
                SepsisStatistics.getInstance().recordTupleProcessing("T_INFER_REQUEST");
            } else {
                SepsisStatistics.getInstance().recordTupleProcessing("T_OFFLOAD_TO_CLOUD");
            }
            
            // Periodically aggregate and communicate with cloud
            cloudAggregationCounter++;
            if (cloudAggregationCounter >= CLOUD_AGGREGATION_INTERVAL) {
                performCloudAggregation();
                cloudAggregationCounter = 0;
            }
            
            return action;
            
        } catch (Exception e) {
            Log.printLine("Error in DQN decision making: " + e.getMessage());
            // Fallback to simple rule-based decision
            int fallbackAction = (state.cpuUtilization > 0.8) ? 1 : 0;
            updateSelectivitiesForDecision(fallbackAction);
            return fallbackAction;
        }
    }
    
    /**
     * Updates the dynamic selectivities based on DQN decision
     */
    private static void updateSelectivitiesForDecision(int decision) {
        try {
            // Use reflection to call the DynamicDQNSelectivity update method
            Class<?> dynamicSelectivityClass = Class.forName("org.fog.application.DynamicDQNSelectivity");
            java.lang.reflect.Method updateMethod = dynamicSelectivityClass.getMethod("updateSelectivitiesFromDQN", int.class);
            updateMethod.invoke(null, decision);
        } catch (Exception e) {
            Log.printLine("Error updating dynamic selectivities: " + e.getMessage());
        }
    }
    
    /**
     * Updates the state of an edge device based on current metrics
     */
    private static void updateEdgeState(String edgeId, FogDevice fogDevice) {
        EdgeState state = edgeStates.computeIfAbsent(edgeId, k -> new EdgeState());
        
        // Calculate CPU utilization from allocated MIPS
        double totalMips = fogDevice.getHost().getTotalMips();
        double allocatedMips = fogDevice.getHost().getUtilizationOfCpu();
        state.cpuUtilization = allocatedMips / totalMips;
        
        // Calculate memory utilization
        int totalRam = fogDevice.getHost().getRam();
        int allocatedRam = fogDevice.getHost().getRamProvisioner().getUsedRam();
        state.memoryUtilization = (double) allocatedRam / totalRam;
        
        // Queue length estimation (number of tuples in queues)
        state.queueLength = fogDevice.getNorthTupleQueue().size() + fogDevice.getSouthTupleQueue().size();
        
        // Simple latency estimation (can be enhanced)
        state.latency = fogDevice.getUplinkLatency();
        
        // Throughput estimation (tuples per second)
        long currentTime = System.currentTimeMillis();
        if (currentTime - state.lastUpdate > 0) {
            state.throughput = 1000.0 / (currentTime - state.lastUpdate); // Simple approximation
        }
        
        state.lastUpdate = currentTime;
    }
    
    /**
     * Gets Q-values from the Edge DQN server
     */
    private static double[] getQValuesFromEdgeDQN(String edgeId, double[] state) throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/get_q_values");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        // Create JSON request
        JSONObject request = new JSONObject();
        request.put("edge_id", edgeId);
        request.put("state", state);
        
        // Send request
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        // Read response
        int responseCode = conn.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                
                JSONParser parser = new JSONParser();
                JSONObject jsonResponse = (JSONObject) parser.parse(response.toString());
                
                Object qValuesObj = jsonResponse.get("q_values");
                if (qValuesObj instanceof java.util.List) {
                    java.util.List<?> qValuesList = (java.util.List<?>) qValuesObj;
                    double[] qValues = new double[qValuesList.size()];
                    for (int i = 0; i < qValuesList.size(); i++) {
                        qValues[i] = ((Number) qValuesList.get(i)).doubleValue();
                    }
                    return qValues;
                }
            }
        }
        
        // Default Q-values if request fails
        return new double[]{0.5, 0.5};
    }
    
    /**
     * Stores the decision for later learning feedback
     */
    private static void storeDecisionForLearning(String edgeId, double[] state, int action, Tuple tuple) {
        // Store the decision context for when the tuple completes execution
        // This will be used to calculate reward and store transition
        Map<String, Object> decisionContext = new HashMap<>();
        decisionContext.put("state", state);
        decisionContext.put("action", action);
        decisionContext.put("timestamp", CloudSim.clock());
        decisionContext.put("edge_id", edgeId);
        
        // Store in static map using tuple's cloudlet ID
        tupleDecisionContexts.put(tuple.getCloudletId(), decisionContext);
    }
    
    /**
     * Provides feedback to the DQN after tuple execution completes
     */
    public static void provideFeedback(Tuple tuple, FogDevice fogDevice, double executionTime, boolean success) {
        Map<String, Object> decisionContext = tupleDecisionContexts.get(tuple.getCloudletId());
        if (decisionContext == null) return;
        
        String edgeId = (String) decisionContext.get("edge_id");
        double[] state = (double[]) decisionContext.get("state");
        int action = (Integer) decisionContext.get("action");
        double startTime = (Double) decisionContext.get("timestamp");
        
        // Calculate reward based on execution performance
        double reward = calculateReward(executionTime, success, action, fogDevice);
        
        // Get next state
        updateEdgeState(edgeId, fogDevice);
        EdgeState nextState = edgeStates.get(edgeId);
        
        try {
            // Send transition to Edge DQN for learning
            sendTransitionToEdgeDQN(edgeId, state, action, reward, nextState.toStateVector(), false, 
                                  executionTime, nextState.cpuUtilization, nextState.queueLength);
            
            lastRewards.put(edgeId, reward);
            
            // Clean up decision context
            tupleDecisionContexts.remove(tuple.getCloudletId());
            
        } catch (Exception e) {
            Log.printLine("Error providing feedback to DQN: " + e.getMessage());
        }
    }
    
    /**
     * Calculates reward based on performance metrics
     */
    private static double calculateReward(double executionTime, boolean success, int action, FogDevice fogDevice) {
        double reward = 0.0;
        
        // Base reward for success
        if (success) {
            reward += 10.0;
        } else {
            reward -= 5.0;
        }
        
        // Latency penalty (lower is better)
        reward -= executionTime / 10.0;
        
        // CPU utilization penalty (balanced utilization is better)
        double cpuUtil = fogDevice.getHost().getUtilizationOfCpu() / fogDevice.getHost().getTotalMips();
        reward -= Math.abs(cpuUtil - 0.7) * 10;
        
        // Action-specific rewards
        if (action == 0) { // Edge processing
            reward += (cpuUtil < 0.8) ? 5.0 : -2.0;
        } else { // Cloud offloading
            reward += (cpuUtil > 0.8) ? 3.0 : -1.0;
        }
        
        return reward;
    }
    
    /**
     * Sends transition data to Edge DQN server for learning
     */
    private static void sendTransitionToEdgeDQN(String edgeId, double[] state, int action, double reward, 
                                              double[] nextState, boolean done, double latency, 
                                              double cpuUtil, double queueLength) throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/store_transition");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        JSONObject request = new JSONObject();
        request.put("edge_id", edgeId);
        request.put("state", state);
        request.put("action", action);
        request.put("reward", reward);
        request.put("next_state", nextState);
        request.put("done", done);
        request.put("latency", latency);
        request.put("cpu_util", cpuUtil);
        request.put("queue_length", queueLength);
        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        conn.getResponseCode(); // Trigger request
    }
    
    /**
     * Performs cloud aggregation and resource allocation
     */
    private static void performCloudAggregation() {
        try {
            // Get Q-vectors from all edge agents
            Map<String, double[]> qVectors = getQVectorsFromEdges();
            
            // Calculate system metrics
            Map<String, Object> systemMetrics = calculateSystemMetrics();
            
            // Send to cloud DQN for aggregation and allocation
            JSONObject cloudResponse = sendToCloudDQN(qVectors, systemMetrics);
            
            // Distribute rewards back to edges
            if (cloudResponse.containsKey("distributed_rewards")) {
                Map<String, Double> distributedRewards = (Map<String, Double>) cloudResponse.get("distributed_rewards");
                for (Map.Entry<String, Double> entry : distributedRewards.entrySet()) {
                    sendGlobalRewardToEdge(entry.getKey(), entry.getValue());
                }
            }
            
            Log.printLine("Cloud aggregation completed. Distributed rewards to " + qVectors.size() + " edges.");
            
        } catch (Exception e) {
            Log.printLine("Error in cloud aggregation: " + e.getMessage());
        }
    }
    
    /**
     * Gets Q-vectors from all edge agents
     */
    private static Map<String, double[]> getQVectorsFromEdges() throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/get_q_vectors");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        JSONObject request = new JSONObject();
        request.put("edge_ids", edgeStates.keySet().toArray(new String[0]));
        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        Map<String, double[]> qVectors = new HashMap<>();
        int responseCode = conn.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                
                JSONParser parser = new JSONParser();
                JSONObject jsonResponse = (JSONObject) parser.parse(response.toString());
                Map<String, java.util.List<Double>> qVectorsMap = (Map<String, java.util.List<Double>>) jsonResponse.get("q_vectors");
                
                for (Map.Entry<String, java.util.List<Double>> entry : qVectorsMap.entrySet()) {
                    java.util.List<Double> qList = entry.getValue();
                    double[] qArray = qList.stream().mapToDouble(Double::doubleValue).toArray();
                    qVectors.put(entry.getKey(), qArray);
                }
            }
        }
        
        return qVectors;
    }
    
    /**
     * Calculates system-wide metrics for cloud DQN
     */
    private static Map<String, Object> calculateSystemMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        double avgCpuUtil = edgeStates.values().stream().mapToDouble(s -> s.cpuUtilization).average().orElse(0.0);
        double avgMemUtil = edgeStates.values().stream().mapToDouble(s -> s.memoryUtilization).average().orElse(0.0);
        double avgQueueLength = edgeStates.values().stream().mapToDouble(s -> s.queueLength).average().orElse(0.0);
        double avgThroughput = edgeStates.values().stream().mapToDouble(s -> s.throughput).average().orElse(0.0);
        
        metrics.put("cpu_util", avgCpuUtil);
        metrics.put("memory_util", avgMemUtil);
        metrics.put("queue_length", avgQueueLength);
        metrics.put("throughput", avgThroughput);
        metrics.put("fairness_index", calculateFairnessIndex());
        
        // Edge contributions (simplified)
        Map<String, Double> contributions = new HashMap<>();
        for (String edgeId : edgeStates.keySet()) {
            contributions.put(edgeId, lastRewards.getOrDefault(edgeId, 0.0));
        }
        metrics.put("edge_contributions", contributions);
        
        return metrics;
    }
    
    /**
     * Calculates fairness index across edges
     */
    private static double calculateFairnessIndex() {
        if (edgeStates.isEmpty()) return 1.0;
        
        double[] utilizations = edgeStates.values().stream().mapToDouble(s -> s.cpuUtilization).toArray();
        double sum = 0.0, sumSquares = 0.0;
        
        for (double util : utilizations) {
            sum += util;
            sumSquares += util * util;
        }
        
        double n = utilizations.length;
        return (sum * sum) / (n * sumSquares);
    }
    
    /**
     * Sends aggregated data to cloud DQN and gets allocation strategy
     */
    private static JSONObject sendToCloudDQN(Map<String, double[]> qVectors, Map<String, Object> systemMetrics) throws Exception {
        URL url = new URL(CLOUD_DQN_SERVER + "/aggregate_and_allocate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        JSONObject request = new JSONObject();
        request.put("edge_q_vectors", qVectors);
        request.put("system_metrics", systemMetrics);
        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
            StringBuilder response = new StringBuilder();
            String responseLine;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            
            JSONParser parser = new JSONParser();
            return (JSONObject) parser.parse(response.toString());
        }
    }
    
    /**
     * Sends global reward to an edge agent
     */
    private static void sendGlobalRewardToEdge(String edgeId, double globalReward) throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/set_global_reward");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        JSONObject request = new JSONObject();
        request.put("edge_id", edgeId);
        request.put("global_reward", globalReward);
        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        conn.getResponseCode(); // Trigger request
    }
} 