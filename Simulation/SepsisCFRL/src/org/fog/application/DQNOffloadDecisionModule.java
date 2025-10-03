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
    private static int simulationStep = 0; 
    private static int lastStepLogged = -1;
    
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
            // Get Q-values AND epsilon from Edge DQN
            JSONObject response = getQValuesFromEdgeDQN(edgeId, state.toStateVector()); // Changed to get JSONObject
    
            // Parse Q-values from List to double array
            Object qValuesObj = response.get("q_values");
            double[] qValues;
            if (qValuesObj instanceof java.util.List) {
                java.util.List<?> qValuesList = (java.util.List<?>) qValuesObj;
                qValues = new double[qValuesList.size()];
                for (int i = 0; i < qValuesList.size(); i++) {
                    qValues[i] = ((Number) qValuesList.get(i)).doubleValue();
                }
            } else {
                Log.printLine("DQN: Unexpected q_values format, using defaults");
                qValues = new double[]{0.5, 0.5};
            }
            
            double epsilon = ((Number) response.get("epsilon")).doubleValue(); // Fix epsilon parsing
    
            int action;
            String decisionType;
    
            // This is the logic to decide if we are exploring or exploiting
            if (Math.random() < epsilon) {
                action = (new java.util.Random()).nextInt(2);
                decisionType = "Explore";
            } else {
                action = (qValues[0] > qValues[1]) ? 0 : 1;
                decisionType = "Exploit";
            }
    
            // Store epsilon and decisionType for the feedback step
            storeDecisionForLearning(edgeId, state.toStateVector(), action, tuple, epsilon, decisionType, qValues);
            
            Log.printLine("DQN: Made decision " + action + " (" + decisionType + ") with epsilon=" + epsilon + 
                         " and Q-values=[" + qValues[0] + ", " + qValues[1] + "]");
            
            // Track statistics based on decision
            SepsisStatistics.getInstance().recordTupleProcessing("T_PREPROCESSED");
            if (action == 0) {
                SepsisStatistics.getInstance().recordTupleProcessing("T_INFER_REQUEST");
                // Record this as an edge decision for statistics
                SepsisStatistics.getInstance().recordEdgeLoopLatency(15.0); // Approximate edge latency
            } else {
                SepsisStatistics.getInstance().recordTupleProcessing("T_OFFLOAD_TO_CLOUD");
                // Record this as a cloud decision for statistics  
                SepsisStatistics.getInstance().recordCloudLoopLatency(75.0); // Approximate cloud latency
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
    private static JSONObject getQValuesFromEdgeDQN(String edgeId, double[] state) throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/get_q_values");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        conn.setConnectTimeout(5000);  // 5 second timeout
        conn.setReadTimeout(5000);     // 5 second timeout
        
        // Create JSON request
        JSONObject request = new JSONObject();
        request.put("edge_id", edgeId);

        java.util.List<Double> stateList = new java.util.ArrayList<>();
        for (double d : state) {
            stateList.add(d);
        }
        request.put("state", stateList);
        
        Log.printLine("DQN: Sending request to Edge DQN server for edge " + edgeId);
        
        // Send request
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
        
        // Read response
        int responseCode = conn.getResponseCode();
        Log.printLine("DQN: Got response code " + responseCode + " from Edge DQN server");
        
        if (responseCode == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                
                JSONParser parser = new JSONParser();
                JSONObject jsonResponse = (JSONObject) parser.parse(response.toString());
                Log.printLine("DQN: Received response from Edge DQN server: " + jsonResponse.toString());
                return jsonResponse;
            }
        }
        
        // Default response if request fails
        Log.printLine("DQN: Using default response due to request failure");
        JSONObject defaultResponse = new JSONObject();
        defaultResponse.put("q_values", new double[]{0.5, 0.5});
        defaultResponse.put("epsilon", 0.1);
        return defaultResponse;
    }
    
    /**
     * Stores the decision for later learning feedback
     */
    private static void storeDecisionForLearning(String edgeId, double[] state, int action, Tuple tuple, double epsilon, String decisionType, double[] qValues) {
        Map<String, Object> decisionContext = new HashMap<>();
        decisionContext.put("state", state);
        decisionContext.put("action", action);
        decisionContext.put("timestamp", CloudSim.clock());
        decisionContext.put("edge_id", edgeId);
        decisionContext.put("epsilon", epsilon);           // Store epsilon
        decisionContext.put("decision_type", decisionType); // Store decision type
        decisionContext.put("q_values", qValues); // Store Q-values
    
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

        String decisionType = (String) decisionContext.get("decision_type");
        
        // Calculate reward based on execution performance
        double reward = calculateReward(executionTime, success, action, fogDevice);
        
        // Get next state
        updateEdgeState(edgeId, fogDevice);
        EdgeState nextState = edgeStates.get(edgeId);
        
        try {
            // Send transition and get the loss back from the server
            JSONObject response = sendTransitionToEdgeDQN(edgeId, state, action, reward, nextState.toStateVector(), false, executionTime, nextState.cpuUtilization, nextState.queueLength);
            double loss = 0.0;
            if (response != null && response.get("loss") != null) {
                loss = ((Number) response.get("loss")).doubleValue();
            }

            // *** THE BIG CHANGE: Call the new logging function ***
            logDQNStep(edgeId, state, (double[]) decisionContext.get("q_values"), action, decisionType, executionTime, nextState.cpuUtilization, (int)nextState.queueLength, reward, loss);
                
            lastRewards.put(edgeId, reward);
            
            // Clean up decision context
            tupleDecisionContexts.remove(tuple.getCloudletId());
            
        } catch (Exception e) {
            Log.printLine("Error providing feedback to DQN: " + e.getMessage());
        }
    }
    
    // In DQNOffloadDecisionModule.java - A completely new method

    private static void logDQNStep(String edgeId, double[] state, double[] qValues, int action, String decisionType, double latency, double cpuUtil, int queue, double reward, double loss) {

        // Print the Simulation Step header only when the step number changes
        int currentStep = (int) CloudSim.clock();
        if (currentStep > lastStepLogged) {
        simulationStep = currentStep;
        lastStepLogged = currentStep;
        System.out.println("\n================================================================================");
        System.out.printf("====== %30s %d %-30s ======\n", "SIMULATION STEP", simulationStep, "");
        System.out.println("================================================================================");
        }

        // Mock Patient State for logging purposes (based on the state vector)
        String patientState = String.format("CPU=%.1f%%, Mem=%.1f%%, Lat=%.2f, Throughput=%.2f",
                state[0] * 100, state[1] * 100, state[3], state[4]);

        String decision = (action == 0) ? "PROCESS ON EDGE" : "OFFLOAD TO CLOUD";

        System.out.printf("[%d] EDGE SIM: %s\n", simulationStep, edgeId);
        System.out.printf("  - State         : %s\n", patientState);
        System.out.printf("  - DQN Q-Values  : Edge=%.3f, Cloud=%.3f\n", qValues[0], qValues[1]);
        System.out.printf("  - Decision      : %s (%s)\n", decision, decisionType);
        System.out.printf("  - Outcome       : Latency=%.2fs, Edge CPU=%.1f%%, Queue=%d\n", latency, cpuUtil * 100, queue);
        System.out.printf("  - Reward        : %.2f\n", reward);
        System.out.printf("  - Stored & Trained (Loss: %.4f)\n\n", loss);
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
    private static JSONObject sendTransitionToEdgeDQN(String edgeId, double[] state, int action, double reward, 
                                              double[] nextState, boolean done, double latency, 
                                              double cpuUtil, double queueLength) throws Exception {
        URL url = new URL(EDGE_DQN_SERVER + "/store_transition");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        
        JSONObject request = new JSONObject();
        request.put("edge_id", edgeId);
        request.put("action", action);
        request.put("reward", reward);
        request.put("done", done);
        request.put("latency", latency);
        request.put("cpu_util", cpuUtil);
        request.put("queue_length", queueLength);

        // FIX: Convert both state arrays to Lists
        java.util.List<Double> stateList = new java.util.ArrayList<>();
        for (double d : state) {
            stateList.add(d);
        }
        request.put("state", stateList);

        java.util.List<Double> nextStateList = new java.util.ArrayList<>();
        for (double d : nextState) {
            nextStateList.add(d);
        }
        request.put("next_state", nextStateList);
        
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = request.toJSONString().getBytes("utf-8");
            os.write(input, 0, input.length);
        }
    
        // Now, read the response to get the loss
        if (conn.getResponseCode() == HttpURLConnection.HTTP_OK) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
                JSONParser parser = new JSONParser();
                return (JSONObject) parser.parse(br.readLine());
            }
        }
        return null; // Trigger request
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
            
            JSONObject cloudResponse = sendToCloudDQN(qVectors, systemMetrics);

            // *** ADD CLOUD LOGGING HERE ***
            System.out.println("================================================================================");
            System.out.printf("====== %28s @ STEP %d %-28s ======\n", "CLOUD COORDINATION", simulationStep, "");
            System.out.println("================================================================================");
            System.out.println("CLOUD SIM: Aggregated Q-vectors from " + qVectors.size() + " edges.");

            JSONObject strategy = (JSONObject) cloudResponse.get("allocation_strategy");
            System.out.println("  - Cloud Decision : New resource strategy is '" + strategy.get("priority").toString().toUpperCase() + "'");

            double globalReward = ((Number) cloudResponse.get("cloud_reward")).doubleValue();
            System.out.printf("  - Global Reward  : Calculated %.3f based on system health.\n", globalReward);

            System.out.println("  - Distributing Rewards to Edges:");
            if (cloudResponse.containsKey("distributed_rewards")) {
                Map<String, Double> distributedRewards = (Map<String, Double>) cloudResponse.get("distributed_rewards");
                for (Map.Entry<String, Double> entry : distributedRewards.entrySet()) {
                    sendGlobalRewardToEdge(entry.getKey(), entry.getValue());
                    System.out.printf("    - %s: %.3f\n", entry.getKey(), entry.getValue());
                }
            }
            // *** END OF LOGGING BLOCK ***

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