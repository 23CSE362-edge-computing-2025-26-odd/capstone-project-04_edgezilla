package org.fog.test.sepsisdetection;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;

import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.Application;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.application.DQNOffloadDecisionModule;
import org.fog.application.DynamicDQNSelectivity;
import org.fog.utils.SepsisStatistics;

import org.fog.entities.Actuator;
import org.fog.entities.FogBroker;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.Controller;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementMapping;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;

public class SepsisDetection {

    static final int NUM_OF_WARDS = 3;
    static final int WEARABLES_PER_WARD = 5;

    static final double SENSOR_MEAN_INTERARRIVAL_SECONDS = 10.0; // Increased from 5.0 to reduce tuple generation

    static final long WEARABLE_MIPS_PER_PE = 500;
    static final int WEARABLE_PES = 1;
    static final int WEARABLE_RAM = 256;

    static final long EDGE_MIPS_PER_PE = 8000;
    static final int EDGE_PES = 4;
    static final int EDGE_RAM = 32768;

    static final long CLOUD_MIPS_PER_PE = 20000;
    static final int CLOUD_PES = 8;
    static final int CLOUD_RAM = 65536;

    static final double LATENCY_WEARABLE_TO_EDGE_MS = 5.0;
    static final double LATENCY_EDGE_TO_CLOUD_MS = 30.0;
    static final double LATENCY_WEARABLE_SENSOR_MS = 6.0;
    static final double ACTUATOR_LATENCY_MS = 1.0;

    static final double WEARABLE_BUSY_POWER = 3.0;
    static final double WEARABLE_IDLE_POWER = 1.0;
    static final double EDGE_BUSY_POWER = 300.0;
    static final double EDGE_IDLE_POWER = 100.0;
    static final double CLOUD_BUSY_POWER = 500.0;
    static final double CLOUD_IDLE_POWER = 200.0;

    static final double OFFLOAD_TO_EDGE_FRACTION = 0.7;
    static final double OFFLOAD_TO_CLOUD_FRACTION = 0.3;

    static final double CLOUD_UPLINK_LATENCY = 100.0;

    static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    static List<FogDevice> wearables = new ArrayList<FogDevice>();
    static List<Sensor> sensors = new ArrayList<Sensor>();
    static List<Actuator> actuators = new ArrayList<Actuator>();
    static double finalSimulationTime = 0.0; // Store the final simulation time

    public static void main(String[] args) {
        Log.printLine("Starting SepsisDetection (improved) ...");

        try {
            // Set up logging to file
            PrintStream logStream = new PrintStream(new FileOutputStream("simulation.log", true));
            Log.setOutput(logStream);
            Log.enable();

            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;

            CloudSim.init(num_user, calendar, trace_flag);

            String appId = "sepsis_app";
            FogBroker broker = new FogBroker("broker");

            Application application = createApplication(appId, broker.getId());
            application.setUserId(broker.getId());

            createFogDevices();

            createEdgeDevices(broker.getId(), appId);

            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();

            for (FogDevice d : fogDevices) {
                if (d.getName().startsWith("w-")) {
                    moduleMapping.addModuleToDevice("preprocessor", d.getName());
                    moduleMapping.addModuleToDevice("patient_client", d.getName());
                }
            }

            for (int w = 0; w < NUM_OF_WARDS; w++) {
                String edgeName = "edge_server_" + w;
                moduleMapping.addModuleToDevice("OffloadDecision", edgeName);
                moduleMapping.addModuleToDevice("InferenceEdge", edgeName);
            }

            moduleMapping.addModuleToDevice("InferenceCloud", "cloud");
            moduleMapping.addModuleToDevice("Analytics", "cloud");

            Controller controller = new Controller("master-controller", fogDevices, sensors, actuators);

            controller.submitApplication(application, new ModulePlacementMapping(fogDevices, application, moduleMapping));
            
            // Temporarily disable DQN updates to fix simulation
            // startPeriodicDQNUpdates();

            try {
                Log.printLine("=== Requested moduleMapping (for reference) ===");
                for (String m : moduleMapping.getModuleMapping().keySet()) {
                    Log.printLine(m + " -> " + moduleMapping.getModuleMapping().get(m));
                }
            } catch (Exception ignored) {}

            exportConfigCSV();

            // Initialize statistics tracking
            SepsisStatistics.getInstance().reset();
            SepsisStatistics.getInstance().recordSimulationStart();

            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
            Log.printLine("Starting CloudSim simulation...");
            
            // Add a termination event to ensure simulation ends
            CloudSim.terminateSimulation(60.0); // Set to 60 seconds to see some simulation time
            
            CloudSim.startSimulation();
            Log.printLine("CloudSim simulation completed. Stopping...");
            
            // Record the final simulation time BEFORE stopping CloudSim
            finalSimulationTime = CloudSim.clock();
            Log.printLine("Final simulation time: " + finalSimulationTime);
            
            CloudSim.stopSimulation();
            Log.printLine("CloudSim stopped. Recording simulation end...");

            // Record simulation end and gather final statistics
            SepsisStatistics.getInstance().recordSimulationEnd();
            Log.printLine("Simulation end recorded. Final clock was: " + finalSimulationTime);
            Log.printLine("Simulation end recorded. Collecting loop statistics...");
            
            // Extract loop delay information from TimeKeeper results
            collectLoopStatistics();
            Log.printLine("Loop statistics collected. Exporting summary CSV...");
            
            exportSummaryCSV();
            Log.printLine("Summary CSV exported.");

            Log.printLine("SepsisDetection finished. Final simulation time was: " + finalSimulationTime);
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Exception in SepsisDetection.main()");
        }
    }

    private static void createEdgeDevices(int userId, String appId) {
        DeterministicDistribution expDist = new DeterministicDistribution(SENSOR_MEAN_INTERARRIVAL_SECONDS);

        for (FogDevice wearable : wearables) {
            String id = wearable.getName();

            Sensor hrSensor = new Sensor("s-hr-" + id, "HR_SPO2", userId, appId, expDist);
            sensors.add(hrSensor);
            hrSensor.setGatewayDeviceId(wearable.getId());
            hrSensor.setLatency(LATENCY_WEARABLE_SENSOR_MS);

            Sensor accel = new Sensor("s-accel-" + id, "ACCEL", userId, appId, expDist);
            sensors.add(accel);
            accel.setGatewayDeviceId(wearable.getId());
            accel.setLatency(LATENCY_WEARABLE_SENSOR_MS);

            Sensor gyro = new Sensor("s-gyro-" + id, "GYRO", userId, appId, expDist);
            sensors.add(gyro);
            gyro.setGatewayDeviceId(wearable.getId());
            gyro.setLatency(LATENCY_WEARABLE_SENSOR_MS);

            Sensor temp = new Sensor("s-temp-" + id, "TEMP", userId, appId, expDist);
            sensors.add(temp);
            temp.setGatewayDeviceId(wearable.getId());
            temp.setLatency(LATENCY_WEARABLE_SENSOR_MS);

            Actuator alertEdge = new Actuator("a-edge-" + id, userId, appId, "ALERT_EDGE");
            actuators.add(alertEdge);
            alertEdge.setGatewayDeviceId(wearable.getId());
            alertEdge.setLatency(ACTUATOR_LATENCY_MS);

            Actuator alertCloud = new Actuator("a-cloud-" + id, userId, appId, "ALERT_CLOUD");
            actuators.add(alertCloud);
            alertCloud.setGatewayDeviceId(wearable.getId());
            alertCloud.setLatency(ACTUATOR_LATENCY_MS);

            Log.printLine("DEBUG: created sensors and actuators for " + id + " (" + hrSensor.getName() + ", " +
                    accel.getName() + ", " + gyro.getName() + ", " + temp.getName() + ", " +
                    alertEdge.getName() + ", " + alertCloud.getName() + ")");
        }
    }

    private static void createFogDevices() {
        FogDevice cloud = createFogDevice("cloud",
                CLOUD_MIPS_PER_PE, CLOUD_PES, CLOUD_RAM,
                100000, 100000, 0, 0.01, CLOUD_BUSY_POWER, CLOUD_IDLE_POWER);
        cloud.setParentId(-1);
        cloud.setUplinkLatency(CLOUD_UPLINK_LATENCY);
        fogDevices.add(cloud);
        Log.printLine("Created cloud device: " + cloud.getName());

        for (int w = 0; w < NUM_OF_WARDS; w++) {
            String edgeName = "edge_server_" + w;
            FogDevice edgeServer = createFogDevice(edgeName,
                    EDGE_MIPS_PER_PE, EDGE_PES, EDGE_RAM,
                    50000, 50000, 1, 0.0, EDGE_BUSY_POWER, EDGE_IDLE_POWER);
            edgeServer.setParentId(cloud.getId());
            edgeServer.setUplinkLatency(LATENCY_EDGE_TO_CLOUD_MS);
            fogDevices.add(edgeServer);
            Log.printLine("Created edge device: " + edgeServer.getName());

            for (int i = 0; i < WEARABLES_PER_WARD; i++) {
                String wearableId = w + "-" + i;
                FogDevice wearable = createFogDevice("w-" + wearableId,
                        WEARABLE_MIPS_PER_PE, WEARABLE_PES, WEARABLE_RAM,
                        5000, 5000, 3, 0.0, WEARABLE_BUSY_POWER, WEARABLE_IDLE_POWER);
                wearable.setParentId(edgeServer.getId());
                wearable.setUplinkLatency(LATENCY_WEARABLE_TO_EDGE_MS);
                wearables.add(wearable);
                fogDevices.add(wearable);
                Log.printLine("Created wearable device: " + wearable.getName());
            }
        }
    }

    private static FogDevice createFogDevice(String nodeName, long mipsPerPe, int pes,
            int ram, long upBw, long downBw, int level, double ratePerMips,
            double busyPower, double idlePower) {

        List<Pe> peList = new ArrayList<Pe>();
        for (int i = 0; i < pes; i++) {
            peList.add(new Pe(i, new PeProvisionerOverbooking(mipsPerPe)));
        }

        int hostId = FogUtils.generateEntityId();
        long storage = 1000000L;
        int bw = 10000;

        PowerHost host = new PowerHost(hostId,
                new RamProvisionerSimple(ram),
                new BwProvisionerOverbooking(bw),
                storage,
                peList,
                new StreamOperatorScheduler(peList),
                new FogLinearPowerModel(busyPower, idlePower));

        List<Host> hostList = new ArrayList<Host>();
        hostList.add(host);

        LinkedList<Storage> storageList = new LinkedList<Storage>();

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(arch, os, vmm, host, time_zone, cost,
                costPerMem, costPerStorage, costPerBw);

        FogDevice fogdevice = null;
        try {
            fogdevice = new FogDevice(nodeName, characteristics, new AppModuleAllocationPolicy(hostList), storageList,
                    10, upBw, downBw, 0, ratePerMips);
        } catch (Exception e) {
            e.printStackTrace();
        }
        fogdevice.setLevel(level);
        return fogdevice;
    }

    @SuppressWarnings({ "serial" })
    private static Application createApplication(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);

        application.addAppModule("patient_client", 128);
        application.addAppModule("preprocessor", 32);
        // CHANGE 1: Use the new custom AppModule for decision making
        application.addAppModule("OffloadDecision", 512);
        application.addAppModule("InferenceEdge", 1024);
        application.addAppModule("InferenceCloud", 2048);
        application.addAppModule("Analytics", 128);

        // Sensor data processing edges
        application.addAppEdge("HR_SPO2", "patient_client", 100.0, 500.0, "HR_SPO2", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("ACCEL", "patient_client", 50.0, 200.0, "ACCEL", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("GYRO", "patient_client", 50.0, 200.0, "GYRO", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("TEMP", "patient_client", 30.0, 50.0, "TEMP", Tuple.UP, AppEdge.SENSOR);

        // Data flow edges
        application.addAppEdge("patient_client", "preprocessor", 50.0, 100.0, "T_CLIENT_RAW", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("preprocessor", "OffloadDecision", 100.0, 200.0, "T_PREPROCESSED", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("OffloadDecision", "InferenceEdge", 2000.0, 200.0, "T_INFER_REQUEST", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("OffloadDecision", "InferenceCloud", 2000.0, 200.0, "T_OFFLOAD_TO_CLOUD", Tuple.UP, AppEdge.MODULE);

        // CHANGE 2: The result from inference now goes directly to an actuator, breaking the loop.
        application.addAppEdge("InferenceEdge", "ALERT_EDGE", 50.0, 50.0, "T_INFER_RESULT", Tuple.DOWN, AppEdge.ACTUATOR);
        application.addAppEdge("InferenceCloud", "ALERT_CLOUD", 50.0, 100.0, "T_CLOUD_RESULT", Tuple.DOWN, AppEdge.ACTUATOR);

        // Fix: Also add edges back to patient_client for proper flow
        application.addAppEdge("InferenceEdge", "patient_client", 50.0, 50.0, "T_INFER_RESULT", Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge("InferenceCloud", "patient_client", 50.0, 100.0, "T_CLOUD_RESULT", Tuple.DOWN, AppEdge.MODULE);

        application.addAppEdge("patient_client", "ALERT_EDGE", 20.0, 50.0, "T_ALERT_EDGE", Tuple.DOWN, AppEdge.ACTUATOR);
        application.addAppEdge("patient_client", "ALERT_CLOUD", 20.0, 50.0, "T_ALERT_CLOUD", Tuple.DOWN, AppEdge.ACTUATOR);

        // Analytics path remains the same
        application.addAppEdge("InferenceEdge", "Analytics", 100.0, 50.0, "T_INFER_RESULT", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("InferenceCloud", "Analytics", 100.0, 50.0, "T_CLOUD_RESULT", Tuple.UP, AppEdge.MODULE);

        // Tuple mappings for the first steps
        application.addTupleMapping("patient_client", "HR_SPO2", "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "ACCEL", "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "GYRO", "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "TEMP", "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("preprocessor", "T_CLIENT_RAW", "T_PREPROCESSED", new FractionalSelectivity(1.0));

        // Add back the DQN selectivity for offloading decisions
        DynamicDQNSelectivity edgeSelectivity = new DynamicDQNSelectivity("EDGE");
        DynamicDQNSelectivity cloudSelectivity = new DynamicDQNSelectivity("CLOUD");
        application.addTupleMapping("OffloadDecision", "T_PREPROCESSED", "T_INFER_REQUEST", edgeSelectivity);
        application.addTupleMapping("OffloadDecision", "T_PREPROCESSED", "T_OFFLOAD_TO_CLOUD", cloudSelectivity);

        application.addTupleMapping("InferenceEdge", "T_INFER_REQUEST", "T_INFER_RESULT", new FractionalSelectivity(1.0));
        application.addTupleMapping("InferenceCloud", "T_OFFLOAD_TO_CLOUD", "T_CLOUD_RESULT", new FractionalSelectivity(1.0));

        // CHANGE 4: The patient_client to actuator mappings are also DELETED.
        // This is now handled by the Inference modules directly.
        
        // Add back the patient_client to actuator mappings
        application.addTupleMapping("patient_client", "T_INFER_RESULT", "T_ALERT_EDGE", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "T_CLOUD_RESULT", "T_ALERT_CLOUD", new FractionalSelectivity(1.0));

        List<AppLoop> loops = new ArrayList<AppLoop>();

        // CHANGE 5: Correct the AppLoop definitions to reflect the new, shorter data flow.
        // The loop now ends when the actuator receives the result.
        AppLoop edgeLoop = new AppLoop(new ArrayList<String>() {{
            add("HR_SPO2"); // Start from a sensor
            add("patient_client");
            add("preprocessor");
            add("OffloadDecision");
            add("InferenceEdge");
            add("ALERT_EDGE"); // End at the actuator
        }});
        loops.add(edgeLoop);

        AppLoop cloudLoop = new AppLoop(new ArrayList<String>() {{
            add("HR_SPO2"); // Start from a sensor
            add("patient_client");
            add("preprocessor");
            add("OffloadDecision");
            add("InferenceCloud");
            add("ALERT_CLOUD"); // End at the actuator
        }});
        loops.add(cloudLoop);

        application.setLoops(loops);

        return application;
    }

    private static void exportConfigCSV() {
        String fname = "sepsis_sim_config.csv";
        try (PrintStream fw = new PrintStream(fname)) {
            fw.append("parameter,value\n");
            fw.append("NUM_OF_WARDS,").append(Integer.toString(NUM_OF_WARDS)).append("\n");
            fw.append("WEARABLES_PER_WARD,").append(Integer.toString(WEARABLES_PER_WARD)).append("\n");
            fw.append("SENSOR_MEAN_INTERARRIVAL_SECONDS,").append(Double.toString(SENSOR_MEAN_INTERARRIVAL_SECONDS)).append("\n");
            fw.append("WEARABLE_MIPS_PER_PE,").append(Long.toString(WEARABLE_MIPS_PER_PE)).append("\n");
            fw.append("WEARABLE_PES,").append(Integer.toString(WEARABLE_PES)).append("\n");
            fw.append("EDGE_MIPS_PER_PE,").append(Long.toString(EDGE_MIPS_PER_PE)).append("\n");
            fw.append("EDGE_PES,").append(Integer.toString(EDGE_PES)).append("\n");
            fw.append("CLOUD_MIPS_PER_PE,").append(Long.toString(CLOUD_MIPS_PER_PE)).append("\n");
            fw.append("CLOUD_PES,").append(Integer.toString(CLOUD_PES)).append("\n");
            fw.append("OFFLOAD_TO_EDGE_FRACTION,").append(Double.toString(OFFLOAD_TO_EDGE_FRACTION)).append("\n");
            fw.append("OFFLOAD_TO_CLOUD_FRACTION,").append(Double.toString(OFFLOAD_TO_CLOUD_FRACTION)).append("\n");
            fw.flush();
            Log.printLine("Exported experiment configuration to " + fname);
        } catch (Exception e) {
            Log.printLine("Failed to write config CSV: " + e.getMessage());
        }
    }

    private static void exportSummaryCSV() {
        String fname = "sepsis_sim_summary.csv";
        try (PrintStream fw = new PrintStream(fname)) {
            SepsisStatistics stats = SepsisStatistics.getInstance();
            
            fw.append("metric,value\n");
            
            // Basic simulation info
            fw.append("simulation_time_s,").append(Double.toString(stats.getSimulationDuration())).append("\n");
            fw.append("simulation_clock_final,").append(Double.toString(finalSimulationTime)).append("\n");
            fw.append("num_fog_devices,").append(Integer.toString(fogDevices.size())).append("\n");
            fw.append("num_wearables,").append(Integer.toString(wearables.size())).append("\n");
            fw.append("num_sensors,").append(Integer.toString(sensors.size())).append("\n");
            fw.append("num_actuators,").append(Integer.toString(actuators.size())).append("\n");
            
            // Application loop metrics
            fw.append("total_edge_loops,").append(Integer.toString(stats.getTotalEdgeLoops())).append("\n");
            fw.append("total_cloud_loops,").append(Integer.toString(stats.getTotalCloudLoops())).append("\n");
            fw.append("avg_edge_loop_latency_ms,").append(Double.toString(stats.getAverageEdgeLoopLatency())).append("\n");
            fw.append("avg_cloud_loop_latency_ms,").append(Double.toString(stats.getAverageCloudLoopLatency())).append("\n");
            
            // Alert counts
            fw.append("edge_alert_count,").append(Integer.toString(stats.getEdgeAlertCount())).append("\n");
            fw.append("cloud_alert_count,").append(Integer.toString(stats.getCloudAlertCount())).append("\n");
            
            // Tuple processing counts
            fw.append("preprocessed_tuples,").append(Integer.toString(stats.getTupleProcessingCount("T_PREPROCESSED"))).append("\n");
            fw.append("edge_inference_tuples,").append(Integer.toString(stats.getTupleProcessingCount("T_INFER_REQUEST"))).append("\n");
            fw.append("cloud_inference_tuples,").append(Integer.toString(stats.getTupleProcessingCount("T_OFFLOAD_TO_CLOUD"))).append("\n");
            
            // Energy metrics
            fw.append("total_energy_consumption,").append(Double.toString(stats.getTotalEnergyConsumption())).append("\n");
            
            // DQN performance metrics (if available)
            if (stats.getTotalEdgeLoops() > 0 && stats.getTotalCloudLoops() > 0) {
                double edgeCloudRatio = (double) stats.getTotalEdgeLoops() / (stats.getTotalEdgeLoops() + stats.getTotalCloudLoops());
                fw.append("dqn_edge_selection_ratio,").append(Double.toString(edgeCloudRatio)).append("\n");
            } else {
                fw.append("dqn_edge_selection_ratio,").append("0.0").append("\n");
            }
            
            fw.flush();
            Log.printLine("Exported simulation summary with calculated metrics to " + fname);
        } catch (Exception e) {
            Log.printLine("Failed to write summary CSV: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Collect loop statistics from simulation results
     */
    private static void collectLoopStatistics() {
        try {
            SepsisStatistics stats = SepsisStatistics.getInstance();
            
            // Get actual counts from the statistics instead of using estimates
            int actualEdgeLoops = stats.getTotalEdgeLoops();
            int actualCloudLoops = stats.getTotalCloudLoops();
            int actualPreprocessedTuples = stats.getTupleProcessingCount("T_PREPROCESSED");
            
            Log.printLine("Actual statistics from simulation:");
            Log.printLine("  - Edge loops: " + actualEdgeLoops);
            Log.printLine("  - Cloud loops: " + actualCloudLoops);
            Log.printLine("  - Preprocessed tuples: " + actualPreprocessedTuples);
            Log.printLine("  - Final simulation clock: " + CloudSim.clock());
            
            // If we don't have real statistics yet, that means the DQN decisions weren't recorded properly
            // Let's not override with hardcoded values
            if (actualEdgeLoops == 0 && actualCloudLoops == 0 && actualPreprocessedTuples == 0) {
                Log.printLine("Warning: No actual statistics recorded during simulation. This means DQN decisions may not be working.");
            }
            
        } catch (Exception e) {
            Log.printLine("Error collecting loop statistics: " + e.getMessage());
            e.printStackTrace();
        }
    }
}