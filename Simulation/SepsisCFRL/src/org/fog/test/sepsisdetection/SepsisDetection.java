package org.fog.test.sepsisdetection;

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

/**
 * Sepsis detection with two separate actuators per wearable:
 *  - ALERT_EDGE  : actuator type for edge-delivered alerts
 *  - ALERT_CLOUD : actuator type for cloud-delivered alerts
 *
 * InferenceEdge -> ALERT_EDGE
 * InferenceCloud -> ALERT_CLOUD
 *
 * This file is a drop-in variant of your working code with those changes.
 */
public class SepsisDetection {

    static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
    static List<FogDevice> wearables = new ArrayList<FogDevice>();
    static List<Sensor> sensors = new ArrayList<Sensor>();
    static List<Actuator> actuators = new ArrayList<Actuator>();

    // SCALE SETTINGS
    static int numOfWards = 3;
    static int numOfWearablesPerWard = 5;
    static double SENSOR_TRANSMISSION_TIME = 5.0; // seconds

    public static void main(String[] args) {
        Log.printLine("Starting SepsisDetection (edge/cloud-actuators) ...");

        try {
            // toggle logging as needed
            // Log.enable();
            Log.disable();

            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;

            CloudSim.init(num_user, calendar, trace_flag);

            String appId = "sepsis_app";
            FogBroker broker = new FogBroker("broker");

            Application application = createApplication(appId, broker.getId());
            application.setUserId(broker.getId());

            // Build physical topology
            createFogDevices();

            // Attach sensors & actuators to wearables (use broker userId and appId)
            createEdgeDevices(broker.getId(), appId);

            // Module mapping (deterministic)
            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();

            for (FogDevice d : fogDevices) {
                if (d.getName().startsWith("w-")) {
                    moduleMapping.addModuleToDevice("preprocessor", d.getName());
                    moduleMapping.addModuleToDevice("patient_client", d.getName());
                }
            }

            // map edge modules explicitly to edge servers
            for (int w = 0; w < numOfWards; w++) {
                String edgeName = "edge_server_" + w;
                moduleMapping.addModuleToDevice("OffloadDecision", edgeName);
                moduleMapping.addModuleToDevice("InferenceEdge", edgeName);
            }

            // cloud modules
            moduleMapping.addModuleToDevice("InferenceCloud", "cloud");
            moduleMapping.addModuleToDevice("Analytics", "cloud");

            Controller controller = new Controller("master-controller", fogDevices, sensors, actuators);

            // Use explicit mapping placement
            controller.submitApplication(application, new ModulePlacementMapping(fogDevices, application, moduleMapping));

            // Debug: print requested mapping (optional)
            try {
                Log.printLine("=== Requested moduleMapping (for reference) ===");
                for (String m : moduleMapping.getModuleMapping().keySet()) {
                    Log.printLine(m + " -> " + moduleMapping.getModuleMapping().get(m));
                }
            } catch (Exception ignored) {}

            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());

            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            Log.printLine("SepsisDetection finished. Clock: " + CloudSim.clock());
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Exception in SepsisDetection.main()");
        }
    }

    /**
     * Attach sensors/actuators to each wearable.
     * Now creates two actuators per wearable:
     *   - ALERT_EDGE  (for alerts triggered by edge inference)
     *   - ALERT_CLOUD (for alerts triggered by cloud inference)
     */
    private static void createEdgeDevices(int userId, String appId) {
        for (FogDevice wearable : wearables) {
            String id = wearable.getName();

            // Heart rate / SpO2 sensor
            Sensor hrSensor = new Sensor("s-hr-" + id, "HR_SPO2", userId, appId,
                    new DeterministicDistribution(SENSOR_TRANSMISSION_TIME));
            sensors.add(hrSensor);
            hrSensor.setGatewayDeviceId(wearable.getId());
            hrSensor.setLatency(6.0);

            // Accelerometer
            Sensor accel = new Sensor("s-accel-" + id, "ACCEL", userId, appId,
                    new DeterministicDistribution(SENSOR_TRANSMISSION_TIME));
            sensors.add(accel);
            accel.setGatewayDeviceId(wearable.getId());
            accel.setLatency(6.0);

            // Gyroscope
            Sensor gyro = new Sensor("s-gyro-" + id, "GYRO", userId, appId,
                    new DeterministicDistribution(SENSOR_TRANSMISSION_TIME));
            sensors.add(gyro);
            gyro.setGatewayDeviceId(wearable.getId());
            gyro.setLatency(6.0);

            // Temperature
            Sensor temp = new Sensor("s-temp-" + id, "TEMP", userId, appId,
                    new DeterministicDistribution(SENSOR_TRANSMISSION_TIME));
            sensors.add(temp);
            temp.setGatewayDeviceId(wearable.getId());
            temp.setLatency(6.0);

            // Actuator for edge alerts
            Actuator alertEdge = new Actuator("a-edge-" + id, userId, appId, "ALERT_EDGE");
            actuators.add(alertEdge);
            alertEdge.setGatewayDeviceId(wearable.getId());
            alertEdge.setLatency(1.0);

            // Actuator for cloud alerts
            Actuator alertCloud = new Actuator("a-cloud-" + id, userId, appId, "ALERT_CLOUD");
            actuators.add(alertCloud);
            alertCloud.setGatewayDeviceId(wearable.getId());
            alertCloud.setLatency(1.0);

            Log.printLine("DEBUG: created sensors and actuators for " + id + " (" + hrSensor.getName() + ", " +
                    accel.getName() + ", " + gyro.getName() + ", " + temp.getName() + ", " +
                    alertEdge.getName() + ", " + alertCloud.getName() + ")");
        }
    }

    /**
     * Build topology: cloud -> edge_server_0..N -> wearables per ward
     */
    private static void createFogDevices() {
        // CLOUD
        FogDevice cloud = createFogDevice("cloud", 20000, 8, 65536, 100000, 100000, 0, 0.01, 500.0, 200.0);
        cloud.setParentId(-1);
        cloud.setUplinkLatency(100.0);
        fogDevices.add(cloud);

        // EDGE SERVERS and wearables
        for (int w = 0; w < numOfWards; w++) {
            String edgeName = "edge_server_" + w;
            FogDevice edgeServer = createFogDevice(edgeName, 8000, 4, 32768, 50000, 50000, 1, 0.0, 300.0, 100.0);
            edgeServer.setParentId(cloud.getId());
            edgeServer.setUplinkLatency(30.0);
            fogDevices.add(edgeServer);

            for (int i = 0; i < numOfWearablesPerWard; i++) {
                String wearableId = w + "-" + i;
                FogDevice wearable = createFogDevice("w-" + wearableId, 500, 1, 256, 5000, 5000, 3, 0.0, 3.0, 1.0);
                wearable.setParentId(edgeServer.getId());
                wearable.setUplinkLatency(5.0);
                wearables.add(wearable);
                fogDevices.add(wearable);
            }
        }
    }

    /**
     * createFogDevice variant with 'pes' for multiple PEs.
     */
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

    /**
     * Application graph:
     * - All sensors send to patient_client (aggregation)
     * - patient_client -> preprocessor -> OffloadDecision
     * - OffloadDecision -> InferenceEdge (edge) and -> InferenceCloud (cloud)
     * - InferenceEdge -> ALERT_EDGE actuator
     * - InferenceCloud -> ALERT_CLOUD actuator
     */
    @SuppressWarnings({ "serial" })
    private static Application createApplication(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);

        // --- Modules & RAM ---
        application.addAppModule("patient_client", 128);
        application.addAppModule("preprocessor", 32);
        application.addAppModule("OffloadDecision", 512);
        application.addAppModule("InferenceEdge", 1024);
        application.addAppModule("InferenceCloud", 2048);
        application.addAppModule("Analytics", 128);

        // --- App Edges (data flows) ---
        // Sensors -> patient_client (aggregation point)
        // NOTE: tupleType strings MUST match the sensor types you create in createEdgeDevices()
        application.addAppEdge("HR_SPO2", "patient_client", 3000.0, 500.0, "HR_SPO2", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("ACCEL",  "patient_client", 1000.0, 200.0, "ACCEL",  Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("GYRO",   "patient_client", 1000.0, 200.0, "GYRO",   Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("TEMP",   "patient_client", 300.0,  50.0,  "TEMP",   Tuple.UP, AppEdge.SENSOR);

        // patient_client -> preprocessor (local cleaning / aggregation)
        application.addAppEdge("patient_client", "preprocessor", 100.0, 100.0, "T_CLIENT_RAW", Tuple.UP, AppEdge.MODULE);

        // preprocessor -> OffloadDecision
        application.addAppEdge("preprocessor", "OffloadDecision", 500.0, 200.0, "T_PREPROCESSED", Tuple.UP, AppEdge.MODULE);

        // OffloadDecision -> InferenceEdge / InferenceCloud
        application.addAppEdge("OffloadDecision", "InferenceEdge", 2000.0, 200.0, "T_INFER_REQUEST", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("OffloadDecision", "InferenceCloud", 2000.0, 200.0, "T_OFFLOAD_TO_CLOUD", Tuple.UP, AppEdge.MODULE);

        // Inference results -> back to patient_client (DOWN so patient_client can generate actuator tuples)
        application.addAppEdge("InferenceEdge",  "patient_client", 50.0,  50.0,  "T_INFER_RESULT", Tuple.DOWN, AppEdge.MODULE);
        application.addAppEdge("InferenceCloud", "patient_client", 50.0,  100.0, "T_CLOUD_RESULT",  Tuple.DOWN, AppEdge.MODULE);

        // patient_client -> Actuators (two different actuator types)
        application.addAppEdge("patient_client", "ALERT_EDGE",  20.0, 50.0, "T_ALERT_EDGE",  Tuple.DOWN, AppEdge.ACTUATOR);
        application.addAppEdge("patient_client", "ALERT_CLOUD", 20.0, 50.0, "T_ALERT_CLOUD", Tuple.DOWN, AppEdge.ACTUATOR);

        // Optional: analytics paths if needed
        application.addAppEdge("InferenceEdge",  "Analytics", 100.0, 50.0, "T_INFER_RESULT", Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("InferenceCloud", "Analytics", 100.0, 50.0, "T_CLOUD_RESULT",  Tuple.UP, AppEdge.MODULE);

        // --- Tuple mappings (selectivities) ---
        // patient_client consolidates sensor inputs into T_CLIENT_RAW
        application.addTupleMapping("patient_client", "HR_SPO2", "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "ACCEL",   "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "GYRO",    "T_CLIENT_RAW", new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "TEMP",    "T_CLIENT_RAW", new FractionalSelectivity(1.0));

        // preprocessor -> T_PREPROCESSED
        application.addTupleMapping("preprocessor", "T_CLIENT_RAW", "T_PREPROCESSED", new FractionalSelectivity(1.0));

        // OffloadDecision splits work to edge/cloud (tune these ratios)
        application.addTupleMapping("OffloadDecision", "T_PREPROCESSED", "T_INFER_REQUEST", new FractionalSelectivity(0.7)); // edge
        application.addTupleMapping("OffloadDecision", "T_PREPROCESSED", "T_OFFLOAD_TO_CLOUD", new FractionalSelectivity(0.3)); // cloud

        // Inference -> results (1:1)
        application.addTupleMapping("InferenceEdge", "T_INFER_REQUEST", "T_INFER_RESULT", new FractionalSelectivity(1.0));
        application.addTupleMapping("InferenceCloud", "T_OFFLOAD_TO_CLOUD", "T_CLOUD_RESULT", new FractionalSelectivity(1.0));

        // patient_client converts inference results -> respective actuator tuples
        application.addTupleMapping("patient_client", "T_INFER_RESULT", "T_ALERT_EDGE",  new FractionalSelectivity(1.0));
        application.addTupleMapping("patient_client", "T_CLOUD_RESULT", "T_ALERT_CLOUD", new FractionalSelectivity(1.0));

        // --- Application loops (start at patient_client so sensor aggregation is included) ---
        List<AppLoop> loops = new ArrayList<AppLoop>();

        // Edge path loop: patient_client -> preprocessor -> OffloadDecision -> InferenceEdge -> patient_client -> ALERT_EDGE
        loops.add(new AppLoop(new ArrayList<String>() {{
            add("patient_client");
            add("preprocessor");
            add("OffloadDecision");
            add("InferenceEdge");
            add("patient_client");
        }}));

        // Cloud path loop: patient_client -> preprocessor -> OffloadDecision -> InferenceCloud -> patient_client -> ALERT_CLOUD
        loops.add(new AppLoop(new ArrayList<String>() {{
            add("patient_client");
            add("preprocessor");
            add("OffloadDecision");
            add("InferenceCloud");
            add("patient_client");
        }}));

        application.setLoops(loops);

        return application;
    }
}