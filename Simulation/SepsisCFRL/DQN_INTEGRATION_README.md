# DQN Integration with Sepsis Detection iFogSim

This document explains the integration of Deep Q-Network (DQN) based Concurrent Federated Reinforcement Learning (CFRL) with the Sepsis Detection simulation in iFogSim.

## System Architecture

The integrated system consists of three main components:

1. **Edge DQN Server** (`edge_dqn_server.py`) - Handles local decision making at edge devices
2. **Cloud DQN Server** (`cloud_dqn_server.py`) - Performs global aggregation and resource allocation
3. **iFogSim Simulation** (`SepsisDetection.java`) - The main fog computing simulation with integrated DQN decision making

## How It Works

### The Edge-Cloud DQN Workflow

1. **Local Edge Loop** (Continuous):
   - Edge devices receive patient vitals (sensor data)
   - Each edge device has a local DQN agent that decides whether to process locally or offload to cloud
   - Decisions are based on current state: CPU utilization, memory usage, queue length, latency
   - Local rewards are calculated based on execution performance

2. **Global Cloud Loop** (Periodic - every 5 decisions):
   - Cloud DQN aggregates Q-vectors from all edge devices (privacy-preserving)
   - Makes global resource allocation decisions
   - Calculates and distributes rewards back to edge devices
   - Enables federated learning across the system

### Key Integration Points

- **DQNOffloadDecisionModule.java**: Main integration point between Java simulation and DQN servers
- **DynamicDQNSelectivity.java**: Replaces static selectivity with dynamic DQN-based decisions
- **Modified SepsisDetection.java**: Uses dynamic selectivity and periodic DQN updates

## Files Created/Modified

### New Files
- `src/org/fog/application/DQNOffloadDecisionModule.java` - Core DQN integration module
- `src/org/fog/application/DynamicDQNSelectivity.java` - Dynamic selectivity model
- `dqn_scripts/run_integrated_simulation.py` - Integration test script
- `DQN_INTEGRATION_README.md` - This documentation

### Modified Files
- `src/org/fog/test/sepsisdetection/SepsisDetection.java` - Integrated DQN decision making

## Setup and Installation

### Prerequisites
- Python 3.7+ with packages: `flask`, `torch`, `numpy`
- Java 8+ with iFogSim dependencies
- `requests` library for Python (for the test script)

### Installation Steps

1. **Install Python Dependencies**:
   ```bash
   cd dqn_scripts
   pip install flask torch numpy requests
   ```

2. **Verify Java Environment**:
   - Ensure iFogSim is properly set up
   - Check that all required JAR files are in the classpath

## Running the System

### Option 1: Automated Integration Test (Recommended)
```bash
cd dqn_scripts
python run_integrated_simulation.py
```

This script will:
- Start both Edge and Cloud DQN servers
- Wait for them to initialize
- Run the Java simulation
- Show real-time output and results
- Clean up servers when done

### Option 2: Manual Execution

1. **Start Edge DQN Server**:
   ```bash
   cd dqn_scripts
   python edge_dqn_server.py
   ```

2. **Start Cloud DQN Server** (in another terminal):
   ```bash
   cd dqn_scripts
   python cloud_dqn_server.py
   ```

3. **Run iFogSim Simulation** (in another terminal):
   ```bash
   cd ..  # Go back to SepsisCFRL directory
   javac -cp ".:bin:src" -d bin src/org/fog/test/sepsisdetection/SepsisDetection.java
   java -cp ".:bin:src:lib/*" org.fog.test.sepsisdetection.SepsisDetection
   ```

## System Configuration

### Edge DQN Configuration
- **State Dimensions**: 6 (CPU util, memory util, queue length, latency, throughput, time)
- **Actions**: 2 (0=Edge processing, 1=Cloud offloading)
- **Network**: 3-layer neural network (256→128→64→2)
- **Learning Rate**: 0.001
- **Replay Buffer**: 10,000 experiences

### Cloud DQN Configuration
- **State Dimensions**: 9 (aggregated Q-vectors + system metrics)
- **Actions**: 4 resource allocation strategies
- **Network**: 4-layer neural network (512→256→128→4)
- **Learning Rate**: 0.0005
- **Replay Buffer**: 20,000 experiences

### Simulation Parameters
- **Number of Wards**: 3
- **Wearables per Ward**: 5
- **Sensor Inter-arrival**: 5 seconds (mean)
- **DQN Update Interval**: Every 5 seconds
- **Cloud Aggregation**: Every 5 decisions

## Monitoring and Results

### Real-time Monitoring
- Check server logs for DQN decisions and Q-values
- Monitor CPU/memory utilization of fog devices
- Track offloading patterns (edge vs cloud)

### Output Files
- `sepsis_sim_config.csv` - Simulation configuration parameters
- `sepsis_sim_summary.csv` - Performance metrics summary
- Console output shows DQN decisions in real-time

### Key Metrics to Watch
- **Decision Distribution**: Percentage of edge vs cloud processing
- **Average Latency**: End-to-end response times
- **Resource Utilization**: CPU and memory usage patterns
- **Learning Progress**: DQN epsilon decay and loss values

## Troubleshooting

### Common Issues

1. **Server Connection Errors**:
   - Ensure DQN servers are running before starting simulation
   - Check if ports 5000 and 5001 are available
   - Verify firewall settings

2. **Compilation Errors**:
   - Check Java classpath includes all iFogSim libraries
   - Ensure JSON simple library is available
   - Verify Java version compatibility

3. **Performance Issues**:
   - Reduce simulation scale for testing
   - Adjust DQN update intervals
   - Monitor system resources

### Debug Mode
To enable detailed logging, modify the DQN servers:
```python
app.run(host='0.0.0.0', port=5000, debug=True)  # For edge server
app.run(host='0.0.0.0', port=5001, debug=True)  # For cloud server
```

## API Endpoints

### Edge DQN Server (Port 5000)
- `POST /get_q_values` - Get Q-values for a state
- `POST /store_transition` - Store experience for learning
- `POST /set_global_reward` - Receive global reward from cloud
- `POST /get_q_vectors` - Get Q-vectors for cloud aggregation
- `GET /health` - Health check

### Cloud DQN Server (Port 5001)
- `POST /aggregate_and_allocate` - Process edge Q-vectors and allocate resources
- `GET /health` - Health check

## Performance Tuning

### For Better Learning
- Increase replay buffer size for more stable learning
- Adjust epsilon decay for exploration/exploitation balance
- Tune reward functions based on system objectives

### For Better Performance
- Reduce state vector dimensions if possible
- Batch multiple decisions for efficiency
- Use asynchronous processing for high-throughput scenarios

## Future Enhancements

1. **Multi-tenant Support**: Handle multiple applications simultaneously
2. **Advanced State Representation**: Include network topology information
3. **Federated Model Sharing**: Share neural network weights instead of just Q-vectors
4. **Real-time Adaptation**: Dynamic learning rate adjustment based on environment changes

## References

- iFogSim: https://github.com/Cloudslab/iFogSim
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Federated Learning: https://federated.withgoogle.com/

---

For questions or issues, please check the simulation logs and verify that all components are properly configured and running. 