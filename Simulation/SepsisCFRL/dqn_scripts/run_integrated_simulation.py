#!/usr/bin/env python3
"""
Integration test script for Sepsis Detection with DQN servers.
This script starts the Edge and Cloud DQN servers and then runs the Java simulation.
"""

import subprocess
import time
import sys
import os
import signal
import threading
import requests

def check_server_health(url, server_name, max_retries=10):
    """Check if a server is healthy and responding"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ {server_name} is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting for {server_name} to start... ({i+1}/{max_retries})")
        time.sleep(2)
    return False

def start_server(script_name, server_name):
    """Start a DQN server in a subprocess"""
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Started {server_name} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"Failed to start {server_name}: {e}")
        return None

def run_java_simulation():
    """Run the Java simulation"""
    print("\n" + "="*50)
    print("STARTING SEPSIS DETECTION SIMULATION")
    print("="*50)
    
    # Change to the simulation directory
    sim_dir = os.path.abspath("../")
    
    try:
        # Compile Java classes
        print("Compiling all Java source files...")
        compile_cmd = [
            "javac", "-cp", ".:bin:src:lib/*:lib/commons-math3-3.5/*",
            "-d", "bin",
            "src/org/fog/test/sepsisdetection/SepsisDetection.java",
            "src/org/fog/utils/SepsisStatistics.java",
            "src/org/fog/application/DQNOffloadDecisionModule.java",
            "src/org/fog/application/DynamicDQNSelectivity.java"
        ]
        
        compile_result = subprocess.run(
            compile_cmd, 
            cwd=sim_dir, 
            capture_output=True, 
            text=True
        )
        
        if compile_result.returncode != 0:
            print(f"Compilation failed: {compile_result.stderr}")
            return False
        
        print("✓ Compilation successful")
        
        # Run the simulation
        print("Running simulation...")
        run_cmd = [
            "java", "-cp", ".:bin:src:lib/*:lib/commons-math3-3.5/*", 
            "org.fog.test.sepsisdetection.SepsisDetection"
        ]
        
        # Run with timeout
        simulation_process = subprocess.Popen(
            run_cmd,
            cwd=sim_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = simulation_process.communicate(timeout=180)  # 3 minute timeout
            
            print("\n--- SIMULATION OUTPUT ---")
            print(stdout)
            if stderr:
                print("\n--- SIMULATION ERRORS ---")
                print(stderr)
            
            if simulation_process.returncode == 0:
                print("✓ Simulation completed successfully")
                return True
            else:
                print(f"✗ Simulation failed with return code {simulation_process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            simulation_process.kill()
            print("✗ Simulation timed out")
            return False
            
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

def main():
    print("="*60)
    print("SEPSIS DETECTION CFRL INTEGRATION TEST")
    print("="*60)
    
    edge_server = None
    cloud_server = None
    
    try:
        # Start Edge DQN Server
        print("Starting Edge DQN Server...")
        edge_server = start_server("edge_dqn_server.py", "Edge DQN Server")
        if not edge_server:
            return 1
        
        # Start Cloud DQN Server  
        print("Starting Cloud DQN Server...")
        cloud_server = start_server("cloud_dqn_server.py", "Cloud DQN Server")
        if not cloud_server:
            return 1
        
        # Wait for servers to start and check health
        print("\nWaiting for servers to initialize...")
        time.sleep(3)
        
        edge_healthy = check_server_health("http://localhost:5000", "Edge DQN Server")
        cloud_healthy = check_server_health("http://localhost:5001", "Cloud DQN Server")
        
        if not edge_healthy or not cloud_healthy:
            print("✗ One or more servers failed to start properly")
            return 1
        
        print("\n✓ All DQN servers are running and healthy")
        
        # Run the Java simulation
        success = run_java_simulation()
        
        if success:
            print("\n" + "="*60)
            print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nCheck the following files for results:")
            print("- sepsis_sim_config.csv")
            print("- sepsis_sim_summary.csv")
            return 0
        else:
            print("\n" + "="*60)
            print("INTEGRATION TEST FAILED!")
            print("="*60)
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Clean up servers
        print("\nShutting down servers...")
        if edge_server and edge_server.poll() is None:
            edge_server.terminate()
            edge_server.wait(timeout=5)
            print("✓ Edge DQN Server stopped")
        
        if cloud_server and cloud_server.poll() is None:
            cloud_server.terminate()
            cloud_server.wait(timeout=5)
            print("✓ Cloud DQN Server stopped")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 