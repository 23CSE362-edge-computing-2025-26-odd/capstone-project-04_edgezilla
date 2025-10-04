# Updated Data Generator with CSV and ML Server Integration

## Overview
The data generator has been completely updated to:
1. **Read patient data from CSV files** instead of generating synthetic data
2. **Make API calls to an ML server** for sepsis detection instead of using simple heuristics
3. **Provide fallback mechanisms** when CSV files or ML server are unavailable

## Files Modified/Created

### Modified Files
- `sepsis_simpy/application/data_generator.py` - Completely rewritten

### New Files Created
1. `dataset/patient_data.csv` - Sample patient dataset
2. `ml_server.py` - Flask-based ML server for sepsis detection
3. `ml_requirements.txt` - Dependencies for ML server
4. `test_data_generator.py` - Test script to verify functionality
5. `start_ml_server.bat` - Windows batch script to start ML server

## Usage Instructions

### 1. CSV Dataset
The `HealthDataGenerator` class now reads from `dataset/patient_data.csv` by default.

**CSV Format Required:**
```csv
heart_rate,blood_oxygen,temperature,movement,sepsis_label
72,98.5,36.8,0,0
85,97.2,37.1,1,0
95,96.8,37.8,0,1
...
```

**Required Columns:**
- `heart_rate` (integer): Heart rate in BPM
- `blood_oxygen` (float): Blood oxygen saturation (%)
- `temperature` (float): Body temperature (°C)
- `movement` (integer): 0=idle, 1=active
- `sepsis_label` (integer): 0=no sepsis, 1=sepsis (used for ML training)

### 2. ML Server Setup

**Option A: Using the batch script (Windows)**
```bash
start_ml_server.bat
```

**Option B: Manual setup**
```bash
# Install dependencies
pip install -r ml_requirements.txt

# Start the ML server
python ml_server.py
```

The server will run on `http://localhost:5002` and provide these endpoints:
- `POST /predict_sepsis` - Make sepsis predictions
- `GET /health` - Health check
- `GET /model_info` - Model information

### 3. Integration with Simulation

The updated classes work seamlessly with your existing simulation:

```python
# Initialize with CSV data
generator = HealthDataGenerator()

# Initialize with ML server (falls back to heuristics if server unavailable)
patient_model = PatientStateModel()

# Generate health data from CSV
health_data = generator.generate_all()

# Get ML-based sepsis prediction
sepsis_risk = patient_model.calculate_sepsis_risk(health_data)
patient_state = patient_model.update_patient_state(health_data)
```

## API Contract for ML Server

### Request Format
```json
{
  "heart_rate": 85,
  "blood_oxygen": 97.2,
  "temperature": 37.1,
  "movement": 1
}
```

### Response Format
```json
{
  "prediction": 0,
  "status": "success",
  "input": {
    "heart_rate": 85,
    "blood_oxygen": 97.2,
    "temperature": 37.1,
    "movement": 1
  }
}
```

## Fallback Mechanisms

### 1. CSV File Fallback
If `dataset/patient_data.csv` is not found:
- Falls back to original synthetic data generation
- Prints warning message but continues operation

### 2. ML Server Fallback
If ML server is unavailable:
- Uses heuristic-based sepsis detection
- Simple rules: sepsis if 2+ conditions met:
  - Heart rate > 90 BPM
  - Blood oxygen < 95%
  - Temperature > 38°C or < 36°C

### 3. Error Handling
- Network timeouts (5 seconds)
- Invalid responses from ML server
- Malformed CSV data
- Missing required columns

## Testing

Run the test script to verify everything works:
```bash
python test_data_generator.py
```

This will test:
1. CSV data loading and generation
2. ML server integration (with fallback if server unavailable)
3. Integration between both components

## Performance Considerations

1. **CSV Reading**: Data is loaded once at initialization and cycled through
2. **API Calls**: 5-second timeout to prevent blocking
3. **Caching**: Last ML prediction is cached for fallback
4. **Memory**: CSV data is loaded into memory for fast access

## Customization

### Using Your Own Dataset
Replace `dataset/patient_data.csv` with your own data following the same format.

### Custom ML Server URL
```python
patient_model = PatientStateModel(ml_server_url="http://your-server:port")
```

### Custom CSV Path
```python
generator = HealthDataGenerator(csv_file_path="path/to/your/data.csv")
```

## Dependencies

### For Simulation (already in project)
- pandas
- numpy
- requests

### For ML Server
- flask
- pandas
- numpy
- scikit-learn
- joblib

Install ML server dependencies:
```bash
pip install -r ml_requirements.txt
```