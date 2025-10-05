### **Q1. “What was the CI part here? Because you mentioned that the DQN algorithm was implemented on your own?”**

**Answer:**
The DQN algorithm was implemented by our **Edge Computing Team**, since resource allocation at the edge is the core challenge we are addressing. The Computational Intelligence (CI) Team focused on building the **ML inference pipeline** for sepsis detection. In our workflow:

* **Edge Team →** handles offloading decisions and optimal resource allocation using DQN.
* **CI Team →** handles the training and deployment of ML models for sepsis detection, which run on the edge after allocation decisions are made.

---

### **Q2. “What exactly does the patient sensing layer process? Is it an edge layer? What computation is performed there?”**

**Answer:**
The **Patient Sensing Layer** acts as the **Thing Layer** in the *Thing → Edge → Cloud* hierarchy. Its role is primarily **data collection** from wearables and transmitting this data to the edge.

* **No computation or inference is performed at this layer.**
* **At the Edge Layer** → we perform resource allocation (via DQN), ML inference for sepsis detection, and trigger appropriate actions.

---

### **Q3. “Since you guys implemented the DQN algorithm, did you also explore or consider other possible implementation approaches?”**

**Answer:**
For this project, we **focused only on the DQN algorithm**, as it was the approach highlighted in the research paper we are following. We wanted to reproduce and validate the paper’s methodology first before branching out.
our aim was to stick with DQN.

---
