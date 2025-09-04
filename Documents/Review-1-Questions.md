# Review 1 Presentation Questions

## 1. Where is the edge layer? 

The edge layer consists of Edge Hosts. In our sepsis detection use case, these are local servers physically located within the hospital, such as in an ICU or a specific ward.

Their primary role is to collect real-time data directly from patient wearables and sensors, including temperature sensors, heart rate monitors, and others, which constantly upload tasks to be processed. Once the Edge Host receives this data, it's responsible for pre-processing it, running the local ML model for sepsis detection, and using its own "Local Resource Allocation Algo" to decide whether to handle the task itself or offload it to the cloud server.

## 2. Why is privacy necessary?
Privacy is critical because the system processes highly sensitive 
Protected Health Information (PHI) like patient vital signs. Traditional methods are vulnerable to gradient inversion attacks, where private patient data can be reconstructed from shared model updates. Our CFRL approach is more secure because it only shares non-sensitive model outputs and rewards, not the vulnerable model gradients themselves.

## 3. Is this personalized ML for each patient?
No, the model is not personalized for each patient. Instead, it is specialized for each hospital unit. Each edge host (e.g., an ICU server) trains a local model that adapts to the specific patient population and task patterns of that unit. This provides localized adaptation for different hospital environments rather than individual patient models
	
## 4. What tasks does the edge do?
The edge host acts as a local manager for sepsis detection tasks. Its primary jobs are to:
 - Receive and queue real-time tasks from patient monitors.
 - Decide whether to process a task locally using its own resources or offload it to the central server if it is overloaded.
 - Share the output of its local strategy (not private data) with the server to help optimize the entire system's resource allocation
 
## 5. Have you explained the DQN algos?
Yes, the system uses two distinct Deep Q-Network (DQN) algorithms that learn simultaneously:

Edge Host DQN (Algorithm 1): This algorithm learns the best local strategy for an edge host to either schedule a sepsis detection task locally or offload it. It bases its decision on its current resource status and the queue of incoming tasks.

Cloud DQN (Algorithm 2): This algorithm learns how to best divide and reserve central resources for all the edge hosts. It bases its decision on the strategy outputs it receives from all the hosts, without seeing any private data.
