import config

class Network:
    """Manages network connections and simulates communication latency."""
    def __init__(self, env):
        self.env = env

    def send_data(self, source, destination, data):
        """Simulates sending data from a source to a destination with latency."""
        latency = self._get_latency(source, destination)
        yield self.env.timeout(latency / 1000.0) # Convert ms to seconds
        # In a real scenario, you'd also pass the data object.
        # For Iteration 1, the timeout is sufficient.

    def _get_latency(self, source_type, destination_type):
        """Returns latency based on the types of the source and destination devices."""
        if source_type == 'wearable' and destination_type == 'edge':
            return config.LATENCY_WEARABLE_TO_EDGE
        elif source_type == 'edge' and destination_type == 'cloud':
            return config.LATENCY_EDGE_TO_CLOUD
        else:
            return 2 # Default small latency for other connections