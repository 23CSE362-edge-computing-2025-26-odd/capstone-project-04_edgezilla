import os
import json
import logging

class DataExporter:
    """Saves simulation results to various file formats."""
    def __init__(self, dataframes, output_dir='results/data'):
        self.dfs = dataframes
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_to_csv(self, strategy_name):
        """Exports all dataframes to CSV files."""
        for name, df in self.dfs.items():
            if not df.empty:
                df.to_csv(os.path.join(self.output_dir, f'{strategy_name}_{name}.csv'), index=False)
        self.logger.info(f"Data for '{strategy_name}' exported to CSV in {self.output_dir}")

    def generate_summary_report(self, strategy_name):
        """Generates a text file with a summary of the simulation results."""
        report_path = os.path.join(self.output_dir, f'{strategy_name}_summary_report.txt')
        perf_df = self.dfs.get('performance')
        
        with open(report_path, 'w') as f:
            f.write(f"--- Simulation Summary Report for Strategy: {strategy_name} ---\n\n")
            
            if perf_df is not None and not perf_df.empty:
                avg_latency = perf_df['latency'].mean()
                max_latency = perf_df['latency'].max()
                total_tasks = len(perf_df)
                
                f.write("Performance Metrics:\n")
                f.write(f"  - Total Tasks Processed: {total_tasks}\n")
                f.write(f"  - Average End-to-End Latency: {avg_latency:.4f} seconds\n")
                f.write(f"  - Maximum Latency: {max_latency:.4f} seconds\n\n")
                
                # Latency by location
                latency_by_loc = perf_df.groupby('processing_location')['latency'].mean()
                f.write("Average Latency by Location:\n")
                for loc, lat in latency_by_loc.items():
                    f.write(f"  - {loc.capitalize()}: {lat:.4f} seconds\n")

            self.logger.info(f"Summary report for '{strategy_name}' generated at {report_path}")