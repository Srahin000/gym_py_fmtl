"""
CH (Cluster Head) Simulation Module
Simplified communication simulator for hierarchical FMTL
"""

class CHCommunicationSimulator:
    """
    Simulates communication between members, CHs, and global aggregator
    Tracks bandwidth and latency for hierarchical federation
    """
    
    def __init__(self, num_clusters: int = 3, members_per_cluster: int = 200):
        self.num_clusters = num_clusters
        self.members_per_cluster = members_per_cluster
        self.total_members = num_clusters * members_per_cluster
        
        # Network parameters
        self.model_size_kb = 246.42
        self.bandwidth_mbps = 100  # Mbps
        
        # CH assignments
        self.cluster_heads = {i: f"CH{i}" for i in range(num_clusters)}
        self.global_aggregator = "CH1"  # CH1 acts as CH*
        
        print(f"✓ CH Communication Simulator initialized")
        print(f"  Clusters: {num_clusters}")
        print(f"  Members per cluster: {members_per_cluster}")
        print(f"  Total UAVs: {self.total_members}")
        print(f"  Global aggregator: {self.global_aggregator}")
    
    def calculate_transmission_time(self, data_size_kb: float) -> float:
        """
        Calculate transmission time for data transfer
        
        Args:
            data_size_kb: Data size in KB
        
        Returns:
            Time in seconds
        """
        data_size_mb = data_size_kb / 1024
        time_seconds = (data_size_mb * 8) / self.bandwidth_mbps
        return time_seconds
    
    def simulate_round(self, round_num: int, participating_clusters: list):
        """
        Simulate one federated learning round
        
        Args:
            round_num: Current round number
            participating_clusters: List of cluster IDs participating
        
        Returns:
            Dict with timing and bandwidth statistics
        """
        stats = {
            'round': round_num,
            'phases': {},
            'total_time': 0,
            'total_data_kb': 0,
        }
        
        # Phase 1: Members → CH (parallel within clusters)
        members_to_ch_time = self.calculate_transmission_time(self.model_size_kb)
        members_to_ch_data = len(participating_clusters) * self.members_per_cluster * self.model_size_kb
        
        stats['phases']['members_to_ch'] = {
            'time': members_to_ch_time,
            'data_kb': members_to_ch_data,
        }
        stats['total_time'] += members_to_ch_time
        stats['total_data_kb'] += members_to_ch_data
        
        # Phase 2: CH → CH* (sequential)
        ch_to_global_time = len(participating_clusters) * self.calculate_transmission_time(self.model_size_kb)
        ch_to_global_data = len(participating_clusters) * self.model_size_kb
        
        stats['phases']['ch_to_global'] = {
            'time': ch_to_global_time,
            'data_kb': ch_to_global_data,
        }
        stats['total_time'] += ch_to_global_time
        stats['total_data_kb'] += ch_to_global_data
        
        # Phase 3: CH* → CH (broadcast)
        global_to_ch_time = self.calculate_transmission_time(self.model_size_kb)
        global_to_ch_data = len(participating_clusters) * self.model_size_kb
        
        stats['phases']['global_to_ch'] = {
            'time': global_to_ch_time,
            'data_kb': global_to_ch_data,
        }
        stats['total_time'] += global_to_ch_time
        stats['total_data_kb'] += global_to_ch_data
        
        # Phase 4: CH → Members (parallel within clusters)
        ch_to_members_time = self.calculate_transmission_time(self.model_size_kb)
        ch_to_members_data = len(participating_clusters) * self.members_per_cluster * self.model_size_kb
        
        stats['phases']['ch_to_members'] = {
            'time': ch_to_members_time,
            'data_kb': ch_to_members_data,
        }
        stats['total_time'] += ch_to_members_time
        stats['total_data_kb'] += ch_to_members_data
        
        return stats
    
    def print_stats(self, stats: dict):
        """Print round statistics"""
        print(f"\nRound {stats['round']} Statistics:")
        print(f"  Total time: {stats['total_time']:.3f} seconds")
        print(f"  Total data: {stats['total_data_kb']/1024:.2f} MB")
        print(f"  Phases:")
        for phase_name, phase_stats in stats['phases'].items():
            print(f"    {phase_name}: {phase_stats['time']:.3f}s, {phase_stats['data_kb']/1024:.2f} MB")


if __name__ == '__main__':
    # Example usage
    simulator = CHCommunicationSimulator(num_clusters=3, members_per_cluster=200)
    
    # Normal round (all clusters)
    stats_normal = simulator.simulate_round(round_num=100, participating_clusters=[0, 1, 2])
    simulator.print_stats(stats_normal)
    
    # D&R-E round (Cluster 0 excluded)
    stats_dre = simulator.simulate_round(round_num=112, participating_clusters=[1, 2])
    simulator.print_stats(stats_dre)
