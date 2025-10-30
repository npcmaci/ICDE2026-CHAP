import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =====================
# Configuration
# =====================
class Config:
    num_vars = 5
    num_samples_per_env = 2500
    num_envs = 4
    noise_scale = 0.8
    mechanism = "nonlinear"
    seed = 42
    nonlinear_functions = ["sin", "tanh", "square", "exp_neg", "log_abs"]
    weight_variation = 0.1
    noise_variation = 0.15
    save_data = True
    save_dir = "../raw_data"
    plot_graph = True
    show_plot = False

# =====================
# Nonlinear functions
# =====================
def apply_nonlinear_function(x: np.ndarray, func_name: str) -> np.ndarray:
    """Apply nonlinear function."""
    if func_name == "sin":
        return np.sin(x)
    elif func_name == "tanh":
        return np.tanh(x)
    elif func_name == "square":
        return np.sign(x) * (x ** 2)
    elif func_name == "exp_neg":
        return np.exp(-np.abs(x))
    elif func_name == "log_abs":
        return np.sign(x) * np.log(1 + np.abs(x))
    else:
        return x

# =====================
# DAG generator
# =====================
class NumericalDAGGenerator:
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.seed)
        
        self.adj_matrix = self._generate_realistic_dag()
        self.graph = nx.from_numpy_array(self.adj_matrix, create_using=nx.DiGraph)
        
        self.weight_matrix = self._generate_weights()
        
        self.nonlinear_functions = self._assign_nonlinear_functions()
        
        self.env_params = self._generate_environment_params()
        
    def _generate_realistic_dag(self) -> np.ndarray:
        """Generate a hand-crafted DAG for n in {5,10}, else random."""
        n = self.config.num_vars
        adj_matrix = np.zeros((n, n))
        
        if n == 5:
            adj_matrix[0, 1] = 1
            adj_matrix[1, 2] = 1
            adj_matrix[2, 3] = 1
            adj_matrix[3, 4] = 1
            adj_matrix[0, 2] = 1
            adj_matrix[1, 3] = 1
            
        elif n == 10:
            adj_matrix[0, 2] = 1
            adj_matrix[0, 3] = 1
            adj_matrix[1, 4] = 1
            adj_matrix[1, 5] = 1
            adj_matrix[2, 4] = 1
            adj_matrix[3, 5] = 1
            adj_matrix[2, 6] = 1
            adj_matrix[3, 6] = 1
            adj_matrix[4, 7] = 1
            adj_matrix[5, 7] = 1
            adj_matrix[2, 8] = 1
            adj_matrix[5, 8] = 1
            adj_matrix[6, 9] = 1
            adj_matrix[7, 9] = 1
            adj_matrix[8, 9] = 1
            
        else:
            for i in range(1, n):
                num_parents = np.random.randint(1, min(i + 1, 4))
                parents = np.random.choice(i, size=num_parents, replace=False)
                adj_matrix[parents, i] = 1
        
        return adj_matrix
    
    def _generate_weights(self) -> np.ndarray:
        """Generate weights for existing edges."""
        n = self.config.num_vars
        weight_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if self.adj_matrix[i, j] == 1:
                    weight = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
                    weight_matrix[i, j] = weight
        
        return weight_matrix
    
    def _assign_nonlinear_functions(self) -> Dict[int, str]:
        """Assign a nonlinear function per node."""
        n = self.config.num_vars
        functions = {}
        
        for i in range(n):
            if self.config.mechanism == "linear":
                functions[i] = "linear"
            else:
                func = np.random.choice(self.config.nonlinear_functions)
                functions[i] = func
        
        return functions
    
    def _generate_environment_params(self) -> List[Dict]:
        """Generate per-environment multipliers."""
        env_params = []
        
        for env_idx in range(self.config.num_envs):
            env_param = {
                'weight_multiplier': 1.0 + np.random.uniform(
                    -self.config.weight_variation, 
                    self.config.weight_variation
                ),
                'noise_multiplier': 1.0 + np.random.uniform(
                    -self.config.noise_variation, 
                    self.config.noise_variation
                )
            }
            env_params.append(env_param)
        
        return env_params
    
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate data (all features continuous)."""
        n = self.config.num_vars
        total_samples = self.config.num_samples_per_env * self.config.num_envs
        
        data = np.zeros((total_samples, n))
        env_labels = np.zeros(total_samples, dtype=int)
        
        topo_order = list(nx.topological_sort(self.graph))
        
        sample_idx = 0
        
        for env_idx in range(self.config.num_envs):
            env_param = self.env_params[env_idx]
            
            for _ in range(self.config.num_samples_per_env):
                for var_idx in topo_order:
                    parents = np.where(self.adj_matrix[:, var_idx] == 1)[0]
                    
                    if len(parents) == 0:
                        value = np.random.normal(0, 1)
                    else:
                        parent_values = data[sample_idx, parents]
                        parent_weights = self.weight_matrix[parents, var_idx]
                        
                        parent_weights = parent_weights * env_param['weight_multiplier']
                        
                        linear_combination = np.sum(parent_values * parent_weights)
                        
                        if self.nonlinear_functions[var_idx] == "linear":
                            value = linear_combination
                        else:
                            value = apply_nonlinear_function(
                                linear_combination, 
                                self.nonlinear_functions[var_idx]
                            )
                        
                        noise_std = self.config.noise_scale * env_param['noise_multiplier']
                        noise = np.random.normal(0, noise_std)
                        value += noise
                    
                    data[sample_idx, var_idx] = value
                
                env_labels[sample_idx] = env_idx
                sample_idx += 1
        
        return data, self.adj_matrix, env_labels
    
    def plot_graph(self, save_path: Optional[str] = None):
        """Plot causal graph."""
        plt.figure(figsize=(12, 8))
        
        pos = nx.spring_layout(self.graph, k=3, iterations=50, seed=self.config.seed)
        
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color='lightblue', 
            node_size=1000,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            alpha=0.6
        )
        
        labels = {i: f'X{i}' for i in range(self.config.num_vars)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=12, font_weight='bold')
        
        edge_labels = {}
        for i in range(self.config.num_vars):
            for j in range(self.config.num_vars):
                if self.adj_matrix[i, j] == 1:
                    weight = self.weight_matrix[i, j]
                    edge_labels[(i, j)] = f'{weight:.2f}'
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
        
        plt.title(f'Numerical Causal DAG (n={self.config.num_vars}, mechanism={self.config.mechanism})', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved: {save_path}")
        
        if self.config.show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_data(self, data: np.ndarray, adj_matrix: np.ndarray, env_labels: np.ndarray, 
                  save_dir: str):
        """Save generated data and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        df_data = pd.DataFrame(data, columns=[f'X{i}' for i in range(self.config.num_vars)])
        df_data['env_label'] = env_labels
        
        data_path = os.path.join(save_dir, f'numerical_dag_data_{self.config.num_vars}vars.csv')
        df_data.to_csv(data_path, index=False)
        print(f"Data saved: {data_path}")
        
        adj_path = os.path.join(save_dir, f'numerical_dag_adj_{self.config.num_vars}vars.npy')
        np.save(adj_path, adj_matrix)
        print(f"Adj saved: {adj_path}")
        
        weight_path = os.path.join(save_dir, f'numerical_dag_weights_{self.config.num_vars}vars.npy')
        np.save(weight_path, self.weight_matrix)
        print(f"Weights saved: {weight_path}")
        
        func_path = os.path.join(save_dir, f'numerical_dag_functions_{self.config.num_vars}vars.txt')
        with open(func_path, 'w') as f:
            f.write("Node\tFunction\n")
            for node, func in self.nonlinear_functions.items():
                f.write(f"X{node}\t{func}\n")
        print(f"Functions saved: {func_path}")
        
        env_path = os.path.join(save_dir, f'numerical_dag_env_params_{self.config.num_vars}vars.txt')
        with open(env_path, 'w') as f:
            f.write("Environment\tWeight_Multiplier\tNoise_Multiplier\n")
            for i, param in enumerate(self.env_params):
                f.write(f"Env{i}\t{param['weight_multiplier']:.4f}\t{param['noise_multiplier']:.4f}\n")
        print(f"Envs saved: {env_path}")
        
        config_path = os.path.join(save_dir, f'numerical_dag_config_{self.config.num_vars}vars.txt')
        with open(config_path, 'w') as f:
            f.write("Configuration Parameters:\n")
            f.write("=" * 30 + "\n")
            f.write(f"num_vars: {self.config.num_vars}\n")
            f.write(f"num_samples_per_env: {self.config.num_samples_per_env}\n")
            f.write(f"num_envs: {self.config.num_envs}\n")
            f.write(f"noise_scale: {self.config.noise_scale}\n")
            f.write(f"mechanism: {self.config.mechanism}\n")
            f.write(f"seed: {self.config.seed}\n")
            f.write(f"weight_variation: {self.config.weight_variation}\n")
            f.write(f"noise_variation: {self.config.noise_variation}\n")
        print(f"Config saved: {config_path}")

# =====================
# Main
# =====================
def main():
    """Entry point."""
    print("=" * 60)
    print("Numerical DAG Data Generator")
    print("=" * 60)
    
    config = Config()
    
    print("Configuration:")
    print(f"  Number of variables: {config.num_vars}")
    print(f"  Samples per environment: {config.num_samples_per_env}")
    print(f"  Number of environments: {config.num_envs}")
    print(f"  Noise scale: {config.noise_scale}")
    print(f"  Mechanism: {config.mechanism}")
    print(f"  Random seed: {config.seed}")
    print("-" * 60)
    
    print("Generating data...")
    generator = NumericalDAGGenerator(config)
    data, adj_matrix, env_labels = generator.generate_data()
    
    print(f"Generated data shape: {data.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Number of edges: {np.sum(adj_matrix)}")
    print(f"Environment distribution: {np.bincount(env_labels)}")
    print("-" * 60)
    
    print("Nonlinear functions:")
    for node, func in generator.nonlinear_functions.items():
        print(f"  X{node}: {func}")
    print("-" * 60)
    
    print("Environment parameters:")
    for i, param in enumerate(generator.env_params):
        print(f"  Env{i}: weight_mult={param['weight_multiplier']:.3f}, "
              f"noise_mult={param['noise_multiplier']:.3f}")
    print("-" * 60)
    
    if config.plot_graph:
        print("Plotting graph...")
        graph_path = os.path.join(config.save_dir, f'numerical_dag_graph_{config.num_vars}vars.png')
        generator.plot_graph(graph_path)
    
    if config.save_data:
        print("Saving data...")
        generator.save_data(data, adj_matrix, env_labels, config.save_dir)
    
    print("=" * 60)
    print("Data generation completed!")
    print("=" * 60)
    
    return data, adj_matrix, env_labels, generator

if __name__ == "__main__":
    data, adj_matrix, env_labels, generator = main()