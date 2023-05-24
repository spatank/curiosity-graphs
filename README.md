# Exploring Graph-Structured Data Using Human Curiosity

## High Level
When sources of information have a natural graph structure, two theories of human curiosity offer graph-theoretic formulations of intrinsic motivations that underlie exploration. We can represent the current state of knowledge as the subgraph induced by the visited nodes in the environment. Information gap theory (IGT) argues that curiosity aims to regulate the number of information gaps---formalized as topological cavities---in the current state of knowledge. Compression progress theory (CPT) argues that information acquisition seeks to build more compressible state representations, formalized as network compressibility. In this work, we adopt the two theories to design reward functions for GNN-based agents exploring graph-structured data. We train the agents to explore graph environments using reinforcement learning while optimizing for the number of information gaps or network compressibility.

## Code Overview

The 'GraphRL/' folder contains the following files:
1) 'environment.py': defines a graph environment and handles agent-environment interactions during simulations.
2) 'agents_baseline.py': implements the maximum degree, minimum degree and random baseline agents.
3) 'agent_GNN.py': implements the GNN agents.
4) 'helpers_simulation.py': contains code to simulate the agent-environment interaction and a function consisting of the training loop.
4) 'helpers_rewards.py': contains functions to compute the number of topological cavities and the compressibility of networks.
5) 'helpers_miscellaneous.py': defines helper functions and classes useful for training and performance evaluation.


### Training for Synthetic Environments
1) Build synthetic graph environments using the notebook located at 'Notebooks/data_processing_synthetic.ipynb.' Generated networks are placed inside the 'Environments/' folder.
2) In 'train_GNN_synthetic.py', specify the parameters for the training process:
      a) run: creates a folder in 'Runs/' to store training and validation curves and model checkpoints specific to the current training run.
      b) network_type: specify the model used to build synthetic graph environments; 'synthetic_ER', 'synthetic_BA', 'synthetic_RG', or 'synthetic_WS'.
      c) size: specify the size of the training, validation, and testing datasets; options include 'mini', 'small', 'medium', or 'large'; note that the datasets must first be created according to Step 1.
      d) feature_mode: specify the node attributes; we use the local degree profile, 'LDP', for each node; options include 'random' and 'constant'.
      e) reward_function: specify which exploration objective the GNN is being trained for, 'betti_numbers' for IGT and 'compressibility' for CPT; any function handle that accepts a network as input and outputs a scalar is a valid specification.
3) After defining the above parameters, a complete training run can be accomplished by running the 'train_GNN.py' script.


We use the DQN algorithm with a replay buffer and a target network to train the GNNs. Hyperparameters for training can be altered inside the 'GraphRL/helpers_miscellaneous.py' file. Algorithmic details about the training process are outlined in the 'Supplement.pdf' file in this submission. 

### Baselines
Baselines can be evaluated using the 'evaluate_baselines_synthetic.py' script. Results are stored inside the 'Baselines/' folder. 

### Plotting
To plot the GNN agents' performance alongside the baselines, use the 'plotting_synthetic.py' file. Make sure to specify the plotting parameters corresponding to the training run for which you wish to generate figures. 

### Training for Real-World Graphs
Real-world graph datasets can be used as environments for training and testing. First, create an adjacency matrix corresponding to the network you wish to train on and place it inside a folder. Then edit the 'train_GNN_real.py' file as necessary before running it to achieve a full training run. Baseline results and performance plots can be generated using 'evaluate_baselines_real.py' and 'plotting_real.py', respectively.


### Personalized Curiosity-Based Centrality
Self-contained notebooks to compute curiosity-based centrality results for each of the three real-world graph datasets studied in the Main manuscript are included in corresponding folder.
