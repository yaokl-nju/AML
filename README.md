README for codes of AML (ACM TKDD 2024)

### Depencies
    1. torch version 1.8.0+cu11.1
    2. torch_geometric version 1.6.3
    3. torch_sparse version 0.6.10
    4. torch_scatter version 2.0.7
    5. torch_cluster version 1.5.9
    6. ogb version 1.3.1
    7. Others that are missed. Readers can install them by yourselves.


### A1. Structures of this directory.
    AML directory contains codes of AML for ogbl-collab, ogbl-ppa and ogbl-citation2.
    MF directory contains codes of MF to generate new node features for ogbl-ppa.


### A2. Download benchmark datasets.
Readers need to download the datasets by yourselves. Then you need to set the follow variable to run the codes:
            
    root: your_datapath
    in config.yaml of directory AML/conf and directory MF/conf.

### A3. Generate new node features for ogbl-ppa.
Before generating results on ogbl-ppa, first get in MF-ogbl and then run the following command to generate new node features.
		
    python main_ogb.py

### A4. Generate results in Tables.
On ogbl-collab, change settings in config.yaml of directory AML/conf.
		
    dataset: ogbl-collab
    model: SAGE			# SAGE and GAT
    hidden_dim: 256
    method: BNS
    method_te: FULL
		

On ogbl-ppa, change settings in config.yaml of directory AML/conf.

    dataset: ogbl-ppa
    model: SAGE			# SAGE and GAT
    hidden_dim: 256
    method: BNS
    method_te: FULL

On ogbl-citation2, change settings in config.yaml of directory AML/conf.

    dataset: ogbl-citation2
    model: SAGE			# SAGE and GAT
    hidden_dim: 256
    method: BNS
    method_te: FULL
 
To generate results for GAE with SAGE and GAT, reset the following variables in config/config.yaml:

    strategy: symmetric

To generate results for AML, reset the following variables in config/config.yaml:
    	
    strategy: asymmetric

Other variables remain unchanged.

Run the following commands can generate results for each setting.
    	
    python main_ogb.py

  
### Citation
<pre>
@article{DBLP:journals/tkdd/YaoL24,
  author       = {Kai{-}Lang Yao and Wu{-}Jun Li},
  title        = {Asymmetric Learning for Graph Neural Network based Link Prediction},
  journal      = {{ACM} Transactions on Knowledge Discovery from Data},
  volume       = {18},
  number       = {5},
  pages        = {106:1--106:18},
  year         = {2024},
}
</pre>
