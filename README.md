# GraphTPA: Parameter-Efficient Tensor Product Attention for Network Intrusion Detection

This repository contains the official PyTorch implementation of **GraphTPA**, a parameter-efficient graph neural network architecture designed for real-time intrusion detection in resource-constrained IIoT environments.

By adapting Tensor Product Attention (TPA) to directed flow-graphs, GraphTPA reduces the $\mathcal{O}(d^2)$ memory complexity of full-rank bilinear attention down to $\mathcal{O}(dr)$. It achieves state-of-the-art detection performance on multiple challenging benchmarks while requiring **2.3× fewer parameters** and executing **2.1× faster** than leading baselines.

---

## 🚀 Key Features

* **Tensor Product Attention Convolution (TPAConv):** Factorizes Q/K/V projections into low-rank components, explicitly encoding edge attributes with minimal parameter overhead.
* **Low-Rank Tensor Edge Representation:** Replaces standard concatenation with a CP-decomposed bilinear tensor product, cleanly capturing multiplicative endpoint interactions vital for isolating structurally obscured attacks (e.g., Man-in-the-Middle).
* **Hierarchical Rank Adaptation:** Progressively restricts feature subspace ranks across deep layers, preventing over-parameterization and acting as a powerful structural regularizer on highly imbalanced datasets.

## 🛠️ Environment Setup

We recommend using `conda` to manage your environment. All dependencies, including PyTorch and PyTorch Geometric, are specified in the requirements file.

```bash
# Clone the repository
git clone https://github.com/Kkyn-ltcode/GraphTPA-Parameter-Efficient-Tensor-Product-Attention-for-Network-Intrusion-Detection.git
cd GraphTPA-Parameter-Efficient-Tensor-Product-Attention-for-Network-Intrusion-Detection

# Create environment and install dependencies
conda create -n graphtpa python=3.10
conda activate graphtpa
pip install -r requirements.txt
```

## 📊 Datasets

GraphTPA was extensively evaluated on five benchmarks:
1. `NF-BoT-IoT-v2`
2. `NF-ToN-IoT`
3. `NF-CSE-CIC-IDS2018-v3`
4. `NF-UNSW-NB15`
5. `NF-UQ-NIDS`

### Data Preparation
You can download the raw NetFlow datasets from the official UQ eSpace repository:
[UQ eSpace NF Datasets](https://espace.library.uq.edu.au/records/search?searchQueryParams%5Ball%5D=NF&page=1&pageSize=20&sortBy=score&sortDirection=Desc)

Once downloaded, place the `.csv` files into their respective directories under `data/`. **Important:** The `.csv` file name must exactly match the folder that contains it. For example, place the `NF-ToN-IoT.csv` file inside the `data/NF-ToN-IoT/` directory.

To build the PyTorch Geometric graphs and generate node/edge features, run the builder script for your desired dataset:

```bash
# Example: Building the graph for NF-ToN-IoT
python data/traffic_graph_builder.py --config_path data/config.json --dataset NF-ToN-IoT
```

This will automatically extract the features, generate `.parquet` files, and handle `id2label` mappings for both binary and multi-class settings.

## 🧠 Training & Evaluation

The training script natively supports PyTorch Distributed Data Parallel (DDP) for multi-GPU training, Mixed Precision (AMP), and gradient accumulation.

To train the GraphTPA model on a specific dataset:

```bash
# Example: Training on NF-ToN-IoT using the default multi-class configuration
python graph_transformers_traffic.py --config_path configs/graph_config_traffic.json --dataset NF-ToN-IoT
```

**Key Configuration Options (`configs/graph_config_traffic.json`):**
* `use_tpa`: Enable/Disable Tensor Product Attention (TPAConv).
* `use_tensor_edge`: Enable/Disable the bilinear tensor edge representation block.
* `tensor_edge_mode`: Set to `"hybrid"` to dynamically gate between concatenated and tensor pathways.
* `rank_adaptation`: Set to `"hierarchical"` to progressively decay rank across deeper layers.

## 📝 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{nguyen2026graphtpa,
  title={GraphTPA: Parameter-Efficient Tensor Product Attention for Network Intrusion Detection},
  author={Nguyen, Luu and others},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026}
}
```
*(Note: Citation will be updated upon publication).*
