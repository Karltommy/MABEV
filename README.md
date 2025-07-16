# MABEV: Motion-Aware BEVFormer
**"Enhancing BEVFormer with Motion-Aware Query Fusion"**

Bird’s Eye View (BEV) perception has become a cornerstone technique in autonomous driving, enabling spatially consistent scene understanding from multi-view cameras. BEVFormer is a representative approach that leverages transformer-based temporal and spatial attention to project multi-view image features into the BEV space. While it performs well in static environments, its effectiveness in dynamic scenes remains limited due to several factors:

- The temporal attention mechanism does not explicitly model the motion discrepancy across frames;
- The fusion between current and previous BEV queries is structure-agnostic, which leads to underutilization of temporal motion cues;
- Static and dynamic objects are modeled uniformly, lacking targeted motion-aware refinement.

To address these limitations, we propose MABEV (Motion-Aware BEVFormer) — a lightweight yet effective enhancement to BEVFormer. MABEV introduces an explicit delta query mechanism to capture the motion-induced differences between temporally aligned BEV queries to generate motion-aware queries for downstream attention.

## Highlights
-  Motion-aware delta query module to enhance dynamic object perception
-  Improved NDS and mAP on NuScenes
-  Compatible with BEVFormer and MMDetection3D framework

##  Overview
- Overview of MABEV Encoder with Motion-Aware Query Fusion

![MABEV_Overview](figs/MABEV_Overview.png "model overview")

- Our proposed MABEV enhances the BEVFormer encoder by introducing a motion-aware delta attention module.
- Given multi-view image features and a historical BEV query, we first apply temporal self-attention to align the past BEV features with the current frame. The aligned previous BEV query is then compared to the current BEV query to compute a delta query, representing motion-induced differences. This delta query is fused with the current query via an MLP-based fusion module to generate a motion-aware query, which is subsequently refined through motion-aware attention.
- This three-stage attention pipeline — temporal → spatial → motion-aware — effectively improves the perception of dynamic objects while maintaining compatibility with the original BEVFormer framework.

##  Experiment

-  Configurations -- a. Samples per GPU: 4; b.Total Batch Size: 4; c. Optimizer: AdamW, lr = 1.5e-4
-  Hardware -- GPU: NVIDIA RTX 3090 (24GB)
-  Performance on NuScenes Full
  
| Method | Backbone | Lr Schd	| NDS	| mAP	| Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MABEV-tiny (Ours)|R50|24ep|38.17|28.39|[ma_bevformer_tiny.py](projects\configs\bevformer\ma_bevformer_tiny.py)|[model](https://github.com/Karltommy/MABEV_FILE/releases/download/V1.0.0/mabev_tiny_epoch_24.pth)/[log](https://github.com/Karltommy/MABEV_FILE/releases/download/V1.0.0/mabev_tiny_epoch_24.log)|
| BEVFormer-tiny|R50|24ep|37.91|27.72|[bevformer_tiny.py](projects\configs\bevformer\bevformer_tiny.py)|[model](https://github.com/Karltommy/MABEV_FILE/releases/download/V1.0.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/Karltommy/MABEV_FILE/releases/download/V1.0.0/bevformer_tiny_epoch_24.log)|

- We compare our model with the official BEVFormer-Tiny, which were both trained under the same conditions for 24 epochs on the nuScenes full dataset without loading pretrained models.
- Our model achieves +0.67% mAP and +0.26% NDS improvement, showing the effectiveness of the proposed modification.
- Due to limited computing resources (single RTX 3090), we only conduct experiments on the Tiny version. Future work will extend to the Base model.

<!-- Our MABEV model shows consistent improvements across all metrics, especially in mAP (+%) and NDS (+%), demonstrating enhanced capability for dynamic object modeling.-->

# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

## Acknowledgement
We appreciate this excellent open source project:
[BEVFormer](https://github.com/fundamentalvision/BEVFormer)



