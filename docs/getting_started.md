# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MABEVFormer with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/ma_bevformer_tiny.py 4
```

Eval MABEVFormer with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/ma_bevformer_base.py ./path/to/ckpts.pth 4
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)
