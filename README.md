### How to train MLP

1. Configure the initialisation settings in *config.json*
2. Run ```python3 train.py config.json``` in command line in DeepLearningA1NE directory


### How to test MLP on test dataset of 10000 observations
1. Run ```python3 test.py```

### Where to find Modules
- Hidden Layers (HiddenLayer.py)
- Activation functions (Activation.py)
- Weight Decay (MLP.py update() Line 85)
- Momentum (Optimiser.py Momentum class)
- Dropout (HiddenLayer.py Line 66 - 68)
- Softmax (Activation.py softmax() Line 23)
- Cross Entropy Loss (MLP.py criterion_CrossEL() or criterion_CrossEL_Batch())
- Mini Batch Training (MiniBatch.py)
- Batch Normalisation (HiddenLayer.py batch_norm_forward() Line 112)
- Adam Optimiser (Optimiser.py AdamOptimiser class)
- Early Stopping (MLP.py EarlyStopping class Line 197)