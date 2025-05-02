# DLAV Phase 1: End-to-End Trajectory Planner

Students: Mert Ertörer, Syrine Noamen
Group Name: euro_truck_simulators

Score on [Kaggle Competition link](https://www.kaggle.com/competitions/phase1/overview): 1.57 for the metric ADE

Course: Deep Learning for Autonomous Vehicles

# Milestone 1: End-to-End Planning

Overview: This project implements an end-to-end deep learning model for the final project of the course DLAV at EPFL in 2025. It is use for predicting future vehicle trajectories using:
- RGB camera input
- Past motion history

We decided to not use the driving command input as it was sometimes very noisy and does not matching the image and position of the ego vehicle.



The model is designed to predict the future trajectory of an ego (self-driving) vehicle based on camera inputs and historical movement data. It supports multimodal predictions, generating multiple possible future trajectories with associated confidence scores to handle the inherent uncertainty in driving scenarios
### Architecture
The model follows an encoder-decoder architecture with the following major components:

- Visual Encoder:

    - ResNet18 pretrained backbone (truncated before global pooling and fully connected layers)
    - Adaptive average pooling to reduce spatial dimensions to a single vector
    - Fully connected layer to project to 256-dimensional visual features


- History Encoder:
    - Processes the vehicle's historical trajectory data
    - Flattens the historical trajectory and passes it through a 2-layer MLP
    - Projects the historical data to 128-dimensional feature space


- Fusion Layer:

    - Concatenates the visual (256-dim) and historical (128-dim) features
    - Processes the combined features through a 2-layer MLP
    - Produces a 512-dimensional fused representation


- Trajectory Decoder:

    - A single fully connected layer that projects the fused features to multiple trajectory predictions
    - Outputs either [Batch, Modes, Steps, 2] or [Batch, Modes, Steps, 3] depending on whether heading is included
    - Default configuration: 6 modes, 60 future time steps

- Confidence Head:

    - A linear layer that projects the fused features to confidence scores
    - Softmax activation to ensure scores sum to 1.0 across modes


### Training Approach
- Loss Function: Smooth L1 Loss weighted by confidence scores across multiple trajectory modes
- Optimizer: Adam optimizer with configurable learning rate (default: 1e-4) and weight decay (default: 2e-5)
- Learning Rate Schedule: ReduceLROnPlateau scheduler that reduces learning rate when validation metrics plateau
- Data Augmentation: Supports horizontal flipping of camera images with corresponding sign flips in the trajectory data (x-coordinates and heading). This creates mirrored driving scenarios to improve generalization.
- Validation Metrics: Tracks ADE and FDE to evaluate prediction accuracy


It was implemented using PyTorch and PyTorch Lightning for structured training. It includes a Lightning module that handles training, validation, and testing loops, supports mixed precision training for improved performance and includes checkpointing to save the best models based on validation metric.


For logging, we used TensorBoard and plotted the training and validation losses and metrics, as well as montiored the learning rate. 

**Other experiments**
We have experimented with an alternative RNN-based architecture that uses a GRU for history encoding and trajectory decoding. Despite the more sophisticated sequence modeling capabilities, this alternative approach didn't necessarily improve performance compared to the MLP-based model described above. The RNN version used a different loss formulation with winner-takes-all trajectory loss combined with a confidence loss term.


We also experimented with a model that takes in the heading (the third dimension of the future trajectory) as an additional input. However, this approach did not necessarily improve performance compared to the model without heading input. 


For hyperparameter tuning, we used Optuna for systematic hyperparameter optimization, conducting multiple trials to find a good configuration. The tuning process explored various hyperparameters including learning rate (1e-5 to 1e-3), weight decay (1e-6 to 1e-3), number of modes (3 to 8), and scheduler parameters. Each configuration was evaluated based on validation ADE/FDE metrics. This hyper parameter tuning led to our final model with 4 modes and weight decay of 2e-6, which provided the best balance between prediction accuracy and computational efficiency.

[Link for weights](https://drive.google.com/file/d/1xoso3Dfy2v38mVxGpaWxlEk4boXm8kv2/view)
