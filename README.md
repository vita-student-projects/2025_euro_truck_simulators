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

# Milestone 2: Perception-Aware Planning

Overview:
For the second milestone of the DLAV project, we extended our end-to-end deep learning model by incorporating **perception-based auxiliary tasks**: **semantic segmentation** and **depth estimation**. The goal was to enhance trajectory prediction by encouraging the model to learn more spatially meaningful representations through dense supervision.

### Architecture

This phase builds upon our previous encoder-decoder model, with the following additions:

* **Shared Visual Encoder**:

  * Based on ResNet18 pretrained on ImageNet.
  * Outputs both a **reduced feature map** (used for depth and semantic decoders) and a **global visual feature vector** (used for trajectory prediction).
  * Includes a `1×1` convolution to reduce channels from 512 to 256 for efficient decoding.

* **Auxiliary Decoders**:

  * **Depth Decoder**:

    * Takes the reduced feature map and predicts a dense depth map.
    * Output resolution: 200 × 300.
    * Loss: Smooth L1 Loss.
  * **Semantic Decoder**:

    * Also operates on the reduced feature map.
    * Predicts per-pixel class logits for 14 semantic classes.
    * Loss: CrossEntropyLoss.

* **Trajectory Head**:

  * Remains the same as Phase 1.
  * Uses the global visual features and history encoding for multimodal trajectory prediction and confidence scoring.

The outputs from the semantic and depth decoders are not fed into the trajectory decoder directly, but their supervision shapes the shared encoder's representations, improving the learned features.

### Training Approach

* **Multi-task Loss**:
  The total training loss is a weighted sum of trajectory, depth, and semantic segmentation losses:

  ```python
  loss = traj_loss + depth_k * depth_loss + semantic_k * semantic_loss
  ```

  * We explored multiple combinations and found that including semantic supervision helped generalization.
  * Our best performing configuration used `depth_k = 0` and `semantic_k = 0.33`.

* **Curriculum Learning**:

  * We gradually increased the trajectory prediction length over the first 10 epochs.

* **Hyperparameter Tuning**:

  * We continued using Optuna to tune learning rate, weight decay, number of modes, and auxiliary task weights.
  * Based on validation ADE and FDE, we selected the best model configuration.

* **Evaluation Metrics**:

  * ADE and FDE were monitored for trajectory prediction performance.
  * Additional logging included validation loss for semantic and depth decoders.

### Results

We observed that semantic supervision consistently helped reduce overfitting, while depth supervision had mixed results. In our final model, we disabled depth loss to achieve better generalization.

| Metric               | Value    |
| -------------------- | -------- |
| Validation ADE       | 1.165     |
| Kaggle ADE           | 1.609    |
| Number of Modes      | 6        |
| Backbone             | ResNet18 |
| Depth Loss Weight    | 0.0      |
| Semantic Loss Weight | 0.33     |

### Summary

* We successfully incorporated auxiliary perception tasks to improve the model's trajectory planning.
* Semantic supervision improved generalization and yielded better validation and test ADE scores.
* Our final model nearly meets the milestone requirement with a **Kaggle ADE of 1.609**, earning full score for this milestone.

[Link for weights](https://drive.google.com/file/d/1HTYCHOZa4ii3Ju8a4pPswd68b9mXXgWx/view?usp=drive_link)
