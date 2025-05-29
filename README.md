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

* **Other Experiments**:

    We explored several variants and ablation studies to understand the role of auxiliary tasks:
    
    * **Depth Masking (Object-Prioritized Loss):**
    
      * We applied a spatial mask to prioritize depth loss for foreground objects (e.g., vehicles, pedestrians) over background regions (e.g., sky, ground).
      * However, this **slightly worsened** overall performance, possibly due to reducing the effective supervision signal.
    
    * **Gradual Decrease in Auxiliary Weights:**
    
      * We experimented with decaying the auxiliary task weights (`depth_k`, `semantic_k`) over time to shift learning focus to the main task.
      * This **did not improve** generalization and led to unstable training.
    
    * **Stronger Backbone – ResNet34:**
    
      * Replacing ResNet18 with ResNet34 in the shared encoder increased model capacity.
      * However, this resulted in **overfitting** and worse validation ADE, even with regularization and dropout.
  
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

* **Other Experiments**:

    We explored several variants and ablation studies to understand the role of auxiliary tasks:
    
    * **Depth Masking (Object-Prioritized Loss):**
    
      * We applied a spatial mask to prioritize depth loss for foreground objects (e.g., vehicles, pedestrians) over background regions (e.g., sky, ground).
      * However, this **slightly worsened** overall performance, possibly due to reducing the effective supervision signal.
    
    * **Gradual Decrease in Auxiliary Weights:**
    
      * We experimented with decaying the auxiliary task weights (`depth_k`, `semantic_k`) over time to shift learning focus to the main task.
      * This **did not improve** generalization and led to unstable training.
    
    * **Stronger Backbone – ResNet34:**
    
      * Replacing ResNet18 with ResNet34 in the shared encoder increased model capacity.
      * However, this resulted in **overfitting** and worse validation ADE, even with regularization and dropout.
  
### Results

We observed that semantic supervision consistently helped reduce overfitting, while depth supervision had mixed results. In our final model, we disabled depth loss to achieve better generalization.

| Metric               | Value    |
| -------------------- | -------- |
| Validation ADE       | 1.165     |
| Kaggle ADE           | 1.609    |
| Number of Modes      | 6        |
| Backbone             | ResNet18 |
| Depth Loss Weight    | 0.1      |
| Semantic Loss Weight | 0.33     |

### Summary

* We successfully incorporated auxiliary perception tasks to improve the model's trajectory planning.
* Semantic supervision improved generalization and yielded better validation and test ADE scores.
* Our final model nearly meets the milestone requirement with a **Kaggle ADE of 1.609**, earning full score for this milestone.

[Link for weights](https://drive.google.com/file/d/1HTYCHOZa4ii3Ju8a4pPswd68b9mXXgWx/view?usp=drive_link)


### Other things we have tried
- Dynamic weighting to the auxilary tasks where the weights for the semantic segmentation and depth estimation start high, then decrease gradually after 40% of the training.
- Custom losses for depth where we used a more forgiving depth loss that focuses on object shapes rather than precise values. It significantly reduces the weight on absolute depth accuracy while maintaining edge detection.
- Using the pretrained vision encoder and history encoder from milestone one.
---

# Milestone 3: Real-World-Aware Multi-Task Planning
File train_milstone3.py is the training script for this milestone.

### Overview:

In the third milestone of the DLAV project, we addressed the **domain gap** between synthetic and real-world data by integrating **real-world driving scenes** into both training and evaluation. While our model from previous milestones performed well on synthetic data, it struggled to generalize to real images. To overcome this, we:

* Incorporated **real-world validation data** into the training pipeline.
* Applied extensive **data augmentations** to both synthetic and real datasets.
* Generated **semantic labels** for real-world data using pretrained models, allowing us to **retain auxiliary supervision**.

These enhancements significantly boosted our model’s performance on real-world inputs.

---

### Architecture

We continued using our encoder-decoder architecture from Milestone 2 with the following maintained and extended components:

* **Shared Visual Encoder**:

  * Based on ResNet18, pretrained on ImageNet.
  * Produces both global visual embeddings (for trajectory prediction) and spatial feature maps (for semantic segmentation).

* **Auxiliary Decoder**:

  * **Semantic Decoder**:

    * Operates on spatial visual feature maps.
    * Predicts per-pixel class logits for 15 semantic classes.
    * Loss: CrossEntropyLoss.
    * Newly added: Real-world semantic labels using pretrained models.

* **Trajectory Head**:

  * Predicts multiple future trajectory modes (4 modes × 60 time steps).
  * Uses fused visual and history features for multimodal prediction.
  * Outputs confidence scores per mode (softmax-normalized).

---

### Training Approach

* **Data Augmentation**:

  * Applied incrementally to synthetic and real-world datasets.
  * Techniques included:

    * Horizontal flip
    * Gamma correction
    * Brightness adjustment
    * Gaussian noise
    * Blur
  * Resulted in a significantly larger and more diverse dataset.

* **Real-World Data Integration**:

  * Half of the real validation data was used in training.
  * Remaining half retained for testing and final evaluation.

* **Multi-task Loss**:
  We retained the semantic auxiliary loss used in Phase 2:

  ```python
  loss = traj_loss + semantic_k * semantic_loss
  ```

  * `semantic_k = 0.3` was used in the final configuration.

* **Hyperparameter Tuning**:

  * Continued using Optuna for tuning:

    * Learning rate
    * Weight decay
    * Number of trajectory modes
    * Semantic loss weight

* **Evaluation Metrics**:

  * ADE and FDE for trajectory prediction.
  * Semantic segmentation accuracy on real validation data.

---

### Results

Real-world supervision and data augmentations led to **significantly improved generalization**. Semantic supervision continued to serve as a powerful regularizer, even when labels were generated automatically.

| Metric               | Value                                |
| --------------------- | ------------------------------------|
| Validation ADE       | 1.4941                               |
| Kaggle ADE           | 1.2665                               |
| Real Data Used       | 50% in training                      |
| Semantic Loss Weight | 0.3                                  |
| Augmentation Methods | Flip, Gamma, Brightness, Noise, Blur |

---

### Summary

* Integrated real-world data into training and validation for better sim-to-real transfer.
* Used pretrained models to generate semantic labels for real scenes—maintaining auxiliary learning benefits.
* Enhanced dataset with diverse augmentations to improve robustness.
* Achieved significantly better real-world performance without sacrificing synthetic data performance.

[Link for weights](https://drive.google.com/file/d/1SfAzDkRln08PfNiXejR1UFK1k2F1B6Hb/view?usp=drive_link)
