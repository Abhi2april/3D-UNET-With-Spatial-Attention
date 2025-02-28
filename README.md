# 3D U-Net with Spatial Attention for Brain Tumor Segmentation

This repository implements a 3D U-Net architecture enhanced with Spatial Attention for segmenting brain tumors. The code is designed to work with volumetric medical images (in NIfTI format) and is primarily applied on the BraTS 2020 dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset and File Structure](#dataset-and-file-structure)
- [Model Architecture](#model-architecture)
- [Custom Metrics and Loss Functions](#custom-metrics-and-loss-functions)
- [Data Generator](#data-generator)
- [Training Configuration](#training-configuration)
- [Prediction and Visualization](#prediction-and-visualization)
- [Saving and Evaluating the Model](#saving-and-evaluating-the-model)
- [Usage](#usage)
- [Notes](#notes)

---

## Overview

The code implements a 3D U-Net with the following key elements:
- **Multi-modality input:** Uses FLAIR, T1, and T1ce images.
- **Spatial Attention:** A custom Keras layer is introduced to re-weight feature maps at several stages of the network.
- **Volumetric Data Processing:** The input volumes and corresponding segmentation masks are loaded from NIfTI files.
- **Custom Data Generator:** Loads, resizes, normalizes the images, and performs one-hot encoding on segmentation masks.
- **Training and Visualization:** Training is performed using multi-GPU support via TensorFlow’s distribution strategy. Visualization functions overlay predictions with the ground truth.

---

## Dependencies

The code makes use of several Python libraries:
- **General purpose:** `os`, `glob`, `shutil`, `numpy`, `pandas`, `cv2` (OpenCV)
- **Plotting and Visualization:** `matplotlib`, `seaborn`, `PIL`, `skimage`, `nilearn`
- **Medical Imaging:** `nibabel` (for NIfTI files), `gif_your_nifti` (installed from GitHub)
- **Deep Learning:** `tensorflow`, `keras`, and various submodules (layers, callbacks, optimizers)
- **Others:** `sklearn` for preprocessing, train-test splitting, and metric evaluation

> **Note:** The library `gif_your_nifti` is installed directly from GitHub using:
>
> ```bash
> pip install git+https://github.com/miykael/gif_your_nifti
> ```

---

## Dataset and File Structure

The code assumes the dataset is organized as follows:
- **Training Data:** Located at  
  `../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/`
- **Validation Data:** Located at  
  `../input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData`

Each case directory is expected to contain NIfTI files for:
- FLAIR (`<case_id>_flair.nii`)
- T1 (`<case_id>_t1.nii`)
- T1ce (`<case_id>_t1ce.nii`)
- (Optionally) T2 (`<case_id>_t2.nii`)
- Segmentation (`<case_id>_seg.nii`)

A mapping of segmentation classes is defined as:

```python
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE',
    2 : 'EDEMA',
    3 : 'ENHANCING' 
}
```

---

## Model Architecture

The network is built using a U-Net structure that includes:

- **Encoder Path:** Multiple stages of 3D convolutions (with 32, 64, 128, 256 filters) followed by max pooling.
- **Spatial Attention:** Applied after key convolutional blocks using a custom layer.  
  The attention layer uses a 3D convolution (with a kernel size of 7 by default) followed by a sigmoid activation to generate an attention map, which is then multiplied with the input feature maps.
- **Decoder Path:** Up-sampling and convolutional layers, with skip connections from the encoder, are used to recover spatial resolution.
- **Final Layer:** A 1×1×1 convolution outputs 4 channels with softmax activation corresponding to the segmentation classes.

The model is defined in the function `build_unet()` and compiled with a set of custom metrics and a combined loss function.

---

## Custom Metrics and Loss Functions

The following custom metric functions are defined:
- **Dice Coefficient (overall and per-class):**  
  - `dice_coef()`: Average dice score over 4 classes.
  - `dice_coef_necrotic()`, `dice_coef_edema()`, `dice_coef_enhancing()`: Dice scores for individual tumor regions.
- **Precision, Sensitivity, and Specificity:** Standard metrics computed using tensor operations.
- **Combined Loss:** Combines the dice loss and categorical crossentropy loss (weighted equally).

---

## Data Generator

A custom data generator (`DataGenerator`) is implemented as a subclass of `keras.utils.Sequence` to efficiently load volumetric data:
- **Input Processing:**  
  For each case, the generator loads the FLAIR, T1ce, and T1 modalities. Each slice is resized to the target dimensions (e.g., 128×128) and normalized.
- **Segmentation Masks:**  
  The segmentation volume is resized using nearest neighbor interpolation. Labels are adjusted (label 4 is converted to 3) and then one-hot encoded into 4 classes.
- **Batch Generation:**  
  The generator yields batches of data with shape `(batch_size, IMG_SIZE, IMG_SIZE, IMG_SIZE, 3)` for inputs and `(batch_size, IMG_SIZE, IMG_SIZE, IMG_SIZE, 4)` for outputs.

---

## Training Configuration

- **Multi-GPU Training:**  
  Uses `tf.distribute.MirroredStrategy` for parallel training.
- **Compilation:**  
  The model is compiled with the RMSprop optimizer (learning rate = 0.0001) and is monitored with various metrics.
- **Callbacks:**  
  A `ReduceLROnPlateau` callback is used (with early stopping and model checkpoint callbacks provided as comments).

---

## Prediction and Visualization

- **Prediction Functions:**
  - `predictByPath(model, case_path, case_id)`: Loads the input modalities, preprocesses them, and returns the model's prediction.
- **Visualization Functions:**
  - `showPredictsById(model, case_id, data_path, start_slice)`: Displays the original FLAIR image, overlays the ground truth segmentation, and shows predictions.
- **Metrics Calculation:**
  - `compute_segmentation_metrics(y_true, y_pred)`: Computes precision, recall, and F1 scores.

---

## Saving and Evaluating the Model

```python
model.save('3d_attention_unet.h5')
```

---

## Usage

1. Install dependencies.
2. Configure dataset paths.
3. Run the script to train and visualize the model.
4. Save the trained model for inference.

---

## Notes

- Adjust parameters as necessary for different datasets.
- Ensure correct version usage for functions with multiple definitions.
- Optimize error handling before deployment.
