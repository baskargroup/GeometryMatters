# Benchmarking Scientific ML Models for Flow Prediction

This repository provides the source code and training scripts for **"Geometry Matters: Benchmarking Scientific ML Approaches for Flow Prediction around Complex Geometries."** The project evaluates state-of-the-art **neural operators** and **foundation models** for predicting fluid dynamics over complex geometries.


## Paper
Our study introduces a benchmark for scientific machine learning (SciML) models in predicting steady-state flow across intricate geometries using high-fidelity simulation data. The full paper can be accessed here:

**"Geometry Matters: Benchmarking Scientific ML Approaches for Flow Prediction around Complex Geometries"** 
- Authors: *Ali Rabeh, Ethan Herron, Aditya Balu, Soumik Sarkar, Chinmay Hegde, Adarsh Krishnamurthy, Baskar Ganapathysubramanian*
- Preprint: [ArXiv Paper](https://arxiv.org/pdf/2501.01453)

## Features
- **Benchmarking of 11 SciML models**, including CNO, FNO, DeepONet, WNO, and Poseidon-based architectures.
- **Evaluation on FlowBench 2D Lid-Driven Cavity dataset**, a publicly available dataset on Hugging Face.
- **Comparison of two geometric representations**: **Signed Distance Fields (SDF)** and **Binary Masks**.
- **Hyperparameter tuning with WandB Sweeps**.
- **Residual and gradient calculations using FEM-based scripts**.

## Datasets
This study utilizes the **FlowBench 2D Lid-Driven Cavity (LDC) dataset**, which is publicly accessible on Hugging Face: [**FlowBench LDC Dataset**](https://huggingface.co/datasets/BGLab/FlowBench/tree/main/LDC_NS_2D/512x512)

The dataset is licensed under **CC-BY-NC-4.0** and serves as a benchmark for the development and evaluation of scientific machine learning (SciML) models.

### Dataset Structure
- **Geometry representation:** SDF and Binary Mask
- **Resolution:** 512×512
- **Fields:** Velocity (u, v) and Pressure (p)
- **Stored as:** Numpy tensors (`.npz` format)

## Installation
To set up the environment and install dependencies:
```bash
python3 -m venv sciml
source sciml/bin/activate 
pip install -r venv_requirements.txt
```

## Model Training
To train a model, run the following command:
```bash
python3 main_sweep.py --model "model_name" --sweep
```
### Supported Models
- **Neural Operators:** `FNO, CNO, WNO, DeepONet, Geometric-DeepONet`
- **Vision Transformers:** `scOT-T, Poseidon-T, scOT-B, Poseidon-B, scOT-L, Poseidon-L`

Before training, you need to specify the dataset paths in the **configurations** (YAML files):
```yaml
data:
  file_path_train_x: ./data/train_x.npz
  file_path_train_y: ./data/train_y.npz
  file_path_test_x: ./data/test_x.npz
  file_path_test_y: ./data/test_y.npz
```

## Model Inference
For model inference, use the scripts in the `plotting_scripts` folder:
```bash
python3 process_NO_deriv.py --model "$model" --config "$config_path" --checkpoint "$checkpoint_file"
```
Use `process_NO.py` for **neural operators** and `process_scot.py` for **vision transformers**.

## Evaluation & Plotting
The `plotting_scripts` folder contains Python scripts for:
- **Evaluating field predictions and errors**.
- **Calculating residuals using finite element methods (FEM)**.
- **Computing solution gradients**.

Example usage:
```bash
python plotting_scripts/plot_predictions.py --data_path ./data/test.npz --model_name fno
```

## Citation
If you use this code, please cite our work:
```bibtex
@misc{rabeh2024flowprediction,
      title={Geometry Matters: Benchmarking Scientific ML Approaches for Flow Prediction around Complex Geometries},
      author={Ali Rabeh, Ethan Herron, Aditya Balu, Soumik Sarkar, Chinmay Hegde, Adarsh Krishnamurthy, Baskar Ganapathysubramanian},
      year={2024},
      eprint={2405.19101},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contributing
We welcome contributions! If you’d like to improve this project, please fork the repository and submit a pull request.

## License
This repository is licensed under the MIT License.
