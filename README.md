# PrimeNet: A Deep Learning Model for Prime Editing Efficiency Prediction

## Introduction
PrimeNet is a high-precision deep learning model designed for predicting the efficiency of prime editing. By integrating sequence-based features and epigenetic modifications, PrimeNet achieves robust cross-cell line performance. The model utilizes CNN-based feature extraction and an attention mechanism optimized with a Sigmoid activation function to capture essential editing determinants.

## Features
- **Sequence-to-Image Transformation**: Converts DNA sequences into images for CNN-based feature extraction.
- **Attention Mechanism**: Enhances feature selection to improve model interpretability.
- **Epigenetic Context Integration**: Considers chromatin accessibility for better generalization.
- **Cross-cell Line Robustness**: Demonstrates high predictive accuracy across different cellular environments.
- **Saliency Map Visualization**: Uses Integrated Gradients to highlight key sequence positions affecting editing efficiency.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/PrimeNet.git
cd PrimeNet

# Create a virtual environment
python -m venv primenet_env
source primenet_env/bin/activate  # On Windows, use `primenet_env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training the Model
```python
python train.py --data path/to/dataset --epochs 50 --batch_size 32
```

### Predicting Editing Efficiency
```python
python predict.py --input input_sequences.fasta --output predictions.csv
```

## Data Format
Input sequences should be provided in FASTA format, with additional annotations for epigenetic features if available.

## Model Performance
PrimeNet outperforms existing models in predicting prime editing efficiency, showing superior accuracy in various benchmarks. The model is optimized using hyperparameter tuning with Optuna.

## Citation
If you use PrimeNet in your research, please cite:
```
@article{yourpaper2025,
  title={PrimeNet: A Deep Learning Framework for Prime Editing Efficiency Prediction},
  author={Your Name and Collaborators},
  journal={Bioinformatics Journal},
  year={2025},
  doi={your-doi-here}
}
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaborations, please contact [your email] or open an issue on GitHub.
