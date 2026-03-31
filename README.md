# AAP: Adversarial Attenutation Patch for SAR objector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implements of the **AAP (Adversarial Attenuation Patch)** method and integrates with popular attack algorithms from the Adversarial Robustness Toolbox (ART).

![20260331151538](https://raw.githubusercontent.com/boremycin/ImageBed/master/20260331151538.png)

## 📌 Key Features

- **AAP Method**: Our novel Adversarial Attenuation Patch approach for SAR objector
- **Multiple Benchmark Methods Support**: Includes AdvPatch, DPatch and RobustDPatch attacks supported By ART.
- **YOLOv5 Integration**: Seamless integration with YOLO series object detection models
- **Modular Design**: Clean architecture for easy extension and experimentation
- **Evaluation Tools**: Built-in metrics and visualization for attack effectiveness assessment

## 📁 Project Structure

```
SAAP/
├── aap_utils/                    # Core utilities and custom implementations
│   ├── adv_attenuation_patch.py  # AAP (Adversarial Attenuation Patch) implementation
│   ├── aap_utils.py             # Helper functions and model wrappers
│   ├── random_attack.py         # Random attack utilities for toy experiments
│   └── random_pipeline.py       # Attack pipeline management
├── art/                         # Adversarial Robustness Toolbox (minimal version)
│   ├── attacks/
│   │   └── evasion/
│   │       ├── adversarial_patch/    # Standard adversarial patch attack
│   │       ├── dpatch.py             # D-Patch attack implementation
│   │       └── dpatch_robust.py      # Robust D-Patch attack implementation
│   ├── estimators/              # Model estimators (PyTorch YOLO support)
│   └── ...                      # Other essential ART components
├── models/                      # YOLO model configurations
│   └── hub/                     # Predefined YOLO architectures
├── utils/                       # General utilities and tools
├── attack_advpatch.py          # Standard adversarial patch attack script
├── attack_dpatch.py            # D-Patch attack script
├── attack_robustdpatch.py      # Robust D-Patch attack script
├── attackv1.py                 # Integrated development version (for learning ART interfaces)
├── attackv2.py                 # Standard production version (using AAP method)
├── detect_ship.py              # Ship detection example
├── val.py                      # Validation and evaluation script
└── advset.yaml                 # Dataset configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.2.2
- YOLOv5 7.0.14
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/SAAP.git
cd SAAP

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

#### Running AAP Attack (Standard Version)
```bash
python attackv2.py
```

#### Running D-Patch Attack
```bash
python attack_dpatch.py
```

#### Running Robust D-Patch Attack
```bash
python attack_robustdpatch.py
```

#### Running Standard Adversarial Patch Attack
```bash
python attack_advpatch.py
```


## 📊 Evaluation

The platform provides comprehensive evaluation metrics including:

- False Negative Rate (FNR) 
- False Positives Per Image (FPPI)
- mAP and detection confidence reduction

Run evaluation with:
```bash
python result_fprfnr.py
python result_pipeline.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SAAP in your research, please cite our work:

```bibtex
@article{Zhang2026SAAP,
  title={Towards Physically Realizable Adversarial Attenuation Patch against SAR Object Detection},
  author={Yiming, Zhang},
  journal={},
  year={2026}
}
```

## 🔒 Disclaimer

This code is intended for research and educational purposes only. Users are responsible for ensuring compliance with applicable laws and regulations when using adversarial attack techniques.

## Acknowledgments
- This project heavily utilizes the [Adversarial Robust Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) for implementing benchmark adversarial  patch methods like AdvPatch, DPatch and RobustDPatch.
- The object detection model is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).