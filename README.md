# AAP: Adversarial Attenuation Patch against SAR Object Detection Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of the **AAP (Adversarial Attenuation Patch)** method and integrates with popular attack algorithms from the Adversarial Robustness Toolbox (ART).

![framework2](https://raw.githubusercontent.com/boremycin/ImageBed/master/framework2.png)


## 📌 Key Features

- **AAP Method**: A novel Adversarial Attenuation Patch approach for SAR object detection.
- **Multiple Benchmark Methods**: Supports standard attacks including AdvPatch, DPatch, and RobustDPatch via ART.
- **YOLO Integration**: Seamless integration with YOLO-based object detection models.
- **Modular Design**: Clean and extensible architecture for research and experimentation.
- **Evaluation Tools**: Built-in metrics and visualization for assessing attack performance.

## 📁 Project Structure

```
SAAP/
├── aap_utils/                    # Core utilities and custom implementations
│   ├── adv_attenuation_patch.py  # AAP (Adversarial Attenuation Patch) implementation
│   ├── aap_utils.py             # Helper functions and model wrappers
│   ├── random_attack.py         # Random attack utilities for toy experiments
│   └── random_pipeline.py       # Attack pipeline management
├── adv_examples/                # Directory for adversarial examples (empty by default)
├── art/                         # Minimal integrated components from the Adversarial Robustness Toolbox (ART)
│   ├── attacks/
│   │   └── evasion/
│   │       ├── adversarial_patch/    # Standard adversarial patch attack
│   │       ├── dpatch.py             # DPatch attack implementation
│   │       └── dpatch_robust.py      # Robust DPatch attack implementation
│   ├── estimators/              # Model estimators (PyTorch YOLO support)
│   └── ...                      # Other essential ART components
├── dataset/                     # Dataset directory (empty by default)
├── models/                      # YOLO model configurations
│   └── hub/                     # Predefined YOLO architectures
├── result_pipeline/             # Evaluation results directory (empty by default)
├── utils/                       # General utilities and tools
├── weights/                     # Model weights directory (empty by default)
├── attack_advpatch.py          # Standard adversarial patch attack script
├── attack_dpatch.py            # DPatch attack script
├── attack_robustdpatch.py      # RobustDPatch attack script
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
- YOLOv5 (Ultralytics implementation)
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

#### Run supported attacking methods

```Bash
python attack_advpatch.py
python attack_dpatch.py
python attack_robustdpatch.py
python attackv2.py  # AAP method
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

If you use AAP in your research, please cite our work:

```bibtex
@article{Zhang2026SAAP,
  title   = {Towards Physically Realizable Adversarial Attenuation Patch against SAR Object Detection},
  author  = {Yiming Zhang and Weibo Qin and Feng Wang},
  journal = {arXiv preprint arXiv:2604.00887},
  year    = {2026}
}
```

## 🔒 Disclaimer

This code is intended for research and educational purposes only. Users are responsible for ensuring compliance with applicable laws and regulations when using adversarial attack techniques.

## Acknowledgments
- This project makes extensive use of the [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (ART) for implementing standard adversarial patch methods, including AdvPatch, DPatch, and RobustDPatch.
- The object detection backbone is based on  [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).
  