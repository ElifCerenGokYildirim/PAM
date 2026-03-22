# Pruned Adaptation Modules for Pre-Trained Model based Class-Incremental Learning

## 📄 Abstract

The continual learning literature has rapidly shifted from traditional class incremental learning (CIL) techniques to foundation model (FM)-based CIL methods without a clear understanding of how these newer approaches compare to strong, lightweight convolutional baselines. This abrupt transition has created a substantial methodological gap, making it difficult to assess whether recent FM-based CIL progress reflects genuine advances or merely the absence of rigorous baselines. To address this gap, we introduce Pruned Adaptation Modules (PAM), a simple yet effective method that freezes the vast majority of the pre-trained ResNet while enabling scalable continual adaptation through sparse task-specific layers. PAM yields up to a ~5×reduction in trainable parameters and a ~6×reduction in total parameters, significantly reducing the cost of continual updates. Across diverse benchmarks, PAM consistently mitigates catastrophic forgetting and outperforms state-of-the-art FM-based CIL approaches. Our findings position PAM as a strong and transparent baseline that helps bridge the gap between traditional and FM-based CIL, guiding future research for a more accurate assessment of true progress in continual adaptation.

![SAM Method Overview](./sam.png)

---

## 🚀 How to Run the Code

All experiments can be run using the `main.py` script. Below is a summary of the key configuration option. Please make sure to change them within the `main.py` file to run in the setup you wish.

### 📌 Main Configurations in `main.py`

| Argument         | Description                                                                                   | Example Values                   |
|------------------|-----------------------------------------------------------------------------------------------|----------------------------------|
| `dataset_name`   | Dataset to be used for class-incremental learning                                             | `'cifar100'`, `'imagenetr'`, `'cars'`, `'cub'` |
| `increment`      | Number of new classes introduced per task                                                    | `5`,`10`, `20`, etc.                 |
| `sparsity`       | Percentage of weights to prune in the PAM module (e.g., `0.96` means 96% sparsity)            | `0.95`, `0.96`, `0.97`           |
| `model`          | Backbone model variant (pre-trained ResNet architecture)                                     | `'resnet18'`, `'resnet50'`, `'resnet101'`, `'resnet152'` |

### 📂 Dataset Requirements

- For **ImageNet-R**, **Stanford Cars**, and **CUB-200**, please upload the datasets into the `data/` directory.


## 🧪 Environment Setup

All experiments were conducted using the environment defined in [`sam.yml`](./sam.yml).

To ensure reproducibility, please create and activate the conda environment using the provided `sam.yml` file:

```bash
conda env create -f sam.yml
conda activate sam
