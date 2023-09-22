# Dynamic Corrected Split Federated Learning with Homomorphic Encryption for U-shaped Medical Image Networks

This repository is a PyTorch implementation of the proposed DC-SFL (accepted by IEEE Journal of Biomedical and Health Informatics). This paper can be downloaded at this [link](https://ieeexplore.ieee.org/document/10256094).

#### Abstract
U-shaped networks have become prevalent in various medical image tasks such as segmentation, and restoration. However, most existing U-shaped networks rely on centralized learning which raises privacy concerns. To address these issues, federated learning (FL) and split learning (SL) have been proposed. However, achieving a balance between the local computational cost, model privacy, and parallel training remains a challenge. In this paper, we propose a novel hybrid learning paradigm called Dynamic Corrected Split Federated Learning (DC-SFL) for U-shaped medical image networks. To preserve data privacy, including the input, model parameters, label and output simultaneously, we propose to split the network into three parts hosted by different parties. We propose a Dynamic Weight Correction Strategy (DWCS) to stabilize the training process and avoid the model drift problem due to data heterogeneity. To further enhance privacy protection and establish a trustworthy distributed learning paradigm, we propose to introduce additively homomorphic encryption into the aggregation process of client-side model, which helps prevent potential collusion between parties and provides a better privacy guarantee for our proposed method. The proposed DC-SFL is evaluated on various medical image tasks, and the experimental results demonstrate its effectiveness. In comparison with state-of-the-art distributed learning methods, our method achieves competitive performance.

#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2023dcsfl,
  title={Dynamic Corrected Split Federated Learning with Homomorphic Encryption for U-shaped Medical Image Networks},
  author={Yang, Ziyuan and Chen, Yingyu and Huangfu, Huijie and Ran, Maosong and Wang, Hui and Li, Xiaoxiao and Zhang, Yi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```

#### Requirements
Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Note
This implementation is released for the restoration task, which is based on RED-CNN. It can be easily reproduced to the segmentation task, and DC-SFL can keep its performance in the segmentation task.

#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is cziyuanyang@gmail.com.
