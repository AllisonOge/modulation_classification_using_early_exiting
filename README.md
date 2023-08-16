# modulation_classification_early_exiting
This repository contains the public source code release for the paper <b>"Using Early Exits for Fast Inference in Automatic Modulation Classification".</b> This project is implemented mainly using PyTorch.

This code is created and maintained by [Elsayed Mohammed](https://github.com/ElSayedMMostafa).


## Abstract
Automatic modulation classification (AMC) plays a critical role in wireless communications by autonomously
classifying signals transmitted over the radio spectrum. Deep learning (DL) techniques are increasingly being used for AMC due to their ability to extract complex wireless signal features. However, DL models are computationally intensive and incur high inference latencies. This paper proposes the application of early exiting (EE) techniques for DL models used for AMC to accelerate inference. We present and analyze four early exiting architectures and a customized multi-branch training algorithm for this problem. Through extensive experimentation, we show that signals with moderate to high signal-to-noise ratios (SNRs) are easier to classify, do not require deep architectures, and can therefore leverage the proposed EE architectures. Our experi- mental results demonstrate that EE techniques can significantly reduce the inference speed of deep neural networks without sacrificing classification accuracy. We also thoroughly study the trade-off between classification accuracy and inference time when using these architectures. To the best of our knowledge, this work represents the first attempt to apply early exiting methods to AMC, providing a foundation for future research in this area.


## Files Description
>  **experiments.ipynb**: the main notebook implementing the 4 architectures and the main one as well as running the experiments and showing the results.

> **trainer.py**: a utility file to customizly train the EE models.

> **getData.py**: a utility file to load and process the dataset.

> **getModel.py**: a utility file to create and design the models.