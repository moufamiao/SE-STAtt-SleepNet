# AdaST
### AdaST: Adaptive Attentional Spatio-Temporal Network for Sleep Stage Classification
## Abstract
![AdaST](imgs/network.pdf)
Sleep is a vital physiological process that significantly impacts health and well-being, with disorderssuch as insomnia and sleep apnea presenting major challenges. Accurate sleep stage classification is essential for diagnosing and managing these conditions. While polysomnography (PSG) remains the gold standard, manual labeling is time-consuming and subject to variability. Deep learning-based methods have shown promise for EEG-based sleep staging, but challenges such as signal degradation and temporal dependencies during stage transitions persist. To address these issues, we propose the adaptive attention spatio-temporal network (AdaST), which integrates three key components: the adaptive signal enhancement pipeline (ASEP), the physiologically-aware spatio-temporal module (PSTM), and the selective feature aggregation module (SFAM). The ASEP enhances the robustness of the model to noisy signals by adaptively enhancing EEG data. The PSTM leverages physiologically informed convolutional kernels to effectively decouple spatio-temporal features during transitional stages, ensuring precise stage boundary detection. SFAM further improves classification accuracy by adaptively aggregating informative temporal features and suppressing noise, particularly in adjacent sleep stages. Extensive evaluations on three benchmark datasets—Sleep-EDF-20, Sleep-EDF-78, and SHHS—demonstrate that AdaST outperforms state-of-the-art methods, achieving superior accuracyand macro F1 scores. 


## Requirmenets:
- Python 3.11
- Pytorch=='1.8'
- Numpy
- Sklearn
- mne=='0.20.7'

## Prepare datasets

We used three public datasets in this study:
Sleep-EDF-20(https://physionet.org/content/sleep-edfx/1.0.0/)
Sleep-EDF-78(https://physionet.org/content/sleep-edfx/1.0.0/)
SHHS dataset(https://sleepdata.org/datasets/shhs)



