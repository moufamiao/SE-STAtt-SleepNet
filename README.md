# AttnSleep
### SE-STAtt-SleepNet: A Signal-Enhanced Spatio Temporal Network with Adaptive Attention for Sleep Stage Classification

## Abstract
![SE-STAtt-SleepNet Architecture](imgs/AttnSleep.png)
Sleep stage classification is crucial for diagnosing sleep disorders and understanding sleep architecture, but existing methods are insufficient in spatiotemporal feature extraction and noise processing. This paper proposes a novel signal enhancement spatiotemporal network SE-STAtt-SleepNet for automatic sleep staging of single-channel electroencephalogram (EEG). The model contains three core modules: (1) Adaptive signal enhancement module, which significantly improves signal quality through dynamic wavelet threshold denoising, random time shift enhancement and amplitude adaptive noise injection; (2) Multi-scale spatiotemporal feature extraction network, combining hierarchical CNN and LSTM with large/small convolution kernels to capture local details and longterm temporal dependencies; (3) Dynamic feature focusing mechanism, which adaptively locates key sleep waveforms (such as spindles and K complexes) through attention weights. The experiment was verified based on three public datasets: Sleep-EDF-20, Sleep-EDF-78 and SHHS. SESTAtt-SleepNet showed better performance than existing methods in terms of overall accuracy (ACC) and macro F1 score (MF1). The model provides a new technical path for improving the accuracy and interpretability of sleep staging models by integrating the collaborative optimization strategy of signal enhancement, spatiotemporal modeling and attention mechanism.


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



