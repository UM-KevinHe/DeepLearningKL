# DeepLearningKL
This project is to train deep learning survival model with prior information, where KL divergence is used for incorporating. The code is modified based on [pycox][1].
## Code Details
We have modified a small part of pycox for training deep learning model. So please make sure that the following steps are completed before running the [notebook][2].
1. Change `LogisticHazard` in `logistic_hazard.py`:
    `def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.NLLLogistiHazardLoss()
        if loss == "option":
            loss = models.loss.NewlyDefinedLoss()
        super().__init__(net, loss, optimizer, device)`

[1]: https://github.com/havakv/pycox
[2]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep%20Learning%20with%20KL%20Divergence.ipynb

