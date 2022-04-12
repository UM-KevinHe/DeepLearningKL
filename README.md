# DeepLearningKL
This project is to train deep learning survival model with prior information, where KL divergence is used for incorporating. The code is modified based on [pycox][1].
## Core Idea
See 2.1 [here][4]

## Preliminary Result
Here is one comparison result, with local model both trained with LogisticHazard (nnet-survival). For the group integrating with KL divergence, the parameters are trained by Logistic Regression using prior data. Experiments are done 50 times with the same training, validation and test data. Cross Validation (CV) is not used for saving time. But based on the result, our model should perform better after applying CV.

![image](https://user-images.githubusercontent.com/48302151/162856554-2e5d4c7b-715b-4791-98a8-0882483064e0.png)

## Code Details
We have modified a small part of pycox for training deep learning model. So please make sure that the following steps are completed before running the [notebook][2].
1. Change `LogisticHazard` in `logistic_hazard.py`:
```diff
def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
    self.duration_index = duration_index
    if loss is None:
        loss = models.loss.NLLLogistiHazardLoss()
+   if loss == "option":
+       loss = models.loss.NewlyDefinedLoss()
    super().__init__(net, loss, optimizer, device)
```
This is because the loss function is slightly changed for our project.

2. Add class `NewlyDefinedLoss` in `loss.py`:

```diff
class NewlyDefinedLoss(_Loss):
    def forward(self, phi: Tensor, combined_info: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return newly_defined_loss(phi, combined_info, idx_durations, events, self.reduction)
```

3. Add method `newly_defined_loss` in `loss.py`:

```diff
def newly_defined_loss(phi: Tensor, combined_info: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    bce = F.binary_cross_entropy_with_logits(phi, combined_info, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)
```

Here is the diff between our loss function and `NLLLogistiHazardLoss`
```diff
def newly_defined_loss(phi: Tensor, combined_info: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:

-   if phi.shape[1] <= idx_durations.max():
-       raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
-                        f" Need at least `phi.shape[1] = {idx_durations.max().item() + 1}`," +
-                        f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
-   y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)  # TODO: Data Expansion!
-   bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
+   bce = F.binary_cross_entropy_with_logits(phi, combined_info, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)
```

where `combined_info` is the hazard function obtained from prior information.

## References
[Di Wang, Wen Ye, Kevin He Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021, PMLR 146:232-239, 2021.][3]

[1]: https://github.com/havakv/pycox
[2]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep%20Learning%20with%20KL%20Divergence.ipynb
[3]: http://proceedings.mlr.press/v146/wang21b/wang21b.pdf
[4]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep_Learning_with_KL_divergence__Code_Details.pdf
