# DeepLearningKL
This project is to train deep learning survival model with prior information, where KL divergence is used for incorporating. The code is modified based on [pycox][1].
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

[1]: https://github.com/havakv/pycox
[2]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep%20Learning%20with%20KL%20Divergence.ipynb

