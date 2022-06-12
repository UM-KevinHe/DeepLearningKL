# DeepLearningKL
This project is to train deep learning survival model with prior information, where KL divergence is used for incorporating. The code is modified based on [pycox][1], with the changes of loss function and the way to obtain necessary "prior information" used for it.
## Core Idea
Basically, we exploit the similarity of the loss function in discrete-time model and the KL divergence. Due to that reason it is really easy to find out how to compute the loss function and the way to unify two loss functions. For more details, see 2.1 [here][4].

## Real Data Result: Visualization
This is one comparison result that trained on SUPPORT data, we sample most of the data as prior data and the remaining data are used as local data. The prior data will be used to train a prior model and we obtain the estimated hazard rates from this prior model. The estimated hazard rates will be used to compute the value of our loss function and the model will be trained based on this loss function with only local data. For other models, only local data is accessible. 

The experiments are done 50 times with different samples of prior and local data. With the increasing size of local data, we can find the trends that our model becomes stabler and stabler. Also we can see our model performs better than existing model (LogisticHazard) in each case, but the difference decreases when the size of local data is large enough for existing model to train a satisfactory result.

![image](https://user-images.githubusercontent.com/48302151/162856554-2e5d4c7b-715b-4791-98a8-0882483064e0.png)

## Tutorial

We have provided with 2 notebooks with interactive codes and results to show the way to train a prior model and use the results from trained prior model to train our model with local data.

## References
[Di Wang, Wen Ye, Kevin He Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021, PMLR 146:232-239, 2021.][3]

[1]: https://github.com/havakv/pycox
[2]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep%20Learning%20with%20KL%20Divergence.ipynb
[3]: http://proceedings.mlr.press/v146/wang21b/wang21b.pdf
[4]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep_Learning_with_KL_divergence__Code_Details.pdf
