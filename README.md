# DeepLearningKL
This project is to train deep learning model in Survival Analysis. However, we allow user to incorporate prior information, which can be either statistical model or other neural networks. KL divergence is used for incorporating, which measures the difference between prior information and local information. The weights of prior and local information are selected by hyperparameter tuning and higher weights of prior model mean the model tends to believe more prior information than the local information, which means the quality local data may not be so satisfactory. Besides that, we also do an extension from single-risk to competing risk case, which means our software can also handle competing risk data.

We have designed our own software and provided with a tutorial. For more information, see below.

## Data
<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Data source</th>
  </tr>
     <tr>
         <td><b>Simulation Data 1</b></td>
        <td>10000</td>
        <td>
        Prior Data for Simulation 1 (Scheme 1), linear and proportional
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td><b>Simulation Data 2</b></td>
        <td>10000</td>
        <td>
        Prior Data for Simulation 1 (Scheme 2), non linear and proportional
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td><b>Simulation Data 3<b/></td>
        <td>10000(Prior)/300(Local)</td>
        <td>
        Prior and Local Data for Simulation 1 (Scheme 3), non linear and non proportional
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td>metabric</td>
        <td>1,904</td>
        <td>
        The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td><b>support</b></td>
        <td>8,804</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td><b>MIMIC-3</b></td>
        <td>35304</td>
        <td>
        Deidentified clinical data of patients admitted to ICU stay.
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
    <tr>
        <td><b>MIMIC-SEQ</b></td>
        <td>35304</td>
        <td>
        Deidentified clinical data of patients admitted to ICU stay (with many designed features related to time).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><a href="https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data">source</a>
    </tr>
</table>

## Models
<table>
    <tr>
        <th>Method</th>
        <th>Description</th>
        <th>Example</th>
    </tr>
    <tr>
        <td><b>KLDL-S/KLDL-C</b></td>
        <td>
        Our model, which requires prior information (model)
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td><b>KLDL-L<b/></td>
        <td>
        Our model, with manually defined 3 link functions as an option of parameter, used when the data shows some statistical properties.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>LogisticHazard (Nnet-survival)</td>
        <td>
        The Logistic-Hazard method parametrize the discrete hazards and optimize the survival likelihood <a href="#references">[12]</a> <a href="#references">[7]</a>.
        It is also called Partial Logistic Regression <a href="#references">[13]</a> and Nnet-survival <a href="#references">[8]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>PMF</td>
        <td>
        The PMF method parametrize the probability mass function (PMF) and optimize the survival likelihood <a href="#references">[12]</a>. It is the foundation of methods such as DeepHit and MTLR.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/pmf.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>DeepHit, DeepHitSingle</td>
        <td>
        DeepHit is a PMF method with a loss for improved ranking that 
        can handle competing risks <a href="#references">[3]</a>.
        </td>
        <td><a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit.ipynb">single</a>
        <a href="https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb">competing</a></td>
    </tr>
</table>

## Metrics
<table>
    <tr>
        <th>Metric</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>concordance_td (antolini)</td>
        <td>
        The time-dependent concordance index evaluated at the event times <a href="#references">[4]</a>.
        </td>
    </tr>
    <tr>
        <td><b>concordance_td (Uno)</b></td>
        <td>
        The time-dependent concordance index evaluated at the event times with truncation <a href="#references">[4]</a>.
        </td>
    </tr>
    <tr>
        <td>integrated_brier_score</td>
        <td>
        The integrated IPCW Brier score. Numerical integration of the `brier_score` <a href="#references">[5]</a><a href="#references">[6]</a>.
        </td>
    </tr>
    <tr>
        <td>integrated_nbll</td>
        <td>
        The integrated IPCW (negative) binomial log-likelihood. Numerical integration of the `nbll` <a href="#references">[5]</a><a href="#references">[1]</a>.
        </td>
    </tr>
</table>

## Hyperparameter Tuning
<table>
    <tr>
        <th>Metric</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>Grid Search</td>
        <td>
        </td>
    </tr>
    <tr>
        <td><b>Random Search</b></td>
        <td>
        </td>
    </tr>
    <tr>
        <td><b>Sobel Sequence</b></td>
        <td>
        </td>
    </tr>
</table>

## Real Data Result: Visualization
This is one comparison result that trained on [SUPPORT][7] data, we sample most of the data as prior data and the remaining data are used as local data. The prior data will be used to train a prior model and we obtain the estimated hazard rates from this prior model. The estimated hazard rates will be used to compute the value of our loss function and the model will be trained based on this loss function with only local data. For other models, only local data is accessible. 

The experiments are done 50 times with different samples of prior and local data. With the increasing size of local data, we can find the trends that our model becomes stabler and stabler. Also we can see our model performs better than existing model (LogisticHazard) in each case, but the difference decreases when the size of local data is large enough for existing model to train a satisfactory result.

![Real Data graph 1](https://user-images.githubusercontent.com/48302151/173245243-4c8eed5a-3923-46fe-8ea3-eabc446c9147.png)

## Tutorial

We have provided with 3 notebooks with interactive codes and results to show the way to train a prior model and use the results from trained prior model to train our model with local data.

1. [Tutorial 1: Using Our Model with Deep Learning as Prior][6]: This tutorial notebook shows how to use our model for a real dataset. We use SUPPORT as an example and we divide data with prior and local data. We provide ways to show **how to obtain the result from the trained prior model and how to define our new loss function**. Except for these novel parts, the necessary data-preprocessing, model training and evaluation tutorials are also shown in this notebook for users' convenience. 

2. Tutorial 2: Using Our Model with GLM as Prior: This tutorial notebooks shows how to use our model for a real dataset SUPPORT, **but with GLM as the prior model**. To train a GLM model we need to do some transformations for the time to event value. The other parts remain the same as Tutorial 1.

3. Tutorial 3: Identifying the Quality of Prior with Cross Validation: For this tutorial notebook, we will do similar data-preprocessing on SUPPORT, but we will modify some parts of the prior data to generate different prior models with different prior data. We will show **how to identify the quality of our prior model with the aid of multiple experiments and Cross Validation**. Basically, higher hyperparameter value means higher quality of prior data, since higher hyperparameter value means higher of the weight for prior data in the loss function.

Although these 3 tutorials show different versions of prior model. The core of using our model is to make sure the hazard rates matrix can be obtained with n be the number of observations (number of rows for this matrix) and K be the number of time intervals of discrete-time model (number of cols). We recommend users to begin with either Tutorial 1 or 2, we provide them with the same amount of details for users, respectively. For tutorial 3, since our goal is more advanced, we will omit some explanation words and refer users to previous tutorials.

## References
[1. Di Wang, Wen Ye, Kevin He Proceedings of AAAI Spring Symposium on Survival Prediction - Algorithms, Challenges, and Applications 2021, PMLR 146:232-239, 2021.][3]

[2. Kvamme H, Borgan Ø. Continuous and discrete-time survival prediction with neural networks. arXiv preprint arXiv:1910.06724. 2019 Oct 15.][5]

[3. Kvamme H, Borgan Ø, Scheel I. Time-to-event prediction with neural networks and Cox regression. arXiv preprint arXiv:1907.00825. 2019 Jul 1.][10]

[4. Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society:
Series B (Methodological), 34(2):187–202.][9]

[5. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep
neural network. BMC medical research methodology, 18(1):1–12.][11]

[1]: https://github.com/havakv/pycox
[2]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep%20Learning%20with%20KL%20Divergence.ipynb
[3]: http://proceedings.mlr.press/v146/wang21b/wang21b.pdf
[4]: https://github.com/UM-KevinHe/DeepLearningKL/blob/main/Deep_Learning_with_KL_divergence__Code_Details.pdf
[5]: https://arxiv.org/abs/1910.06724
[6]: https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_1_Using_Our_Model_with_Deep_Learning_as_Prior.ipynb
[7]: https://biostat.app.vumc.org/wiki/Main/SupportDesc
[8]: https://lifelines.readthedocs.io/en/latest/Quickstart.html
[9]: http://www.biecek.pl/statystykamedyczna/cox.pdf
[10]: https://www.jmlr.org/papers/volume20/18-424/18-424.pdf?ref=https://githubhelp.com
[11]: https://bmcmedresmethodol.biomedcentral.com/track/pdf/10.1186/s12874-018-0482-1.pdf
