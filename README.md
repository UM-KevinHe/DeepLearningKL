# DeepLearningKL
This project is to train a deep learning model in Survival Analysis. However, we allow user to incorporate prior information, which can be either statistical models or other neural networks. KL divergence is used for incorporating, which measures the difference between prior information and local information. The weights of prior and local information are selected by hyperparameter tuning and higher weights of prior model mean the model tends to believe more prior information than the local information, which means the quality local data may not be so satisfactory. Besides, we also do an extension from single-risk to competing risk case, which means our software can also handle competing risk data.

## Tutorials

We have provided two kinds of tutorials. One kind of them is designed as a **dictionary**, containing a detailed and comprehensive introduction of all common usages related to the software (). For the other one, it contains a series of small tutorials, aiming at teaching users how to apply specific models in our software, you can see **Models** below for more information.

## Data
<table>
    <tr>
        <th>Dataset</th>
        <th>Size</th>
        <th>Dataset</th>
        <th>Data source/Generation Code</th>
  </tr>
     <tr>
         <td><b>Simulation Data 1</b></td>
        <td>10000</td>
        <td>
        Prior Data for Simulation 1 (Scheme 1), linear and proportional
        </td>
         <td><code>read_data.simulation_data(option="linear ph", n=10000)</code></td>
    </tr>
    <tr>
        <td><b>Simulation Data 2</b></td>
        <td>10000</td>
        <td>
        Prior Data for Simulation 1 (Scheme 2), non linear and proportional
        </td>
        <td><code>read_data.simulation_data(option="non linear ph", n=10000)</code></td>
    </tr>
    <tr>
        <td><b>Simulation Data 3<b/></td>
        <td>10000(Prior)/300(Local)</td>
        <td>
        Prior and Local Data for Simulation 1 (Scheme 3), non linear and non proportional
        </td>
        <td><code>read_data.simulation_data(option="non linear non ph", n=10000)</code></td>
    </tr>
    <tr>
        <td><b>MNIST</b></td>
        <td>60000(training set)+10000(test set)</td>
        <td>
        A commonly-seen benchmark for image-processing tasks. We use it here as the simulation data to illustrate the usage of CNN for our model.
        </td>
        <td><code>read_data.image_data()</code></td>
    </tr>
    <tr>
        <td>metabric</td>
        <td>1,904</td>
        <td>
        The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).
        See <a href="#references">[2]</a> for details.
        </td>
        <td><code>read_data.metabric_data()</code></td>
    </tr>
    <tr>
        <td><b>support</b></td>
        <td>8,873</td>
        <td>
        Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT), we transfer it to discrete outcomes with a threshold based on its setting. See <a href="#references">[7]</a> for details.
        </td>
        <td><code>read_data.support_data()</code></td>
    </tr>
    <tr>
        <td><b>MIMIC-3</b></td>
        <td>35304</td>
        <td>
        Deidentified clinical data of patients admitted to ICU stay. The version follows <a href="#references">[8]</a>.
        </td>
        <td> Private Data</td>
    </tr>
    <tr>
        <td><b>MIMIC-SEQ</b></td>
        <td>35304</td>
        <td>
        Deidentified clinical data of patients admitted to ICU stay (with time series features).
        See <a href="#references">[10]</a> for details.
        </td>
        <td>Private Data</td>
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
        <td><b>KLDL-S</b></td>
        <td>
        Our model, which requires prior information (model). S means single risk
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20KLDL-S.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td><b>KLDL-C</b></td>
        <td>
        Our model, which requires prior information (model). C means competing risk.
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20KLDL-C.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td><b>KLDL-L<b/></td>
        <td>
        Our model, with manually defined 3 link functions as an option of parameter, used when the data shows some statistical properties.
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20KLDL-L.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>LogisticHazard (Nnet-survival)</td>
        <td>
        The Logistic-Hazard method parametrize the discrete hazards and optimize the survival likelihood <a href="#references">[12]</a> <a href="#references">[7]</a>.
        It is also called Partial Logistic Regression and Nnet-survival.
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20KLDL-S.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>PMF</td>
        <td>
        The PMF method parametrize the probability mass function (PMF) and optimize the survival likelihood <a href="#references">[12]</a>. It is the foundation of methods such as DeepHit and MTLR.
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20Deephit%2C%20PMF%20and%20MTLR.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>DeepHitSingle</td>
        <td>
        DeepHit is a PMF method with a loss for improved ranking that 
        can handle competing risks <a href="#references">[3]</a>. This is the version for single-risk setting.
        </td>
        <td><a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20Deephit%2C%20PMF%20and%20MTLR.ipynb">notebook</a>
        </td>
    </tr>
    <tr>
        <td>DeepHit</td>
        <td>
        Same as above, but this is the version for the competing risk setting.
        </td>
        <td>
        <a href="https://nbviewer.org/github/UM-KevinHe/DeepLearningKL/blob/main/Tutorial_%20KLDL-C.ipynb">notebook</a></td>
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
        The time-dependent concordance index evaluated at the event times with truncation <a href="#references">[9]</a>.
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

## Simulation Result: visualization

40000 individuals are sampled and 2% of them are used for local data, while the other 98% are prior data. $x_1, x_2$ are two covariates in $x, y$-axis, respectively. The average predicted survival value for each time point is on $z$-axis and the color demonstrates the value with the sidebar as a legend on the right. The first graph (from left to right, top to bottom) is the truth, which is the relationship between two covariates of individuals and true survival values visualized as an ellipse. The second graph is the predicted survival values trained by LogisticHazard with only prior data. Since the size of it is large, the result differs not too much from the truth. The third graph is the predicted survival values trained by LogisticHazard with only local data, this result is unsatisfactory due to the small size of data. The fourth graph is the results for our model, combining local data and prior information. The result looks much more reasonable compared with the third graph, with a more accurate range of predicted values and also a more regular ellipse shape.

![Simulation_2_true](https://user-images.githubusercontent.com/48302151/191780765-93c073ad-aa6f-4589-8b61-b2378121146e.png)
![Simulation_2_prior](https://user-images.githubusercontent.com/48302151/191780742-11ae271d-0169-4dce-9009-01f780e65f0c.png)
![Simulation_2_local](https://user-images.githubusercontent.com/48302151/191780710-228fcae0-4bb0-4a45-b3b5-06959c27b78a.png)
![Simulation_2_KL](https://user-images.githubusercontent.com/48302151/191780659-3ef2d029-db44-4dc3-a90e-a50575ae06cf.png)

## Real Data Result: Visualization
This is one comparison result that trained on MIMIC-3 data, we sample most of the data as prior data and the remaining data are used as local data. The prior data will be used to train a prior model and we obtain the estimated hazard rates from this prior model. The estimated hazard rates will be used to compute the value of our loss function and the model will be trained based on this loss function with only local data. For other models, only local data is accessible. Note that only part of the features will be selected out as those in the prior model, which means the prior information is imperfect.

The experiments are done 50 times with different samples of prior and local data. With the increasing size of local data, we can find the trends that our model becomes stabler and stabler. Also we can see our model performs better than existing models. Especially when the ratio for local data is 20\%, the result is even better than LogisticHazard trained on the prior data, which means our model has the potential to achieve an extraordinary prediction result with a moderate size of data at the case when the prior information is imperfect.

![Real_Data_MIMIC_1_Imperfect](https://user-images.githubusercontent.com/48302151/191658962-dc53d9d2-dedb-4706-92a6-60e4e1c53498.png)

## References
[1. Wang, D., Ye, W., Sung, R., Jiang, H., Taylor, J. M. G., Ly, L., and He, K. (2021). Kullback-leibler-based discrete failure time models for integration of published prediction models with new time-to-event dataset.][3]

[2. Kvamme H, Borgan Ø. Continuous and discrete-time survival prediction with neural networks. arXiv preprint arXiv:1910.06724. 2019 Oct 15.][5]

[3. Kvamme H, Borgan Ø, Scheel I. Time-to-event prediction with neural networks and Cox regression. arXiv preprint arXiv:1907.00825. 2019 Jul 1.][10]

[4. Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society:
Series B (Methodological), 34(2):187–202.][9]

[5. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep
neural network. BMC medical research methodology, 18(1):1–12.][11]

[6. Liu L, Fang X, Wang D, Tang W, He K. KL-divergence Based Deep Learning for Discrete Time Model. arXiv preprint arXiv:2208.05100. 2022 Aug 10.][12]

[7. Knaus WA, Harrell FE, Lynn J, Goldman L, Phillips RS, Connors AF, Dawson NV, Fulkerson WJ, Califf RM, Desbiens N, Layde P. The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults. Annals of internal medicine. 1995 Feb 1;122(3):191-203.][13]

[8. Purushotham S, Meng C, Che Z, Liu Y. Benchmarking deep learning models on large healthcare datasets. Journal of biomedical informatics. 2018 Jul 1;83:112-34.][14]

[9. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011 May 10;30(10):1105-17. doi: 10.1002/sim.4154. Epub 2011 Jan 13. PMID: 21484848; PMCID: PMC3079915.][15]

[10. Tang W, Ma J, Mei Q, Zhu J. SODEN: A Scalable Continuous-Time Survival Model through Ordinary Differential Equation Networks. J. Mach. Learn. Res.. 2022 Jan 1;23:34-1.][16]

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
[12]: https://arxiv.org/abs/2208.05100
[13]: https://www.acpjournals.org/doi/full/10.7326/0003-4819-122-3-199502010-00007
[14]: https://www.sciencedirect.com/science/article/pii/S1532046418300716
[15]: https://pubmed.ncbi.nlm.nih.gov/21484848/
[16]: https://www.jmlr.org/papers/volume23/20-900/20-900.pdf
