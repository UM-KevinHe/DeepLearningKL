from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch  # For building the networks
import torchtuples as tt  # Some useful functions
import Structure

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder

from pycox.datasets import metabric
from pycox.models import LogisticHazard
from pycox.models import PMF
from pycox.models import MTLR
from pycox.models import DeepHitSingle
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

concordance_td_list_1 = []
integrated_brier_score_list_1 = []
integrated_nbll_list_1 = []
concordance_td_list_prior_1 = []
integrated_brier_score_list_prior_1 = []
integrated_nbll_list_prior_1 = []
concordance_td_list_old_1 = []
integrated_brier_score_list_old_1 = []
integrated_nbll_list_old_1 = []
best_eta_list = []


class NewlyDefinedLoss(nn.Module):
    def __init__(self, eta, model, time_intervals, option=None):
        super().__init__()
        self.eta = eta
        self.model = model
        self.time_intervals = time_intervals
        self.option = option

    def newly_defined_loss(self, phi, x_train, idx_durations, events):
        option = self.option
        prior_info_train = self.model.predict_hazard(x_train)

        time = idx_durations.cpu()
        time = np.array(time).reshape(time.shape[0], -1)
        zeros_train = np.zeros((time.shape[0], self.time_intervals + 1))  # 21 - 1 = 20
        np.put_along_axis(zeros_train, time, np.array(events.cpu()).reshape(-1, 1), axis=1)
        zeros_train = zeros_train[:, :-1]

        combined_info = (torch.tensor(zeros_train, device=phi.device) + self.eta * prior_info_train) / (1 + self.eta)

        idx_durations = idx_durations.view(-1, 1)
        if option is None:
            bce = F.binary_cross_entropy_with_logits(phi, combined_info, reduction='none')
        else:
            bce = nn.BCELoss(phi, combined_info, reduction='none')
        loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
        return loss.mean()

    def forward(self, phi, x_train, idx_durations, events):
        return self.newly_defined_loss(phi, x_train, idx_durations, events)


class NewlyDefinedLoss2(nn.Module):
    def __init__(self, option=None):
        super().__init__()
        self.option = option

    def nll_logistic_hazard(self, phi, idx_durations, events):
        option = self.option
        if phi.shape[1] <= idx_durations.max():
            raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                             f" Need at least `phi.shape[1] = {idx_durations.max().item() + 1}`," +
                             f" but got `phi.shape[1] = {phi.shape[1]}`")
        if events.dtype is torch.bool:
            events = events.float()
        events = events.view(-1, 1)
        idx_durations = idx_durations.view(-1, 1)
        y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
        if option is None:
            bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
        else:
            bce = F.binary_cross_entropy(phi, y_bce, reduction='none')
        loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
        return loss.mean()

    def forward(self, phi, idx_durations, events):
        return self.nll_logistic_hazard(phi, idx_durations, events)


class NewlyDefinedLoss3(nn.Module):
    def __init__(self):
        super().__init__()

    def nll_pmf_cr_2(self, phi, idx_durations, events, reduction='mean'):

        N = phi.shape[0]
        Q = phi.shape[1]
        K = phi.shape[2]

        test = torch.zeros((N, Q, K))

        for i in range(N):
            time_temp = idx_durations[i]
            status_temp = events[i] - 1
            if status_temp == -1:
                continue
            test[i][status_temp][time_temp] = 1

        test = torch.cat((test, 1 - torch.sum(test, axis=1).reshape(N, 1, K)), 1)
        phi = torch.cat((phi, 1 - torch.sum(phi, axis=1).reshape(N, 1, K)), 1)
        phi = F.softmax(phi, dim=1)

        bce = torch.sum(-torch.log(phi) * test, 1)

        idx_durations = idx_durations.view(-1, 1)
        events = events.view(-1, 1)

        loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
        return loss.mean()

    def forward(self, phi, idx_durations, events, reduction='mean'):
        return self.nll_pmf_cr_2(phi, idx_durations, events, reduction='mean')


class NewlyDefinedLoss4(nn.Module):
    def __init__(self, eta, model):
        super().__init__()

        self.eta = eta
        self.model = model

    def nll_pmf_cr_2(self, phi, x_train, idx_durations, events, reduction='mean'):

        N = phi.shape[0]
        Q = phi.shape[1]
        K = phi.shape[2]

        test = torch.zeros((N, Q, K))

        for i in range(N):
            time_temp = idx_durations[i]
            status_temp = events[i] - 1
            if status_temp == -1:
                continue
            test[i][status_temp][time_temp] = 1

        pmf_pre = torch.tensor(self.model.predict(x_train))
        N = pmf_pre.shape[0]
        K = pmf_pre.shape[2]
        pmf_pre = torch.cat((pmf_pre, 1 - torch.sum(pmf_pre, axis=1).reshape(N, 1, K)), 1)
        pmf = F.softmax(pmf_pre, dim=1)
        pmf = pmf[:, :-1, :]

        test = (self.eta * pmf + test) / (1 + self.eta)

        test = torch.cat((test, 1 - torch.sum(test, axis=1).reshape(N, 1, K)), 1)
        phi = torch.cat((phi, 1 - torch.sum(phi, axis=1).reshape(N, 1, K)), 1)
        phi = F.softmax(phi, dim=1)

        bce = torch.sum(-torch.log(phi) * test, 1)

        idx_durations = idx_durations.view(-1, 1)
        events = events.view(-1, 1)

        loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
        return loss.mean()

    def forward(self, phi, x_train, idx_durations, events, reduction='mean'):
        return self.nll_pmf_cr_2(phi, x_train, idx_durations, events, reduction='mean')


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


def cont_to_disc(data, labtrans=None, scheme="quantiles", time_intervals=20, competing=False):
    get_target = lambda df: (df['duration'].values, np.array(df['event'].values, dtype=np.float32))
    if labtrans is None:
        if competing is True:
            labtrans = LabTransform(time_intervals)
        else:
            labtrans = LogisticHazard.label_transform(time_intervals, scheme)
        y_train = labtrans.fit_transform(*get_target(data))
        data["duration"] = y_train[0]
        return labtrans, data
    else:
        y_train = labtrans.transform(*get_target(data))
        data["duration"] = y_train[0]
        return data



def hyperparameter_set_list(hidden_nodes=None,
                            hidden_layers=None,
                            batch_norm=None,
                            learning_rate=None,
                            batch_size=None,
                            dropout=None,
                            optimizer=None,
                            alpha=None,  # Specially designed for DeepHit
                            sigma=None,  # Specially designed for DeepHit
                            ):
    if optimizer is None:
        optimizer = [tt.optim.Adam()]
    if batch_size is None:
        batch_size = [32, 64, 128]
    if learning_rate is None:
        learning_rate = [0.0005, 0.001, 0.01, 0.1]
    if batch_norm is None:
        batch_norm = [True, False]
    if hidden_layers is None:
        hidden_layers = [2, 3, 4]
    if hidden_nodes is None:
        hidden_nodes = [32, 64, 128]
    if dropout is None:
        dropout = [0, 0.1, 0.25]
    if alpha is None:
        alpha = [1]
    if sigma is None:
        sigma = [0.1]

    set_list = []
    for a in hidden_nodes:
        for b in hidden_layers:
            for c in batch_norm:
                for d in learning_rate:
                    for e in batch_size:
                        for f in dropout:
                            for g in optimizer:
                                for h in alpha:
                                    for i in sigma:
                                        hyperparameter_set = {"hidden_nodes": a, "hidden_layers": b, "batch_norm": c,
                                                              "learning_rate": d, "batch_size": e, "dropout": f,
                                                              "optimizer": g, "alpha": h, "sigma": i}
                                        set_list.append(hyperparameter_set)
    return set_list


def mapper_generation(cols_standardize=None, cols_leave=None, cols_categorical=None):
    """
    Used for Standardization of the data.
    It will be different for various datasets.

    Usage: Firstly fit_transform() for the training data, then apply it (transform()) on the testing data.
    """

    standardize = False
    leave = False

    standardize_features = []
    leave_features = []
    categorical_features = []

    if cols_standardize is None and cols_leave is None and cols_categorical is None:
        raise ValueError("Please at least assign one kinds of data to be processed. Either standardize, categorical "
                         "or leave")
    if cols_standardize is not None:
        standardize_features = [([col], StandardScaler()) for col in cols_standardize]
    if cols_leave is not None:
        leave_features = [(col, None) for col in cols_leave]
    if cols_categorical is not None:
        categorical_features = [([col], OneHotEncoder()) for col in cols_categorical]

    x_mapper_float = DataFrameMapper(standardize_features + leave_features + categorical_features)

    return x_mapper_float


def cross_validation_eta(df_local, eta_list, model_prior,
                         parameter_set=None,
                         time_intervals=20,
                         epochs=512,
                         patience=5,
                         n_splits=5,
                         metric="C-index",
                         verbose=False,
                         cols_standardize=None,
                         cols_leave=None,
                         cols_categorical=None,
                         cols_standardize_prior=None,
                         cols_leave_prior=None,
                         cols_categorical_prior=None,
                         net=None,
                         competing=False):
    """
  Do Cross Validation and select the best eta with only local data

  df_local: The local data
  eta_list: The etas that used for training
  model_prior: The prior model used for generating prior information
  time_intervals: time points for the discrete model
  hidden_nodes: The number of hidden nodes for each layer
  hidden_layers: The number of hidden layers
  batch_norm: Whether we use the batch normalization
  dropout: dropout rate
  learning_rate: learning rate
  batch_size: batch size
  optimizer: optimizer, wrapped by torchtuple
  epochs: epochs
  patience: The waiting steps used in earlystopping
  verbose: Whether you want to print out the logs for training
"""

    if parameter_set is None:
        parameter_set = {"hidden_nodes": 32, "hidden_layers": 2, "batch_norm": True,
                         "learning_rate": 0.01, "batch_size": 32, "dropout": 0.1,
                         "optimizer": tt.optim.Adam(), "alpha": 1}

    if metric not in ['C-index', 'IBS', 'INBLL']:
        raise ValueError("Please provide with a metric used to do hyperparameter tuning, which should be one of ["
                         "'C-index', 'IBS', 'INBLL']")

    df_test = df_local.sample(frac=0.2)
    df_train = df_local.drop(df_test.index)

    # if cols_standardize is None:
    #     cols_standardize = ['x1', 'x2', 'x3']
    mapper = mapper_generation(cols_standardize=cols_standardize, cols_leave=cols_leave,
                               cols_categorical=cols_categorical)
    _ = mapper.fit_transform(df_train).astype('float32')
    x_test = mapper.transform(df_test).astype('float32')

    # Difference: The mapper will differ in the columns, but applying on the same data
    # to adapt to the difference in the columns for prior and local model

    # if cols_standardize_prior is None:
    #     cols_standardize_prior = cols_standardize.copy()
    mapper_prior = mapper_generation(cols_standardize=cols_standardize_prior, cols_leave=cols_leave_prior,
                                     cols_categorical=cols_categorical_prior)
    _ = mapper_prior.fit_transform(df_train).astype('float32')
    x_test_prior = mapper_prior.transform(df_test).astype('float32')

    get_target = lambda df: (df['duration'].values, np.array(df['event'].values, dtype=np.float32))
    durations_test, events_test = get_target(df_test)

    # Begin create the CV sets
    kf = KFold(n_splits=n_splits)  # Define the split - into 2 folds
    kf.get_n_splits(df_train)  # returns the number of splitting iterations in the cross-validator

    best_eta = 0
    best_concordance = 0
    best_ibll = 10000000
    best_ibs = 10000000

    for eta in eta_list:
        concordance_td_list_CV = []
        integrated_brier_score_list_CV = []
        integrated_nbll_list_CV = []
        # likelihood_list_CV = []

        for train_index, test_index in kf.split(df_train):
            data_local_train_index = train_index
            data_local_val_index = test_index
            data_local_train = df_train.iloc[data_local_train_index,]
            data_local_val = df_train.iloc[data_local_val_index,]

            x_train = mapper.transform(data_local_train).astype('float32')
            x_val = mapper.transform(data_local_val).astype('float32')

            x_train_prior = mapper_prior.transform(data_local_train).astype('float32')
            x_val_prior = mapper_prior.transform(data_local_val).astype('float32')

            y_train = (x_train_prior, data_local_train['duration'].values, data_local_train['event'].values)
            y_val = (x_val_prior, data_local_val['duration'].values, data_local_val['event'].values)

            model, _ = model_generation(x_train, x_val, y_train, y_val, eta=eta, model_prior=model_prior,
                                        parameter_set=parameter_set, time_intervals=time_intervals,
                                        epochs=epochs, patience=patience, verbose=verbose, net=net, competing=competing)

            concordance_td, integrated_brier_score, integrated_nbll = evaluation_metrics(x_test, durations_test,
                                                                                         events_test,
                                                                                         model)

            concordance_td_list_CV.append(concordance_td)
            integrated_brier_score_list_CV.append(integrated_brier_score)
            integrated_nbll_list_CV.append(integrated_nbll)

        print("eta: ", eta)
        if metric == "C-index":
            print(concordance_td_list_CV)
            Concordance_index = sum(concordance_td_list_CV) / len(concordance_td_list_CV)
            print(Concordance_index)
            if best_concordance < Concordance_index:
                best_concordance = Concordance_index
                best_eta = eta
        elif metric == "IBS":
            print(integrated_brier_score_list_CV)
            integrated_brier_score = sum(integrated_brier_score_list_CV) / len(integrated_brier_score_list_CV)
            print(integrated_brier_score)
            if best_ibs > integrated_brier_score:
                best_ibs = integrated_brier_score
                best_eta = eta
        elif metric == "INBLL":
            print(integrated_nbll_list_CV)
            integrated_nbll = sum(integrated_nbll_list_CV) / len(integrated_nbll_list_CV)
            print(integrated_nbll)
            if best_ibll > integrated_nbll:
                best_ibll = integrated_nbll
                best_eta = eta

    print("CV ends")
    return best_eta, df_train, df_test, x_test, x_test_prior


def model_generation(x_train, x_val, y_train, y_val, with_prior=True, eta=None, model_prior=None,
                     parameter_set=None,
                     time_intervals=20,
                     epochs=512,
                     patience=5,
                     verbose=False,
                     option=None,
                     Model=None,
                     net=None,
                     competing=False):
    """
  Generate a model with or without the aid of prior information

  x_train, x_val, y_train, y_val: x and y data that used for training the model
  with_prior: whether you want to incorporate prior model. If False, eta and model_prior will not be required
  eta: The parameter eta used for combining prior and local information, this should be obtained after CV
  model_prior: The prior model used for generating prior information
  time_intervals: time points for the discrete model
  hidden_nodes: The number of hidden nodes for each layer
  hidden_layers: The number of hidden layers
  batch_norm: Whether we use the batch normalization
  dropout: dropout rate
  learning_rate: learning rate
  batch_size: batch size
  optimizer: optimizer, wrapped by torchtuple
  epochs: epochs
  patience: The waiting steps used in earlystopping
  verbose: Whether you want to print out the logs for training
  """

    if parameter_set is None:
        hidden_nodes = 32
        hidden_layers = 2
        batch_norm = True
        learning_rate = 0.01
        batch_size = 32
        dropout = 0.1
        optimizer = tt.optim.Adam()
        alpha = 1
        sigma = 0.1
    else:
        hidden_nodes = parameter_set["hidden_nodes"]
        hidden_layers = parameter_set['hidden_layers']
        batch_norm = parameter_set['batch_norm']
        learning_rate = parameter_set['learning_rate']
        batch_size = parameter_set['batch_size']
        dropout = parameter_set['dropout']
        optimizer = parameter_set['optimizer']
        alpha = parameter_set['alpha']
        sigma = parameter_set['sigma']

    if with_prior:
        if (eta is None) or (model_prior is None):
            raise ValueError("Please provide with eta and model_prior")

    val = tt.tuplefy(x_val, y_val)

    if net is None:
        in_features = x_train.shape[1]
        num_nodes = [hidden_nodes for _ in range(hidden_layers)]
        out_features = time_intervals
        if option is None:
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        else:
            net = Structure.MLPProportional(in_features, num_nodes, out_features, batch_norm, dropout, option=option)
    if Model is None:
        if competing is True:
            if with_prior:
                loss = NewlyDefinedLoss4(eta, model_prior)
                model = LogisticHazard(net, optimizer, loss=loss)
            else:
                loss = NewlyDefinedLoss3()
                model = LogisticHazard(net, optimizer, loss=loss)
        else:
            if with_prior:
                loss = NewlyDefinedLoss(eta, model_prior, time_intervals, option)
                model = LogisticHazard(net, optimizer, loss=loss)
            else:
                loss = NewlyDefinedLoss2(option)
                model = LogisticHazard(net, optimizer, loss=loss)
    else:
        if Model == "DeepHit":
            model = DeepHitSingle(net, optimizer, alpha=alpha, sigma=sigma)
        if Model == "DeepHitCompeting":
            model = DeepHit(net, optimizer, alpha=alpha, sigma=sigma)
        if Model == "PMF":
            model = PMF(net, optimizer)
        if Model == "MTLR":
            model = MTLR(net, optimizer)

    model.optimizer.set_lr(learning_rate)

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]

    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val)

    return model, log


def prior_model_generation(data,
                           parameter_set=None,
                           time_intervals=20,
                           epochs=512,
                           patience=5,
                           verbose=False,
                           cols_standardize=None,
                           cols_leave=None,
                           cols_categorical=None,
                           net=None,
                           competing=False):
    """
  Generate a model used for prior information.

  data: prior_data
  time_intervals: time points for the discrete model
  hidden_nodes: The number of hidden nodes for each layer
  hidden_layers: The number of hidden layers
  batch_norm: Whether we use the batch normalization
  dropout: dropout rate
  learning_rate: learning rate
  batch_size: batch size
  optimizer: optimizer, wrapped by torchtuple
  epochs: epochs
  patience: The waiting steps used in earlystopping
  verbose: Whether you want to print out the logs for training

  """

    if parameter_set is None:
        parameter_set = {"hidden_nodes": 32, "hidden_layers": 2, "batch_norm": True,
                         "learning_rate": 0.01, "batch_size": 32, "dropout": 0.1,
                         "optimizer": tt.optim.Adam(), "alpha": 1, "sigma": 0.1}

    data_prior_val = data.sample(frac=0.2)
    data_prior_train = data.drop(data_prior_val.index)
    # if cols_standardize is None:
    #     cols_standardize = ['x1', 'x2', 'x3']
    mapper = mapper_generation(cols_standardize=cols_standardize, cols_leave=cols_leave, cols_categorical=cols_categorical)
    x_train = mapper.fit_transform(data_prior_train).astype('float32')
    x_val = mapper.transform(data_prior_val).astype('float32')

    if competing is True:
        get_target = lambda df: (df['duration'].values, df['event'].values)
    else:
        get_target = lambda df: (df['duration'].values, np.array(df['event'].values, dtype=np.float32))
    y_train = get_target(data_prior_train)
    y_val = get_target(data_prior_val)

    model_prior, _ = model_generation(x_train, x_val, y_train, y_val, with_prior=False, parameter_set=parameter_set,
                                      verbose=verbose, time_intervals=time_intervals, epochs=epochs, patience=patience
                                      , net=net, competing=competing)

    return model_prior


def evaluation_metrics(x_test, durations_test, events_test, model):
    surv = model.predict_surv_df(x_test)
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    concordance_td = ev.concordance_td('antolini')
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)

    return concordance_td, integrated_brier_score, integrated_nbll
