import requests
import zipfile
import numpy as np
import pandas as pd
from data_simulation import SimStudyLinearPH, SimStudyNonLinearNonPH, SimStudyNonLinearPH
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
from random import sample

from pycox.models import LogisticHazard


def support_data():
    url = "https://hbiostat.org/data/repo/support2csv.zip"
    data_storage = "/content/drive/MyDrive/Kevin He/"
    path = data_storage + "a.zip"

    with requests.Session() as s:
        r = s.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(data_storage)

    support = pd.read_csv(data_storage + "support2.csv")
    support = support[
        ["age", "sex", "race", "num.co", "diabetes", "dementia", "ca", "meanbp", "hrt", "resp", "temp", "wblc", "sod",
         "crea", "death", "d.time", "aps"]].dropna()

    support["x0"] = support["age"]
    support["x1"] = support["sex"]
    support["x2"] = support["race"]
    support["x3"] = support["num.co"]
    support["x4"] = support["diabetes"]
    support["x5"] = support["dementia"]
    support["x6"] = support["ca"]
    support["x7"] = support["meanbp"]
    support["x8"] = support["hrt"]
    support["x9"] = support["resp"]
    support["x10"] = support["temp"]
    support["x11"] = support["wblc"]
    support["x12"] = support["sod"]
    support["x13"] = support["crea"]
    support["x14"] = support["aps"]
    support["duration"] = support["d.time"]
    support["event"] = support["death"]

    support = support.drop(
        ["age", "sex", "race", "num.co", "diabetes", "dementia", "ca", "meanbp", "hrt", "resp", "temp", "wblc", "sod",
         "crea", "death", "d.time", "aps"], axis=1)
    support = support.reset_index(drop=True)

    # creating instance of labelencoder
    labelencoder = LabelEncoder()  # Assigning numerical values and storing in another column
    support['x1'] = labelencoder.fit_transform(support['x1'])
    support['x2'] = labelencoder.fit_transform(support['x2'])
    support['x6'] = labelencoder.fit_transform(support['x6'])
    # support['x14'] = labelencoder.fit_transform(support['x14'])

    support = support.astype("float32")
    support = support.astype({"duration": "int64", "event": "int32"})

    return support


def metabric_data():
    from pycox.datasets import metabric
    return metabric.read_df()


def simulation_data(option="non linear non ph", n=300, grouping_number=0):
    if (option == "non linear non ph"):
        Sim = SimStudyNonLinearNonPH()
    elif (option == "non linear ph"):
        Sim = SimStudyNonLinearPH()
    elif (option == "linear ph"):
        Sim = SimStudyLinearPH()
    else:
        raise ValueError("Please select a correct option")

    return Sim.simulate_df(n, grouping_number)


def sim_event_times(mnist, max_time=365):
    digits = mnist.targets.numpy()
    betas = 365 * np.exp(-0.6 * digits) / np.log(1.2)
    event_times = np.random.exponential(betas)
    censored = event_times > max_time
    event_times[censored] = max_time
    return event_times, ~censored


def image_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist_train = datasets.MNIST('.', train=True, download=True,
                                 transform=transform)
    mnist_test = datasets.MNIST('.', train=False, transform=transform)

    sim_train = sim_event_times(mnist_train)
    sim_test = sim_event_times(mnist_test)

    list1 = [i for i in range(60000)]
    index = sample(list1, 30000)

    img = [mnist_train[i][0] for i in index]
    img = torch.stack(img)
    sim_train_subset = (sim_train[0][index], sim_train[1][index])

    labtrans = LogisticHazard.label_transform(20)
    target_train = labtrans.fit_transform(*sim_train_subset)

    return img, *target_train
