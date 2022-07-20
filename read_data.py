import requests
import zipfile
from data_simulation import SimStudyLinearPH, SimStudyNonLinearNonPH, SimStudyNonLinearPH

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
  support = support[["age", "sex", "race", "num.co", "diabetes", "dementia", "ca", "meanbp", "hrt", "resp", "temp", "wblc", "sod", "crea", "death", "d.time", "aps"]].dropna()

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

  support = support.drop(["age", "sex", "race", "num.co", "diabetes", "dementia", "ca", "meanbp", "hrt", "resp", "temp", "wblc", "sod", "crea", "death", "d.time", "aps"], axis = 1)
  support = support.reset_index(drop = True)

  return support

def metabric_data():
  from pycox.datasets import metabric
  return metabric.read_df()

def simulation_data(option = "non linear non ph", n = 300, grouping_number = 0):
  if (option == "non linear non ph"):
    Sim = SimStudyNonLinearNonPH()
  elif (option == "non linear ph"):
    Sim = SimStudyNonLinearPH()
  elif (option == "linear ph"):
    Sim = SimStudyLinearPH()
  else:
    raise ValueError("Please select a correct option")

  return Sim.simulate_df(n, grouping_number)
