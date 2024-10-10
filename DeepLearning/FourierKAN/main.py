import torch
import model
from dataset import get_Dataset
import pandas as pd
import numpy as np

train_x, train_y, test_x, test_y = get_Dataset()

models = {
    "relu" : model.MLP().to("cuda"),
    "siren" : model.MLP(act = model.SIREN()).to("cuda"),
    "snake" : model.MLP(act = model.SNAKE()).to("cuda"),
    "finer" : model.MLP(act = model.FINER()).to("cuda"),
    "fkan" : model.FKAN(latent_dim = 8).to("cuda"),
}

log_loss = []

sample_x1, sample_y1, sample_x2, sample_y2 = get_Dataset(shuffle = False)
sample_x = torch.cat((sample_x1, sample_x2))
sample_y = torch.cat((sample_y1, sample_y2))
results = [sample_x.cpu().numpy()[:,0], sample_y.cpu().numpy()[:,0]]
for key in models.keys():
    loss = models[key].train(train_x, train_y, test_x, test_y)
    log_loss.append(loss)
    results.append(models[key].prediction(sample_x))
log_loss = np.stack(log_loss, axis = 1)
results = np.stack(results, axis = 1)

log_loss = pd.DataFrame(log_loss, columns = models.keys())
log_loss.to_csv("loss.csv")
results = pd.DataFrame(results, columns = ["x", "y"] + list(models.keys()))
results.to_csv("prediction.csv")