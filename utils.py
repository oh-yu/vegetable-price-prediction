import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn import preprocessing


def get_terminal_score():
    target_vegetables = [
        'だいこん', 'にんじん', 'キャベツ', 'レタス',
        'はくさい', 'こまつな', 'ほうれんそう', 'ねぎ',
        'きゅうり', 'トマト', 'ピーマン', 'じゃがいも',
        'なましいたけ', 'セルリー', 'そらまめ', 'ミニトマト'
    ]
    scores = []
    
    # Load Train
    train = pd.read_csv("./data/train.csv")
    train["date"] = pd.to_datetime(train["date"], format="%Y%m%d")

    train["year"] = train.date.dt.year
    years = pd.get_dummies(train["year"])
    train = train.drop(columns="year")
    train = pd.concat([train, years], axis=1)

    train["month"] = train.date.dt.month
    months = pd.get_dummies(train["month"])
    train = train.drop(columns="month")
    train = pd.concat([train, months], axis=1)

    train["weekday"] = train.date.dt.weekday
    weekdays = pd.get_dummies(train["weekday"])
    train = train.drop(columns="weekday")
    train = pd.concat([train, weekdays], axis=1)

    areas = pd.get_dummies(train["area"])
    train = train.drop(columns="area")
    train_df = pd.concat([train, areas], axis=1)
    
    # Get Score For Each Vegetable
    for target in target_vegetables:
        # Set Train Size 
        if target == "レタス":
            train_size = 1000
        elif target == "なましいたけ":
            train_size = 3000
        elif target == "セルリー":
            train_size = 2000
        elif target == "そらまめ":
            train_size = 800
        elif target == "ミニトマト":
            train_size = 1500
        else:
            train_size = 4000
        
        # Preprocess Data
        target_values = get_target_values(train_df, target)
        train_x, train_y, test_y, train, test, ss = preprocess_data(target_values, train_size=train_size, T=10)
        # Training, Test
        _, loss = pipeline_rnn(train_x, train_y, train, test, test_y,
                               future=target_values.shape[0]-train_size, num_epochs=30)
        scores.append(loss)
        print(f"{target}: {loss}")
    
    # Log
    print(f"MSE: {np.mean(scores)}({np.std(scores)})")


def get_temp_features(train, temps):
    """ definition of train, temps
    train = pd.read_csv("./data/train.csv")
    train["date"] = pd.to_datetime(train["date"], format="%Y%m%d")
    temps = pd.read_csv("./data/weather.csv")
    temps["date"] = pd.to_datetime(temps["date"], format="%Y%m%d")
    """
    df = pd.DataFrame()
    for row,vals in train.iterrows():
        date = vals["date"]
        area = vals["area"]
        temp = temps[(temps.area == area) & (temps.date == date)]
        temp = temp.drop(columns=["date", "area"])
        if temp.empty:
            temp = pd.DataFrame(temps[temps.date == date].mean()).T
            # TODO: create more reasonable logic
        df = pd.concat([df, temp], axis=0)
    return df


def get_target_values(train, target_str):
    target_df = train[train.kind == target_str].sort_values("date")
    target_df = target_df.drop(columns=["kind", "date"])
    
    # change the order for amount, mode_price
    tmp = target_df["amount"]
    target_df["amount"] = target_df["mode_price"]
    target_df["mode_price"] = tmp
    
    target_values = target_df.values
    return target_values


""" Corresponds To 1_variable_rnn_submit.ipynb
def preprocess_data(train, T=10):
    feature_size = train.shape[1]
    
    ss = preprocessing.StandardScaler()
    ss.fit(train[:, 0].reshape(-1, 1))
    train[:, 0] = ss.transform(train[:, 0].reshape(-1, 1)).reshape(-1)

    train_N = train.shape[0] // T
    train = train[:train_N * T]
    train = train.reshape(train_N, T, feature_size)
    train_x = train[:, :-1, :]
    train_y = train[:, 1:, :1]
    
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    return train_x, train_y, train, ss
"""

# Corresponds To 1_variable_rnn.ipynb
def preprocess_data(target_values, train_size=4000, T=10):
    feature_size = target_values.shape[1]
    
    train = target_values[:train_size, :]
    test = target_values[train_size:, :]

    ss = preprocessing.StandardScaler()
    ss.fit(train[:, :2])
    train[:, :2] = ss.transform(train[:, :2])
    test[:, :2] = ss.transform(test[:, :2])

    train_N = train.shape[0] // T
    train = train[:train_N * T]
    train = train.reshape(train_N, T, feature_size)
    train_x = train[:, :-1, :]
    train_y = train[:, 1:, :1]
    
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_y = test[:, 0]
    test_y = torch.tensor(test_y, dtype=torch.float32)
    return train_x, train_y, test_y, train, test, ss


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=1, dropout_ratio=0.5):
        super().__init__()
        self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_size, int(hidden_size/2), batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc1 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(int(hidden_size/4), int(hidden_size/10))

        self.fc3 = nn.Linear(int(hidden_size/10), output_size)

    def forward(self, x, train, test, future):
        out, (h_t1, c_t1) = self.rnn1(x)
        out, (h_t2, c_t2) = self.rnn2(out)
        out = self.dropout1(F.relu(self.fc1(out)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        
        # prepare start_x
        # start_x shape: (1, 1, feature_size)
        # out shape: (N, T, 1)
        preds = torch.zeros(1, future, 1)
        start_x0 = out[-1, -1, :].reshape(1, -1)
        start_x_other = torch.tensor(train[-1, -1, 1:].reshape(1, -1), dtype=torch.long)
        start_x = torch.cat((start_x0, start_x_other), axis=1)
        start_x = start_x.unsqueeze(1)
        
        # prepare h_t, c,t
        # h_t, c_t shape: (num_layers, N, H)
        h_t1, c_t1 = h_t1[:, -1, :].unsqueeze(1), c_t1[:, -1, :].unsqueeze(1)
        h_t2, c_t2 = h_t2[:, -1, :].unsqueeze(1), c_t2[:, -1, :].unsqueeze(1)
        
        # future prediction
        for t in range(future):
            pred, (h_t1, c_t1) = self.rnn1(start_x, (h_t1, c_t1))
            pred, (h_t2, c_t2) = self.rnn2(pred, (h_t2, c_t2))
            pred = self.dropout1(F.relu(self.fc1(pred)))
            pred = self.dropout2(F.relu(self.fc2(pred)))
            pred = self.fc3(pred).squeeze(1)
            preds[:, t, :] = pred

            start_x0 = pred
            start_x_other = torch.tensor(test[t, 1:].reshape(1, -1), dtype=torch.long)
            start_x = torch.cat((start_x0, start_x_other), axis=1)
            start_x = start_x.unsqueeze(0)
            # start_x shape: (1, 1, feature_size)
        return out, preds


""" Corresponds To 1_variable_rnn_submit.ipynb
def pipeline_rnn(train_x, train_y, train, test, future=375, num_epochs=100):
    # Instantiate Model, Optimizer, Criterion
    model = rnn(input_size = train_x.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    # Training & Test Loop
    for epoch in range(num_epochs):

        # Training
        optimizer.zero_grad()
        out, pred_y = model(train_x, train, test, future)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"training loss = {loss}")

    return pred_y


def plot_prediction(pred_y, test_y, ss):
    pred_y = pred_y.detach().numpy()

    test_y = test_y.reshape(-1, 1)
    test_y = ss.inverse_transform(test_y)
    pred_y = pred_y.reshape(-1, 1)
    pred_y = ss.inverse_transform(pred_y)

    plt.title("pred vs test")
    plt.plot(test_y, label="test")
    plt.plot(pred_y, label="pred")
    plt.legend()
"""


# Corresponds To 1_variable_rnn.ipynb
def pipeline_rnn(train_x, train_y, train, test, test_y, future=375, num_epochs=100):
    # Instantiate Model, Optimizer, Criterion
    model = RNN(input_size = train_x.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    # Training & Test Loop
    for epoch in range(num_epochs):

        # Training
        optimizer.zero_grad()
        out, pred_y = model(train_x, train, test, future)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"training loss = {loss}")

        # Test
        with torch.no_grad():
            model.eval()
            pred_y = pred_y.reshape(-1)
            loss = criterion(pred_y, test_y)

            if epoch % 10 == 0:
                print(f"test loss = {loss}")
    return pred_y, loss


def plot_prediction(pred_y, test, ss):
    pred_y = pred_y.detach().numpy()

    test[:, :2] = ss.inverse_transform(test[:, :2])
    
    pred_y = pred_y.reshape(-1, 1)
    pred_y = np.concatenate([pred_y, test[:, 1:]], axis=1)
    pred_y[:, :2] = ss.inverse_transform(pred_y[:, :2])

    plt.title("pred vs test")
    plt.plot(test[:, 0], label="test")
    plt.plot(pred_y[:, 0], label="pred")
    plt.legend()
