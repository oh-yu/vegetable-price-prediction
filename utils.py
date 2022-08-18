import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VEGETABLES = [
    'だいこん', 'にんじん', 'キャベツ', 'レタス',
    'はくさい', 'こまつな', 'ほうれんそう', 'ねぎ',
    'きゅうり', 'トマト', 'ピーマン', 'じゃがいも',
    'なましいたけ', 'セルリー', 'そらまめ', 'ミニトマト'
]


class RMSPELoss:
    def __init__(self):
        pass

    def __call__(self, preds, ys):
        preds = preds.reshape(-1)
        ys = ys.reshape(-1)
        N = len(ys)
        losses = torch.zeros(N)
        for i, (pred, y) in enumerate(zip(preds, ys)):
            losses[i] = ((pred-y) / y)**2
        loss = torch.sum(losses)
        loss = 100 * ((loss/N)**0.5)
        return loss


def get_sorted_weather(train, temps):

    """
    Sort features extracted from the temperature-related data in the correspoding date and vegetable kind.
    Currently missiing values are interpolated by the mean of several areas in a given date.

    Parameters
    ----------
    train : pandas.DataFrame
        Two-dimensional array.
        Represents explanatory variables except for the temperature features and target for train.
        axis0 is the number of samples, axis1 is the features.

        train = pd.read_csv("./data/train.csv")
        train["date"] = pd.to_datetime(train["date"], format="%Y%m%d")

    temps : pandas.DataFrame
        Two-dimensional array.
        Represents the temperature-related data.
        axis0 is the number of samples and axis1 is the features.

        temps = pd.read_csv("./data/mapped_adjusted_weather.csv")
        temps["date"] = pd.to_datetime(temps["date"], format="%Y%m%d")
    """

    df = pd.DataFrame()
    c = 0
    for _, vals in train.iterrows():
        date = vals["date"]
        area = vals["area"]
        temp = temps[(temps.areas == area) & (temps.dates == date)]
        temp = temp.drop(columns=["dates", "areas"])
        if temp.empty:
            c += 1
            temp = pd.DataFrame(temps[temps.date == date].mean()).T
            # TODO: to be deleted
        df = pd.concat([df, temp], axis=0)
    print(f"missing: {c}")
    return df


def get_target_values(train, target_vegetable):

    """
    Extracts features corresponding to one target vegetable.
    Interpolates missing values in temperature related features.

    Parameters
    ----------
    train : pandas.DataFrame
        Two-dimensional array.
        Represents explanatory variables including temperature features, axis0 is the number of samples and axis1 is the features.

        train_test = pd.read_csv("./data/mapped_train_test.csv")
        train_test["date"] = pd.to_datetime(train_test["date"], format="%Y-%m-%d")
        weather = pd.read_csv("./data/sorted_mapped_adjusted_weather.csv")
        train_test = pd.concat([train_test, weather], axis=1)

        train_test["year"] = train_test.date.dt.year
        years = pd.get_dummies(train_test["year"])
        train_test = train_test.drop(columns="year")
        train_test = pd.concat([train_test, years], axis=1)

        train_test["month"] = train_test.date.dt.month
        months = pd.get_dummies(train_test["month"])
        train_test = train_test.drop(columns="month")
        train_test = pd.concat([train_test, months], axis=1)

        areas = pd.get_dummies(train_test["area"])
        train_test = train_test.drop(columns="area")
        train_test = pd.concat([train_test, areas], axis=1)

        train = train_test[:pd.read_csv("./data/train.csv").shape[0]]

    target_vegetable : string
        This string specify a target of vegetbles.
    """

    target_df = train[train.kind == target_vegetable].sort_values("date")
    interpolated_cols = ["mean_temp", "max_temp", "min_temp", "sum_rain", "mean_humid"]
    for col in interpolated_cols:
        target_df[col] = target_df[col].interpolate(limit=None, limit_direction='both').values
    target_df = target_df.drop(columns=["kind", "date"])
    target_values = target_df.values
    return target_values


def preprocess_data(target_values, train_size=4000, T=10, batch_size=16, continuous_feature_index=7):
    # 1. split data into training, test
    feature_size = target_values.shape[1]
    train = target_values[:train_size, :]
    test = target_values[train_size:, :]

    # 2. standardization
    ss = preprocessing.StandardScaler()
    ss.fit(train[:, :continuous_feature_index])
    train[:, :continuous_feature_index] = ss.transform(train[:, :continuous_feature_index])
    test[:, :continuous_feature_index] = ss.transform(test[:, :continuous_feature_index])

    # 3. reshape training data into shape of (N, T, D)
    if train.shape[0] % T == 0:
        train_N = train.shape[0] // T
    else:
        print("some samples in training data are ignored")
    train = train[:train_N * T]
    train = train.reshape(train_N, T, feature_size)
    train_x = train[:, :-1, :]
    train_y = train[:, 1:, :1]

    # 4. convert into torch.tensor, on DEVICE, DataLoader 
    train_x = torch.tensor(train_x, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(DEVICE)
    test_y = test[:, 0]
    test_y = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)
    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_y, train, test, ss


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=1, dropout_ratio=0.5, is_attention=False):
        super().__init__()
        self.is_attention = is_attention
        self.hidden_size = hidden_size
        self.fc1_size = int(hidden_size/2)*2 if is_attention else int(hidden_size/2)

        self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_size, int(hidden_size/2), batch_first=True)

        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc1 = nn.Linear(self.fc1_size, int(hidden_size/4))
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(int(hidden_size/4), int(hidden_size/10))
        self.fc3 = nn.Linear(int(hidden_size/10), output_size)

    def forward(self, x):
        self.train()
        out, (h_t1, c_t1) = self.rnn1(x)
        out, (h_t2, c_t2) = self.rnn2(out)
        
        if self.is_attention:
            contexts = get_contexts_by_selfattention(out, DEVICE)
            out = torch.cat((contexts, out), dim=2)

        out = self.dropout1(F.relu(self.fc1(out)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)

        self.out = out
        self.h_t1 = h_t1
        self.c_t1 = c_t1
        self.h_t2 = h_t2
        self.c_t2 = c_t2
        hidden_memory = {
            "h_t1": h_t1,
            "c_t1": c_t1,
            "h_t2": h_t2,
            "c_t2": c_t2,
        }
        return out, hidden_memory

    def predict(self, train, test, future):
        # assign hidden vector, memory cell, output from network
        out = self.out
        h_t1 = self.h_t1
        c_t1 = self.c_t1
        h_t2 = self.h_t2
        c_t2 = self.c_t2

        # prepare start_x
        # start_x shape: (1, 1, feature_size)
        # out shape: (N, T, 1)
        start_x0 = out[-1, -1, :].reshape(1, -1)
        start_x_other = torch.tensor(train[-1, -1, 1:].reshape(1, -1), dtype=torch.long).to(DEVICE)
        start_x = torch.cat((start_x0, start_x_other), axis=1)
        start_x = start_x.unsqueeze(1)

        # prepare h_t, c,t
        # h_t, c_t shape: (num_layers, N, H)
        h_t1, c_t1 = h_t1[:, -1, :].unsqueeze(1), c_t1[:, -1, :].unsqueeze(1)
        h_t2, c_t2 = h_t2[:, -1, :].unsqueeze(1), c_t2[:, -1, :].unsqueeze(1)

        # future prediction
        preds = torch.zeros(1, future, 1).to(DEVICE)
        hs = torch.zeros(1, future, int(self.hidden_size/2)).to(DEVICE) if self.is_attention else None

        self.eval()
        for t in range(future):
            pred, (h_t1, c_t1) = self.rnn1(start_x, (h_t1, c_t1))
            pred, (h_t2, c_t2) = self.rnn2(pred, (h_t2, c_t2))
            
            if self.is_attention:
                hs[:, t, :] = pred.squeeze(1)
                context = get_contexts_by_selfattention_during_prediction(t, pred, hs, DEVICE)
                pred = torch.cat((context, pred), dim=2)

            pred = self.dropout1(F.relu(self.fc1(pred)))
            pred = self.dropout2(F.relu(self.fc2(pred)))
            pred = self.fc3(pred).squeeze(1)
            preds[:, t, :] = pred

            start_x0 = pred
            start_x_other = torch.tensor(test[t, 1:].reshape(1, -1), dtype=torch.long).to(DEVICE)
            start_x = torch.cat((start_x0, start_x_other), axis=1)
            start_x = start_x.unsqueeze(0)
            # start_x shape: (1, 1, feature_size)
        return preds


def get_contexts_by_selfattention(hs, device):
    N, T, H = hs.shape
    contexts = torch.zeros(N, T, H).to(device)
    for t in range(T):
        h_t = hs[:, t, :].unsqueeze(1)
        h_t = h_t.repeat(1, t+1, 1)
        attention = (h_t*hs[:, :t+1, :]).sum(axis=2)
        attention = F.softmax(attention, dim=1)
        attention = attention.unsqueeze(2)
        attention = attention.repeat(1, 1, H)
        context = (attention*hs[:, :t+1, :]).sum(axis=1)
        contexts[:, t, :] = context
    return contexts


# TODO: refactor(this function mostly duplicates get_contexts_by_selfattention())
def get_contexts_by_selfattention_during_prediction(t, pred, hs, device):
    h_t = pred.repeat(1, t+1, 1)
    attention = (h_t*hs[:, :t+1, :]).sum(axis=2)
    attention = F.softmax(attention, dim=1)
    attention = attention.unsqueeze(2)
    attention = attention.repeat(1, 1, hs.shape[2])
    context = (attention*hs[:, :t+1, :]).sum(axis=1)
    context = context.unsqueeze(1)
    return context


def pipeline_rnn(train_loader, train, test, test_y, future=375, num_epochs=100, lr=0.005,
                 weight_decay=1e-3, eps=1e-8, hidden_size=500, dropout_ratio=0.5, is_attention=False):
    # Variable To Store Prediction
    train_losses = []

    # Instantiate Model, Optimizer, Criterion, EarlyStopping
    model = RNN(input_size=train.shape[2], hidden_size=hidden_size,
                dropout_ratio=dropout_ratio, is_attention=is_attention).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    criterion = nn.MSELoss()

    # Training & Test Loop
    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        idx = None

        for idx, (batch_x, batch_y) in enumerate(train_loader):
            # Forward
            out, _ = model(batch_x)
            loss = criterion(out, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update Params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # Add Training Loss
            running_loss += loss.item()
        train_losses.append(running_loss / (idx+1))

        # Test
        with torch.no_grad():
            model.eval()
            pred_y = model.predict(train, test, future)
            pred_y = pred_y.reshape(-1)
            loss = criterion(pred_y, test_y)

    return pred_y, loss


def plot_prediction(pred, test, ss):
    test[:, :7] = ss.inverse_transform(test[:, :7])
    pred[:, :7] = ss.inverse_transform(pred[:, :7])

    plt.title("pred vs test")
    plt.plot(test[:, 0], label="test")
    plt.plot(pred[:, 0], label="pred")
    plt.legend()
