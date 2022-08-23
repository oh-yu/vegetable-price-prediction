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
CONTINUOUS_FEATURE_INDEX = 7


def get_sorted_weather(train, temps):
    """
    Sort features extracted from the temperature-related data,
    in the correspoding date and vegetable kind.
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
    for _, vals in train.iterrows():
        date = vals["date"]
        area = vals["area"]
        temp = temps[(temps.areas == area) & (temps.dates == date)]
        temp = temp.drop(columns=["dates", "areas"])
        df = pd.concat([df, temp], axis=0)
    return df


def get_target_values(train, target_vegetable):
    """
    Extracts features corresponding to one target vegetable.
    Interpolates missing values in temperature related features.

    Parameters
    ----------
    train : pandas.DataFrame
        Two-dimensional array.
        Represents explanatory variables including temperature features,
        axis0 is the number of samples and axis1 is the features.

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


def preprocess_data(target_values, train_size=4000,
                    T=10, batch_size=16):
    # 1. split data into training, test
    feature_size = target_values.shape[1]
    train = target_values[:train_size, :]
    test = target_values[train_size:, :]

    # 2. standardization
    ss = preprocessing.StandardScaler()
    ss.fit(train[:, :CONTINUOUS_FEATURE_INDEX])
    train[:, :CONTINUOUS_FEATURE_INDEX] = ss.transform(train[:, :CONTINUOUS_FEATURE_INDEX])
    test[:, :CONTINUOUS_FEATURE_INDEX] = ss.transform(test[:, :CONTINUOUS_FEATURE_INDEX])

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
    # TODO: undersntad not-callable
    train_y = torch.tensor(train_y, dtype=torch.float32).to(DEVICE)
    test_y = test[:, 0]
    test_y = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)
    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_y, train, test, ss


class RNN(nn.Module):
    # pylint: disable=too-many-instance-attributes
    # It seems reasonable in this case, because LSTM-Attention needs all of that.
    # pylint: disable=too-many-arguments
    # It seems reasonable in this case, because of the same reason.

    def __init__(self, input_size, hidden_size=500,
                 output_size=1, dropout_ratio=0.5, is_attention=False):
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

        # these variables should be included in RNN, used for prediction
        self.out = None
        self.h_t1 = None
        self.c_t1 = None
        self.h_t2 = None
        self.c_t2 = None

    def forward(self, x):
        # x shape: (N, T, D)
        out, (h_t1, c_t1) = self.rnn1(x)
        out, (h_t2, c_t2) = self.rnn2(out)
        # out shape: (N, T, int(hidden_size/2))

        if self.is_attention:
            contexts = get_contexts_by_attention(out, DEVICE)
            out = torch.cat((contexts, out), dim=2)
            # out shape: (N, T, int(hidden_size/2)*2)

        out = self.dropout1(F.relu(self.fc1(out)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        # out shape: (N, T, 1)

        self.out = out
        self.h_t1 = h_t1
        self.c_t1 = c_t1
        self.h_t2 = h_t2
        self.c_t2 = c_t2
        return out

    def predict(self, train, test, future):
        # pylint: disable=too-many-locals
        # It seems reasonable in this case, because LSTM-Attention's prediction needs all of that.

        """
        LSTM-Attention predicts value for future time steps,
        following the tail end of the training data.

        Parameters
        ----------
        train : pandas.DataFrame
            Two-dimensional array.
            Represents explanatory variables for training including temperature features,
            axis0 is the number of samples and axis1 is the features.

            train_loader, test_y, train, test, ss =
            utils.preprocess_data(target_values, train_size=training_size,
                                  T=param["T"], batch_size=param["batch_size"])
        test : pandas.DataFrame
            Two-dimensional array.
            Represents explanatory variables for testing including temperature features,
            axis0 is the number of samples and axis1 is the features.

            train_loader, test_y, train, test, ss =
            utils.preprocess_data(target_values, train_size=training_size,
                                  T=param["T"], batch_size=param["batch_size"])
        future : int
            This specifies how long model should predict.
        """
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
        hs = torch.zeros(1, future, int(self.hidden_size/2)).to(DEVICE) \
            if self.is_attention else None
        self.eval()

        for t in range(future):
            pred, (h_t1, c_t1) = self.rnn1(start_x, (h_t1, c_t1))
            pred, (h_t2, c_t2) = self.rnn2(pred, (h_t2, c_t2))

            if self.is_attention:
                hs[:, t, :] = pred.squeeze(1)
                # TODO: understand unsupported-assignment-operation
                context = get_contexts_by_attention_during_prediction(t, pred, hs)
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


def get_contexts_by_attention(hs, device):
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


# TODO: refactor(this function mostly duplicates get_contexts_by_attention())
def get_contexts_by_attention_during_prediction(t, pred, hs):
    h_t = pred.repeat(1, t+1, 1)
    attention = (h_t*hs[:, :t+1, :]).sum(axis=2)
    attention = F.softmax(attention, dim=1)
    attention = attention.unsqueeze(2)
    attention = attention.repeat(1, 1, hs.shape[2])
    context = (attention*hs[:, :t+1, :]).sum(axis=1)
    context = context.unsqueeze(1)
    return context


def pipeline_rnn(train_loader, train, test, param, future=375):
    # pylint: disable=too-many-locals
    # It seems reasonable in this case, because pipeline needs all of that.

    # Instantiate Model, Optimizer, Criterion
    model = RNN(input_size=train.shape[2], hidden_size=param["hidden_size"],
                dropout_ratio=param["dropout_ratio"], is_attention=param["is_attention"]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=param["learning_rate"],
                           weight_decay=param["weight_decay"], eps=param["eps"])
    criterion = nn.MSELoss()

    # Training & Test Loop
    for _ in range(param["num_epochs"]):
        model.train()

        for _, (batch_x, batch_y) in enumerate(train_loader):
            # Forward
            out = model(batch_x)
            loss = criterion(out, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update Params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # Test
        with torch.no_grad():
            model.eval()
            pred_y = model.predict(train, test, future)
            pred_y = pred_y.reshape(-1)

    return pred_y


def plot_prediction(pred, test, ss):
    test[:, :CONTINUOUS_FEATURE_INDEX] = ss.inverse_transform(test[:, :CONTINUOUS_FEATURE_INDEX])
    pred[:, :CONTINUOUS_FEATURE_INDEX] = ss.inverse_transform(pred[:, :CONTINUOUS_FEATURE_INDEX])

    plt.title("pred vs test")
    plt.plot(test[:, 0], label="test")
    plt.plot(pred[:, 0], label="pred")
    plt.legend()
