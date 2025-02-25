import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MyLSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.Wf = np.random.randn(hidden_size, input_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size) * 0.01

        self.Uf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Uo = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Uc = np.random.randn(hidden_size, hidden_size) * 0.01

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

    def forward(self, x_t, h_prev, c_prev):

        f_t = sigmoid(np.dot(self.Wf, x_t) + np.dot(self.Uf, h_prev) + self.bf)
        i_t = sigmoid(np.dot(self.Wi, x_t) + np.dot(self.Ui, h_prev) + self.bi)
        o_t = sigmoid(np.dot(self.Wo, x_t) + np.dot(self.Uo, h_prev) + self.bo)
        c_hat_t = tanh(np.dot(self.Wc, x_t) + np.dot(self.Uc, h_prev) + self.bc)

        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * tanh(c_t)

        self.cache = (x_t, h_prev, c_prev, f_t, i_t, o_t, c_hat_t, c_t, h_t)
        return h_t, c_t

    def backward(self, dh_t, dc_t, lr):
        x_t, h_prev, c_prev, f_t, i_t, o_t, c_hat_t, c_t, h_t = self.cache

        do_t = dh_t * tanh(c_t) * d_sigmoid(o_t)
        dc_t += dh_t * o_t * d_tanh(tanh(c_t))
        di_t = dc_t * c_hat_t * d_sigmoid(i_t)
        df_t = dc_t * c_prev * d_sigmoid(f_t)
        dc_hat_t = dc_t * i_t * d_tanh(c_hat_t)

        dWf = np.dot(df_t, x_t.T)
        dUf = np.dot(df_t, h_prev.T)
        dbf = np.sum(df_t, axis=1, keepdims=True)

        dWi = np.dot(di_t, x_t.T)
        dUi = np.dot(di_t, h_prev.T)
        dbi = np.sum(di_t, axis=1, keepdims=True)

        dWo = np.dot(do_t, x_t.T)
        dUo = np.dot(do_t, h_prev.T)
        dbo = np.sum(do_t, axis=1, keepdims=True)

        dWc = np.dot(dc_hat_t, x_t.T)
        dUc = np.dot(dc_hat_t, h_prev.T)
        dbc = np.sum(dc_hat_t, axis=1, keepdims=True)

        dh_prev = np.dot(self.Wf.T, df_t) + np.dot(self.Wi.T, di_t) + np.dot(self.Wo.T, do_t) + np.dot(self.Wc.T,
                                                                                                       dc_hat_t)
        dc_prev = dc_t * f_t

        self.Wf -= lr * dWf
        self.Uf -= lr * dUf
        self.bf -= lr * dbf

        self.Wi -= lr * dWi
        self.Ui -= lr * dUi
        self.bi -= lr * dbi

        self.Wo -= lr * dWo
        self.Uo -= lr * dUo
        self.bo -= lr * dbo

        self.Wc -= lr * dWc
        self.Uc -= lr * dUc
        self.bc -= lr * dbc

        return dh_prev, dc_prev


class Encoder:
    def __init__(self, input_size, hidden_size, num_layers=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layers = [MyLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for index in inputs:
            x_t = np.zeros((self.input_size, 1))
            x_t[index] = 1
            for layer in self.lstm_layers:
                h, c = layer.forward(x_t, h, c)
                x_t = h
        return h, c

    def backward(self, dh, dc, lr):
        for layer in reversed(self.lstm_layers):
            dh, dc = layer.backward(dh, dc, lr)
        return dh, dc


class Decoder:
    def __init__(self, hidden_size, output_size, num_layers=4):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm_layers = [MyLSTMCell(hidden_size, hidden_size) for _ in range(num_layers)]
        self.Wo = np.random.randn(output_size, hidden_size) * 0.01
        self.bo = np.zeros((output_size, 1))

    def forward(self, h, target_length):
        outputs = []
        x_t = h

        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        for _ in range(target_length):
            for layer in self.lstm_layers:
                h, c = layer.forward(x_t, h, c)
                x_t = h
            output = softmax(np.dot(self.Wo, h) + self.bo)
            outputs.append(output)
        return outputs

    def backward(self, dh, dc, lr):
        for layer in reversed(self.lstm_layers):
            dh, dc = layer.backward(dh, dc, lr)
        return dh, dc


class MyAttentionRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def train(self, inputs, targets, lr):
        h, c = self.encoder.forward(inputs)
        outputs = self.decoder.forward(h, len(targets))
        loss = sum(-np.log(softmax(y)[targets[i]] + 1e-7) for i, y in enumerate(outputs))
        self.backward(outputs, targets, lr)
        return loss

    def backward(self, outputs, targets, lr):
        dh, dc = np.zeros((self.encoder.hidden_size, 1)), np.zeros((self.encoder.hidden_size, 1))
        self.decoder.backward(dh, dc, lr)
        self.encoder.backward(dh, dc, lr)

    def generate(self, h, max_sequence_length=25):
        y = self.decoder.forward(h, max_sequence_length)
        return np.argmax(y, axis=0).flatten()
