import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=42):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._rng = np.random.default_rng(seed=seed)
        self.W = self._rng.normal(size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.b = np.zeros(self.out_channels)

    def forward(self, X: np.ndarray):
        # X NCHW
        if len(X.shape) != 4:
            raise ValueError("X must be NCHW")
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        batch, in_channels, in_h, in_w = X_padded.shape

        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, self.out_channels, out_h, out_w))

        for b in range(batch):
            for channel in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = self.stride * h
                        w_start = self.stride * w
                        window = X_padded[b, :, 
                                          h_start: h_start + self.kernel_size, 
                                          w_start: w_start + self.kernel_size]
                        out[b, channel, h, w] = np.sum(window * self.W[channel]) + self.b[channel]
        
        self._cache_X_padded = X_padded
        return out
        

    def backward(self, dout):
        """dout is the upstream gradient"""
        X_padded = self._cache_X_padded
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(X_padded)

        for co in range(self.out_channels):
            db[co] = np.sum(dout[:, co, :, :])

        
        batch, in_channels, in_h, in_w = X_padded.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        for b in range(batch):
            for channel in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = self.stride * h
                        w_start = self.stride * w
                        window = X_padded[b, :, 
                                          h_start: h_start + self.kernel_size, 
                                          w_start: w_start + self.kernel_size]
                        dW[channel] += window * dout[b, channel, h, w]

                        dX_padded[b, :,
                                  h_start: h_start + self.kernel_size,
                                  w_start: w_start + self.kernel_size
                                  ] += self.W[channel] * dout[b, channel, h, w]
                        
        if self.padding == 0:
            dX = dX_padded
        else:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dX, dW, db
    
class MaxPool2D:
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
         # X NCHW
        if len(X.shape) != 4:
            raise ValueError("X must be NCHW")
        
        batch, in_channels, in_h, in_w = X.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        out = np.zeros((batch, in_channels, out_h, out_w))
        argmax_row = np.zeros((batch, in_channels, out_h, out_w), dtype=int)
        argmax_col = np.zeros((batch, in_channels, out_h, out_w), dtype=int)

        for b in range(batch):
            for channel in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = self.stride * h
                        w_start = self.stride * w
                        window = X[b, channel, 
                                    h_start: h_start + self.kernel_size, 
                                    w_start: w_start + self.kernel_size]
                        
                        out[b, channel, h, w] = np.max(window)
                        max_idx = np.argmax(window)
                        row, col = divmod(max_idx, self.kernel_size) # works only for square kernel
                        argmax_row[b, channel, h, w] = row
                        argmax_col[b, channel, h, w] = col

        self._argmax_row = argmax_row
        self._argmax_col = argmax_col
        self._X_shape = X.shape

        return out
    
    def backward(self, dout):
        dX = np.zeros(self._X_shape)
        batch, in_channels, in_h, in_w = self._X_shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        for b in range(batch):
            for channel in range(in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = self.stride * h
                        w_start = self.stride * w
                        row = self._argmax_row[b, channel, h, w]
                        col = self._argmax_col[b, channel, h, w]
                        dX[b, channel, h_start + row, w_start + col] += dout[b, channel, h, w]

        return dX


class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        self.momentum = momentum
        self.eps = eps
        self.cache = {}

    def forward(self, X, training=True):
        # X (NCHW)
        N, C, H, W = X.shape

        if training:
            batch_mean = np.mean(X, axis=(0, 2, 3))  # shape (C,)
            batch_var = np.mean((X - batch_mean.reshape(1, C, 1, 1))**2, axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            use_mean, use_var = batch_mean, batch_var

        else:
            use_mean, use_var = self.running_mean, self.running_var

        X_hat = (X - use_mean.reshape(1, C, 1, 1)) / np.sqrt(use_var.reshape(1, C, 1, 1) + self.eps)
        out = self.gamma.reshape(1, C, 1, 1) * X_hat + self.beta.reshape(1, C, 1, 1)

        if training:
            self.cache['X'] = X
            self.cache['X_hat'] = X_hat
            self.cache['use_mean'] = use_mean
            self.cache['use_var'] = use_var

        return out

    def backward(self, dout):
        if not self.cache:
            raise RuntimeError("Cache is empty")
        X = self.cache['X']
        X_hat = self.cache['X_hat']
        use_mean = self.cache['use_mean']
        use_var = self.cache['use_var']

        N, C, H, W = X.shape
        m = N * H * W
        dgamma = np.sum(dout * X_hat, axis=(0, 2, 3))  # (C,)
        dbeta = np.sum(dout, axis=(0, 2, 3))  # (C,)

        dx_hat = dout * self.gamma.reshape(1, C, 1, 1)

        std_inv = 1 / np.sqrt(use_var.reshape(1, C, 1, 1) + self.eps)
        dvar = np.sum(dx_hat * (X - use_mean.reshape(1, C, 1, 1)) *
                      -0.5 * std_inv**3, axis=(0, 2, 3))
        dmean = (np.sum(dx_hat * -std_inv, axis=(0, 2, 3)) + 
                 dvar * np.mean(-2*(X - use_mean.reshape(1, C, 1, 1)), axis=(0, 2, 3)))

        dX = (dx_hat * std_inv + 
              dvar.reshape(1, C, 1, 1) * 2 * (X - use_mean.reshape(1, C, 1, 1)) / m +
              dmean.reshape(1, C, 1, 1) / m)

        return dX, dgamma, dbeta

class Dense:
    def __init__(self, in_features, out_features, seed=42):
        self.W = np.random.default_rng(seed).normal(size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.cache = {}

    def forward(self, X):
        self.cache['X'] = X
        return np.dot(X, self.W) + self.b

    def backward(self, dout):
        X = self.cache['X']
        dW = np.dot(X.T, dout)
        db = np.sum(dout, axis=0)
        dX = np.dot(dout, self.W.T)
        return dX, dW, db

class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = {}

    def forward(self, logits, y):
        self.cache['y'] = y
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)

        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        log_probs = np.log(probs[np.arange(y.shape[0]), y])
        loss = -np.mean(log_probs)
        self.cache['probs'] = probs
        return loss

    def backward(self):
        y = self.cache['y']
        probs = self.cache['probs']
        dlogits = probs.copy()
        dlogits[np.arange(y.shape[0]), y] -= 1
        dlogits /= y.shape[0]
        return dlogits


class ReLU:
    def forward(self, X):
        self._mask = X > 0
        return X * self._mask

    def backward(self, dout):
        return dout * self._mask


class Flatten:
    def forward(self, X):
        self._input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._input_shape)


class LeNet:
    def __init__(self, num_classes=10, seed=42):
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, seed=seed)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, seed=seed + 1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Dense(in_features=16 * 5 * 5, out_features=120, seed=seed + 2)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_features=120, out_features=84, seed=seed + 3)
        self.relu4 = ReLU()
        self.fc3 = Dense(in_features=84, out_features=num_classes, seed=seed + 4)

        self.loss_fn = SoftmaxCrossEntropy()

        self._param_layers = {
            "conv1": self.conv1, "conv2": self.conv2,
            "fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3,
        }

        # He initialization
        for layer in (self.conv1, self.conv2):
            fan_in = layer.in_channels * layer.kernel_size ** 2
            layer.W *= np.sqrt(2.0 / fan_in)
        for layer in (self.fc1, self.fc2, self.fc3):
            fan_in = layer.W.shape[0]
            layer.W *= np.sqrt(2.0 / fan_in)

    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        out = self.relu4.forward(out)
        logits = self.fc3.forward(out)
        return logits

    def backward(self, dlogits):
        dout, dW_fc3, db_fc3 = self.fc3.backward(dlogits)
        dout = self.relu4.backward(dout)
        dout, dW_fc2, db_fc2 = self.fc2.backward(dout)
        dout = self.relu3.backward(dout)
        dout, dW_fc1, db_fc1 = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout, dW_conv2, db_conv2 = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dX, dW_conv1, db_conv1 = self.conv1.backward(dout)

        grads = {
            "conv1": (dW_conv1, db_conv1),
            "conv2": (dW_conv2, db_conv2),
            "fc1": (dW_fc1, db_fc1),
            "fc2": (dW_fc2, db_fc2),
            "fc3": (dW_fc3, db_fc3),
        }
        return dX, grads

    def train_step(self, X, y, lr):
        logits = self.forward(X)
        loss = self.loss_fn.forward(logits, y)
        dlogits = self.loss_fn.backward()
        _, grads = self.backward(dlogits)

        for name, layer in self._param_layers.items():
            dW, db = grads[name]
            layer.W -= lr * dW
            layer.b -= lr * db

        return loss

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
