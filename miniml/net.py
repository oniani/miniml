import miniml.tensor as T
import miniml.loss as L


class Model:
    def __init__(self):
        self._layers: list = []
        self._loss: list = []

    def add_layer(self, layer):
        """Add a network layer to the model."""

        self._layers.append(layer)

    def epoch(self, x, y, lr: float):
        """One full epoch."""

        # Forward pass through the network
        # NOTE: loop by index is needed for saving results
        forward: T.Tensor = x
        for idx, _ in enumerate(self._layers):
            forward = self._layers[idx].forward(x)
            x = forward

        # Compute loss and first gradient
        mse = L.MeanSquaredError(T.Ops.reshape(forward, y.shape), y)
        # print(forward.shape, y.shape)
        error = mse.forward()
        gradient = mse.backward()

        self._loss.append(error)

        # Backpropagation
        for idx, _ in reversed(list(enumerate(self._layers))):
            if self._layers[idx].type == "Linear":
                gradient, dW, db = self._layers[idx].backward(gradient)
                self._layers[idx].optimize(dW, db, lr)
            else:
                gradient = self._layers[idx].backward(gradient)

        return error

    def train(self, x_train, y_train, lr: float, epochs: int) -> None:
        """Train the model."""

        for epoch in range(epochs):
            loss = self.epoch(x_train, y_train, lr)

            if epoch % 25 == 0:
                print(f"Epoch: {epoch} | Loss: {loss}")

    def predict(self, x: T.Tensor) -> T.Tensor:
        """Predict by performing a forward pass on a trained network."""

        forward: T.Tensor = x
        for idx, _ in enumerate(self._layers):
            forward = self._layers[idx].forward(x)
            x = forward

        return forward
