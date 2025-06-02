# Neural Network

## Adapt previous example
- [ ] start with the example from exercise 03
- [ ] optimize number of layers, learning rate, and activation function
- [ ] replace the loss function with a neural network training
- [ ] start with the following simple CNN

```python
limits = {
    "num_layers": (2, 10),
    "activation": ("relu", "sigmoid", "tanh"),
    "lr": (0.01, 0.0001),
    "d_hidden": (2, 128),
    "batch_size": ("1", "2", "4", "8", "16", "32", "64", "128"),
}
```

```python
class CNN(nn.Module):
    def __init__(self, num_layers, activation, d_hidden):
        super().__init__()

        layers = []  # Set up the model architecture (depending on number of convolutional layers specified).
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            ),
        ]
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            )
            for _ in range(num_layers - 1)
        ]

        # NOTE due to padding output of final conv layer is 28*28*d_hidden
        self.fc = nn.Linear(in_features=28*28*d_hidden, out_features=10)
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```


## Monitor GPU utlization
```bash
srun --jobid <jobid> --interactive --pty /bin/bash
rocm-smi
```

# Useful links
- [https://propulate.readthedocs.io/](https://propulate.readthedocs.io/)
