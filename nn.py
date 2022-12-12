import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x) -> None:
        return self.model(x)


clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


def predict():
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    img = Image.open("img_3.jpg")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))


def train(epochs=10):
    for epoch in range(epochs):
        for batch in dataset:
            X, y = batch
            X, y = X.to(device), y.to(device)
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch} loss is {loss.item()}")

    with open("model_state.pt", "wb") as f:
        save(clf.state_dict(), f)


if __name__ == "__main__":
    train(10)
