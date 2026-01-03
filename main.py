import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = 100255

with open("training.txt", "r", encoding="utf-8", errors="ignore") as file:
    tdata = file.read()

class DeepPredictor(nn.Module):
    def __init__(self, vocab_size=vocab_size, hidden_size=512, num_layers=12, dropout=0.1):
        super(DeepPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x[:, -1, :]
        for layer in self.layers:
            x = layer(x) + x
        return self.output_layer(x)

model = DeepPredictor()
FILE = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if os.path.exists(FILE):
    model.load_state_dict(torch.load(FILE, map_location=device))
    model.eval()
    print("Model loaded from file.")
else:
    print("No saved model found.")

if input("Train model? (y/n): ") == "y":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    tokens = enc.encode(tdata)
    data_tensor = torch.tensor(tokens, dtype=torch.long).to(device)

    model.train()
    steps_per_epoch = 1
    for epoch in range(25):
        total_loss = 0
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(data_tensor) - 257, (256,))
            batch_X = torch.stack([data_tensor[i:i+256] for i in idx])
            batch_y = torch.stack([data_tensor[i+256] for i in idx])
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Average Loss: {total_loss / steps_per_epoch}")

    torch.save(model.state_dict(), FILE)
    print("Training complete. Model saved.")

def infer(data_list, temperature):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.LongTensor(data_list).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits / temperature, dim=-1)
        prediction = torch.multinomial(probs, num_samples=1)
    return prediction.item()

while True:
    user = input("> ")
    toks = enc.encode(user)
    if len(toks) > 256:
        toks = toks[-256:]
    while len(toks) < 256:
        toks.insert(0, 0)

    generated_indices = []
    line = ""
    i = 0
    while i < 2048:
        i += 1
        dynamic_temp = 0.5
        if len(generated_indices) > 100:
            dynamic_temp = ((100-len(set(generated_indices[-50:]))) / 250)
        print(round(dynamic_temp * 100) / 100, end="\r")
        out_tok = infer(toks, dynamic_temp)
        dec_tok = enc.decode([out_tok])
        if dec_tok == "\n":
            print(line)
            line = ""
        else:
            line = line + dec_tok
        generated_indices.append(out_tok)
        toks.pop(0)
        toks.append(out_tok)
