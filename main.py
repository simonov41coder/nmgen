"""
Character-level LSTM Name Generator
====================================
Usage:
  1. Prepare your dataset: a .txt file with one name per line
  2. Train:    python name_generator.py --train --data names.txt
  3. Generate: python name_generator.py --generate --count 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import json
import os

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = "name_model.pth"
CONFIG_PATH = "name_config.json"

HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
EPOCHS       = 1000
LR           = 0.003
BATCH_SIZE   = 64
SEQ_LEN      = 30   # max name length during training
TEMPERATURE  = 0.8  # higher = more creative, lower = more conservative

# ─── Model ─────────────────────────────────────────────────────────────────────

class NameLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(x), hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c

# ─── Data ──────────────────────────────────────────────────────────────────────

def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names

def build_vocab(names):
    chars = sorted(set("".join(names)))
    chars = ["<PAD>", "<SOS>", "<EOS>"] + chars
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def name_to_tensor(name, char2idx, max_len):
    sos = char2idx["<SOS>"]
    eos = char2idx["<EOS>"]
    pad = char2idx["<PAD>"]
    indices = [sos] + [char2idx.get(c, pad) for c in name] + [eos]
    indices = indices[:max_len + 2]
    return indices

def make_batches(names, char2idx, max_len, batch_size):
    encoded = [name_to_tensor(n, char2idx, max_len) for n in names]
    random.shuffle(encoded)
    batches = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i:i + batch_size]
        max_l = max(len(s) for s in batch)
        pad   = char2idx["<PAD>"]
        padded = [s + [pad] * (max_l - len(s)) for s in batch]
        t = torch.tensor(padded, dtype=torch.long)
        batches.append(t)
    return batches

# ─── Train ─────────────────────────────────────────────────────────────────────

def train(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Device: {device}")

    names = load_names(data_path)
    print(f"📝 Loaded {len(names)} names")

    char2idx, idx2char = build_vocab(names)
    vocab_size = len(char2idx)
    print(f"🔤 Vocab size: {vocab_size} chars")

    # Save config
    with open(CONFIG_PATH, "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": {str(k): v for k, v in idx2char.items()}}, f)

    model     = NameLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])

    print(f"\n🚀 Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        batches   = make_batches(names, char2idx, SEQ_LEN, BATCH_SIZE)
        total_loss = 0

        model.train()
        for batch in batches:
            batch = batch.to(device)
            x, y  = batch[:, :-1], batch[:, 1:]

            hidden = model.init_hidden(batch.size(0), device)
            optimizer.zero_grad()

            out, _ = model(x, hidden)
            # out: (B, T, vocab) → (B*T, vocab)
            loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)
        if epoch % 10 == 0:
            sample = generate_name(model, char2idx, idx2char, device)
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Sample: {sample}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

# ─── Generate ──────────────────────────────────────────────────────────────────

def generate_name(model, char2idx, idx2char, device, max_len=20, temp=TEMPERATURE):
    model.eval()
    with torch.no_grad():
        x      = torch.tensor([[char2idx["<SOS>"]]], dtype=torch.long).to(device)
        hidden = model.init_hidden(1, device)
        name   = []

        for _ in range(max_len):
            out, hidden = model(x, hidden)
            logits      = out[0, -1] / temp
            probs       = torch.softmax(logits, dim=-1)
            next_idx    = torch.multinomial(probs, 1).item()
            char        = idx2char[next_idx]

            if char == "<EOS>":
                break
            if char not in ("<PAD>", "<SOS>"):
                name.append(char)

            x = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return "".join(name).capitalize()

def generate(count, temp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CONFIG_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ No trained model found. Run --train first.")
        return

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    char2idx = cfg["char2idx"]
    idx2char = {int(k): v for k, v in cfg["idx2char"].items()}
    vocab_size = len(char2idx)

    model = NameLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print(f"\n✨ Generated {count} names (temp={temp}):\n")
    seen = set()
    attempts = 0
    while len(seen) < count and attempts < count * 10:
        name = generate_name(model, char2idx, idx2char, device, temp=temp)
        if name and name not in seen and len(name) > 1:
            seen.add(name)
            print(f"  {len(seen):2}. {name}")
        attempts += 1

# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Name Generator")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate names")
    parser.add_argument("--data",     type=str, default="names.txt", help="Path to dataset (.txt, one name per line)")
    parser.add_argument("--count",    type=int, default=10, help="Number of names to generate")
    parser.add_argument("--temp",     type=float, default=TEMPERATURE, help="Sampling temperature (0.5–1.2)")
    args = parser.parse_args()

    if args.train:
        train(args.data)
    elif args.generate:
        generate(args.count, args.temp)
    else:
        parser.print_help()

