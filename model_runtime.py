import re
import torch
import torch.nn as nn

class MessageGRU(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden
    
class DiscordTextGenerator:
    def __init__(self, checkpoint_path="discordModel.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            bundle = torch.load(checkpoint_path, map_location=self.device)

        self.stoi = bundle["stoi"]
        self.itos = bundle["itos"]
        self.pad_idx = bundle["pad_idx"]
        self.unk_idx = bundle["unk_idx"]
        self.seq_len = bundle.get("seq_len", 128)

        state = bundle["model_state_dict"]

        if "model_config" in bundle:
            cfg = bundle["model_config"]
            vocab_size = cfg["vocab_size"]
            embed_dim = cfg["embed_dim"]
            hidden_dim = cfg["hidden_dim"]
            num_layers = cfg["num_layers"]
            dropout = cfg.get("dropout", 0.2)
        else:
            # Infer old checkpoint config from tensor shapes, since I didn't save those in the model training
            vocab_size = bundle["vocab_size"]
            embed_dim = state["embedding.weight"].shape[1]
            hidden_dim = state["gru.weight_hh_l0"].shape[1]
            num_layers = sum(
                1 for key in state.keys()
                if key.startswith("gru.weight_ih_l")
            )
            dropout = 0.2

        self.model = MessageGRU(
            vocab_size=vocab_size,
            pad_idx=self.pad_idx,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_idx) for ch in text]

    @torch.no_grad()
    def generate(self, prompt="lol ", max_new_chars=300, temperature=0.8):
        prompt = prompt or "lol "
        temperature = max(0.1, min(float(temperature), 2.0))
        max_new_chars = max(1, min(int(max_new_chars), 1200))

        input_ids = self.encode(prompt)[-self.seq_len:]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        hidden = None
        logits, hidden = self.model(input_ids)

        current = input_ids[:, -1:]
        generated = []

        for _ in range(max_new_chars):
            logits, hidden = self.model(current, hidden)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            next_char = self.itos[next_id.item()]
            if next_char in ["<PAD>", "<UNK>"]:
                next_char = " "

            generated.append(next_char)
            current = next_id

        text = "".join(generated)

        # Remove the special sequences
        text = text.replace("<EOS>", "")
        text = text.replace("<CHANNEL_BREAK>", "")

        # Remove dangling partial tags like "<EO" or "<CHANNEL".
        text = re.sub(r"<[^>\n]{0,30}$", "", text)

        return text.strip()