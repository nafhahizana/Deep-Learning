code:

import torch
import torch.nn as nn

sentences = [["I", "love", "NLP"], ["He", "plays", "football"]]
tags = [["PRON", "VERB", "NOUN"], ["PRON", "VERB", "NOUN"]]

word2idx = {w: i+2 for i, w in enumerate(set(w for s in sentences for w in s))}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

tag2idx = {t: i+1 for i, t in enumerate(set(t for ts in tags for t in ts))}
tag2idx["<PAD>"] = 0

max_len = max(len(s) for s in sentences)

def encode(seq, vocab, is_tag=False):
    if is_tag:
        # Tags don't have <UNK>, so map directly and pad
        idxs = [vocab[w] for w in seq]
    else:
        idxs = [vocab.get(w, vocab["<UNK>"]) for w in seq]
    return idxs + [vocab["<PAD>"]] * (max_len - len(seq))

X = torch.tensor([encode(s, word2idx) for s in sentences])
Y = torch.tensor([encode(t, tag2idx, is_tag=True) for t in tags])

class POSModel(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = POSModel(len(word2idx), len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    output = model(X)  # output shape: (batch_size, seq_len, tag_size)
    loss = criterion(output.view(-1, len(tag2idx)), Y.view(-1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def predict(sentence):
    model.eval()
    x = torch.tensor([encode(sentence, word2idx)])
    with torch.no_grad():
        preds = model(x).argmax(dim=2).squeeze().tolist()[:len(sentence)]
    idx2tag = {v: k for k, v in tag2idx.items()}
    return [idx2tag[i] for i in preds]

for sent in sentences:
    print(f"Sentence: {sent}")
    print(f"Predicted tags: {predict(sent)}")

    output:

    <img width="349" height="141" alt="Screenshot 2025-10-08 110113" src="https://github.com/user-attachments/assets/722d4613-b23a-4ea0-b48a-72e8c994db91" />
