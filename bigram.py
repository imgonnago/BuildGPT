import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from chatacters to intefers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda i: ''.join([itos[i] for i in i]) # decoder: take a list of integers, output a string

#train and test splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 마지막 len(data) 에서 block_size 만큼 뽑을 수 있도록 하기 위해 len(data)-block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _ , loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each toekn directly reads off the logits for the token the next toeken from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor if integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array if indeices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # nn.Moduel에서 self를 호출하는건 forward를 호출하는 것과 같음
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilites 
            probs = F.softmax(logits, dim=-1) # (B, C) | C = vocab_size, 길이가 vocab_size개 벡터 출력
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) | multinomial은 확률분포에 따라 샘플링해서 인덱스를 뽑고, 각 인덱스가 뽑힐 확률은 probs 값과 비례. num_samples=1 이므로 하나만 뽑음
            # append sampled index tothe running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) | 예측한 다음 토큰과 현재 토큰을 concat 해서 T+1을 만들며 max_new_token 길이 만큼 반복
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    #every once in a while evaluate th loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'setp {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # samople a batxh of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device) # 입력값으로 토큰 0
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))