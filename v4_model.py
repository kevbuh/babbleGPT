# https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
import os

batch_size = 64 # how many sequences we process every forward and backwards pass
block_size = 256 # maximum context length for predictions

num_iterations = 5000
eval_interval = 500
learning_rate = 3e-4
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    

eval_iterations = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# translating characters into integers

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]          # encoder: String -> List[Ints]
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: List[Ints] -> String

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(device):
    out = {}

    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)

        # perform the wighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention, in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + ___ are residual connections
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # we need a linear layer to go from token embeddings to logits

        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     nn.LayerNorm(n_embed)
        # )
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed// 4)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = positional_embeddings + token_embeddings #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop context to block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        # device = torch.device("mps")
        device = "mps"

        print(f"Using {device}")
    else:
        device = 'cpu'
        print ("MPS device not found.")

    model = BigramLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint = torch.load('models/v4_1675802408_200_loss2.47.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint['losses']


    print(f"putting {device} to model")


    for iter in range(num_iterations):
        timestamp = int(time.time())

        st = time.monotonic()

        if iter == 0:
            print(f"initializing model....")
            # losses = estimate_loss(device)


            # print(f"iter: {iter}")


        # print(f"test 0 ....")

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 and iter > 0:
            losses = estimate_loss(device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


        # print(f"test....")
        

        if ((iter % 100 == 0 or iter == num_iterations - 1) and iter != 0):
            fn = f"models/v4_{timestamp}_{iter}_loss{losses['val']:.2f}.pt"
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses

            }, fn)

            print(f"saved model {fn} with size {os.path.getsize(fn)}")
            

        # sample a batch of data
        # print("Getting batch....")
        xb, yb = get_batch('train', device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        et = time.monotonic() - st
        if iter % 10 == 0:
            print(f"{et*1000:.2f} ms  {1/et:.2f} its/sec, train_loss: {loss:.2f} ")

            # generate from the model
            context = torch.randn((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=20)[0].tolist()))


        
