# modified version of https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
import os

device = None
if torch.backends.mps.is_available():
    device = "mps"
    print(f"Using {device}")
else:
    device = "cpu"
    print ("MPS device not found.")

# ----------- Hyper Parameters -----------
batch_size = 64         # how many sequences we process every forward and backwards pass
block_size = 256        # maximum context length for predictions

num_iterations = 5000
eval_interval = 500
learning_rate = 3e-4

eval_iterations = 200
d_model = 384
n_head = 6
n_layer = 6
dropout = 0.2

# ----------- Read File -----------

with open('harry_potter.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
total_chars = len(text)
vocab_size = len(chars)

print(f"{total_chars},{vocab_size}")

# ----------- Encode/Decode -----------
# translating characters into integers

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]             # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])    # decoder: take a list of integers, output a string

# ----------- Train/Val/Test Splits -----------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# ----------- Data Loading -----------
def get_batch(split, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return 

# ----------- Model Classes -----------
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5                          # (B,T,C) @ (B,C,T) --> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        weights = F.softmax(weights, dim=-1)                                # (B,T,T)
        weights = self.dropout(weights)

        # perform the wighted aggregation of the values
        v = self.value(x)   # (B,T,C)
        out = weights @ v   # (B,T,T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention, in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.proj = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ Transformer Feed Forward NN """

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """ Single Transformer Block """

    def __init__(self, d_model, n_head):
        super().__init__()
        head_size = d_model // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x + ___ are residual connections
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_model = nn.Embedding(vocab_size, d_model)
        # self.position_embedding_table = nn.Embedding(block_size, d_model)

        self.toked_modelding_table = nn.Embedding(vocab_size, d_model)
        self.positiod_modelding_table = nn.Embedding(block_size, d_model)


        # Blocks, Layer Norm, Self-Attention Heads, Feed Forward Netwrok
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        # we need a linear layer to go from token embeddings to logits
        self.lm_head = nn.Linear(d_model, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.toked_modelding_table(idx)                                      # (B,T,C)
        positional_embeddings = self.positiod_modelding_table(torch.arange(T, device=device))   # (T,C)
        x = positional_embeddings + token_embeddings                                            #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                                # (B,T,vocab_size)

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

# ----------- Run via Command Line -----------
if __name__ == "__main__":
    

    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"{num_params} million parameters") 

    losses = None
    checkpoint = torch.load('models/hp/harrypotter_1675929710_2200_loss1.29.pt') 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses = checkpoint['losses']

    # you can add tqdm() here
    for iter in range(num_iterations):
        timestamp = int(time.time())
        st = time.monotonic()

        if iter == 0 and losses == None:
            print(f"initializing model....")
            losses = estimate_loss(device)


        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 and iter > 0:
            losses = estimate_loss(device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        

        if ((iter % 200 == 0 or iter == num_iterations - 1) and iter > 0):
            fn = f"models/hp/harrypotter_{timestamp}_{iter}_loss{losses['val']:.2f}.pt"
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses
            }, fn)

            print(f"saved model {fn} with size {os.path.getsize(fn)}")
            

        # sample a batch of data
        xb, yb = get_batch('train', device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        et = time.monotonic() - st
        
        if iter % 20 == 0:
            print(f"{et*1000:.2f} ms  {1/et:.2f} its/sec, train_loss: {loss:.2f} ")

        # if iter % 200 == 0:
            # generate from the model
            # context = torch.zeros((1, 1), dtype=torch.long, device=device)
            # print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))

