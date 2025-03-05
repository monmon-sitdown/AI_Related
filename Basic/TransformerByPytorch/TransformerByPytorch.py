import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337) #Random Seed
file_name = "ThreeKingdom.txt"

with open(file_name, "r", encoding='utf-8') as f:
  text = f.read() #str

chars = sorted(list(set(text)))
vocab_size = len(chars)

#The projection between char and int 
stoi = {ch : i  for i, ch in enumerate(chars)} #char to int
itos = {i: ch for i, ch in enumerate(chars)} #int to char
encode = lambda s: [stoi[c] for c in s] #String to int(List) 
decode = lambda list1: ''.join([itos[i] for i in list1])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data)) #90% for training
train_data = data[:n]   #training data
val_data = data[n:]     #test data

BLOCK_SIZE = 5
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split, block_size, batch_size, train_data, val_data):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #According to x predict x + 1
  x, y = x.to(device), y.to(device)
  return x, y

size = 10 #
n_embedding = 3 #
embedding_table = nn.Embedding(size, n_embedding)

#idx = torch.tensor(0)
idx = torch.arange(size)
#print(idx)
#print(embedding_table(idx))
#print(embedding_table.weight)

x, y = get_batch("train", BLOCK_SIZE, BATCH_SIZE, train_data, val_data)
token_embedding_table = nn.Embedding(vocab_size, n_embedding).to(device)
embd = token_embedding_table(x)

position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embedding).to(device)
position_embd = position_embedding_table(torch.arange(BLOCK_SIZE).to(device))
#print(position_embd, position_embd.shape)

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.value = nn.Linear(n_embedding, head_size, bias = False)
    self.key = nn.Linear(n_embedding, head_size, bias = False)
    self.query = nn.Linear(n_embedding, head_size, bias = False)
    #OneHead is n_embedding, MultiHead the headsize is n_embedding // numofhead
    self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    #A structure which can not be trained , see it as constant

    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    B, T, C = x.shape
    wel = torch.ones(T, T) #Attention matrix
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wel = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 #transpose k at (-2,-1) is to transpose k at dimension T and headsize, divide square toot of headsize for number explosion later(exponential cal)


    wel = wel.to(device)       # Move wel to the correct device
    self.tril = self.tril.to(device)
    wel = wel.masked_fill(self.tril == 0, float('-inf')) #mask matrix
    wel = F.softmax(wel, dim = -1) #？！
    wel = self.dropout(wel)   #Randomly delete some value to make the network stable

    v = self.value(x)
    out = wel @ v
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embedding)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embedding):
    super().__init__()
    self.net = nn.Sequential( #Sequence of execution
        nn.Linear(n_embedding, 4 * n_embedding),
        nn.ReLU(),
        nn.Linear(4 * n_embedding, n_embedding),
        nn.Dropout(0.2)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embedding, num_heads):
    super().__init__()
    self.sa = MultiHeadAttention(num_heads, n_embedding // num_heads) #sa self-attention multi
    self.ffwd = FeedForward(n_embedding)
    self.ln1 = nn.LayerNorm(n_embedding)
    self.ln2 = nn.LayerNorm(n_embedding)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) #Residual Multi-Head Attention
    x = x + self.ffwd(self.ln2(x)) #Forward layer
    return x 
  
import random
import textwrap

n_embedding = 384
num_heads = 8
head_size = n_embedding // num_heads


class LanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embedding).to(device)
    self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embedding).to(device)
    
    self.blocks = nn.Sequential(*[Block(n_embedding, num_heads) for _ in range(6)]) #6 is numberoflayers * is to split
    self.ln_f = nn.LayerNorm(n_embedding) #final layer norm
    self.lm_head = nn.Linear(n_embedding, vocab_size).to(device)#input n_embd, output :vocab_size


  def forward(self, idx, targets = None):
    B, T = idx.shape #(B, T)B batchsize, T blocksize tokens is data
    token_embd = self.token_embedding_table(idx)
    position_embd = self.position_embedding_table(torch.arange(T).to(device))
    x = token_embd + position_embd #(B, T, n_embd)

    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) #(output in  vocabsize

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C) #"flattening" or "reshaping" 3D to 2D 8x5 -> 40x1
      targets = targets.view(B * T)  #2D to 1D
      loss = F.cross_entropy(logits, targets) #Calculate loss function

    return logits, loss

  def generate(self, token_squ, max_new_tokens): #token_squ: known context， max_new_tokens: how long need to write
    for _ in range(max_new_tokens):
      tokens_input = token_squ[:, -BLOCK_SIZE:]  #every time add a block from last and delete a block from the fron  batch全要，对于T是加blocksize
      logits, loss = self.forward(tokens_input)
      logits = logits[:, -1, :]                   #slice logits [B,-1,vocal_size] Only need the last string, vector of probability distribution
      probs = F.softmax(logits, dim = -1)         #Let low p lower and high probability higher 
      token_next = torch.multinomial(probs, num_samples=1)   #Distribution change to one-hot
      token_squ = torch.cat((token_squ, token_next.to(device)), dim = 1) #join two together
    return token_squ[:,-max_new_tokens:]



import os
from pathlib import Path

BLOCK_SIZE = 256
BATCH_SIZE = 64
def main():
  print(f"training:{file_name}")
  model = LanguageModel()

  save_dir = "checkpoints"

  model_path = save_dir + "/model_epoch_1000.pth"

  if os.path.exists(model_path):
    print("Loading pretrained model...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    model = model.to(device)
  else:
    print("No pretrained model found. Starting training...")

    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004) #learning rate

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #train loop
    for i in range(1001): #train times

      #sampling
      xb, yb = get_batch("train", BLOCK_SIZE, BATCH_SIZE, train_data, val_data)
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True) #Old grad to 0
      loss.backward() #backward, calculate new grad
      optimizer.step() #



      if i % 100 == 0:
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'model_epoch_{i}.pth')
        losses = estimate_loss(model)
        print(f"step:{i}, train loss:{losses['train']:.4f}, val loss:{losses['val']:.4f}")

  max_new_tokens = 384
  start_idx = random.randint(0, len(val_data) - BLOCK_SIZE - max_new_tokens)
  #context
  context = torch.zeros((1, BLOCK_SIZE), dtype = torch.long, device = device)    #B = 1, T = blpcksize
  context[0, :] = val_data[start_idx:start_idx+BLOCK_SIZE]
  context_str = decode(context.tolist()[0]) #1DTnesor
  wrapped_context_str = textwrap.fill(context_str, width = 50)

  #Real text
  real_next_tokens = torch.zeros((1, max_new_tokens), dtype = torch.long, device = device)
  real_next_tokens[0, :] = val_data[start_idx+BLOCK_SIZE:start_idx+BLOCK_SIZE+max_new_tokens]
  real_next_tokens_str = decode(real_next_tokens[0].tolist())
  wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width = 50)

  #predicted text
  generated_tokens = model.generate(context, max_new_tokens)
  generated_tokens_str = decode(generated_tokens[0].tolist())
  wrapped_generated_tokens_str = textwrap.fill(generated_tokens_str, width = 50)

  # out = model(x)
  print("The original context")
  print(wrapped_context_str)
  print("-----")
  print("The original continuation：")
  print(wrapped_real_next_tokens_str)
  print("=====")
  print("AI-generated continuation")
  print(wrapped_generated_tokens_str)

main()