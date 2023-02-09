import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
import os
from babyGPT import GPT, decode

if __name__ == "__main__":

    print("initializing generation...")

    CHAR_COUNT = int(os.getenv("CHAR_COUNT"))

    if not CHAR_COUNT:
        print("Please use command line argument of CHAR_COUNT=100")
        exit(1)

    device = None
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using {device}")
    else:
        device = "cpu"
        print ("MPS device not found.")


    model = GPT()
    model.to(device)

    checkpoint = torch.load('models/v4_1675811956_1950_loss1.56.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"generating output with {CHAR_COUNT} words...")
    print("--------------------------------------------------")

    st = time.monotonic()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=CHAR_COUNT)[0].tolist()))

    et = time.monotonic() - st

    print(f"\n Inference took: {et*1000:.2f} ms, ms/char: {et*1000//CHAR_COUNT} \n")

    print("--------------------------------------------------")

