# babbleGPT

Unleashing boundless Shakespearean prose with the power of a transformer-based language model, trained on the complete works of The Bard himself.

```babyGPT.py``` --> 10.79 million parameters, character-level generative decoder-only transformer based off of Andrej Karpathy's lecture.

### Output from model_v0

![v4 output](v4_output.png)

Todo:
- [ ] fix 'hhhhhhhhhhhhhhhhhhh' bug
- [ ] make infinite inference
- [ ] web interface
    - Svelte frontend
    - Flask backend
- [ ] custom authors?
    [x] JK Rowling
    - Steven King
    - Leonardo DaVinci
- [ ] graph loss --> weights and biases?
    - https://wandb.ai
- [ ] OpenAI Text Embedding
    - https://openai.com/blog/new-and-improved-embedding-model
- [x] add easy way to get pretrained model for inference 
- [x] tqdm
- [x] gpt file
- [x] timeit for steps
