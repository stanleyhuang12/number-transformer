# TODO: experiment with different model hyperparameters

import math
import random
import torch
import sys 

characters = "()+0123456789="
TOKENS = ["<bos>", "<eos>", "<pad>"] + [c for c in characters]
TOKEN_MAP = dict((t, i) for i, t in enumerate(TOKENS))

BOS = TOKEN_MAP["<bos>"]
EOS = TOKEN_MAP["<eos>"]
PAD = TOKEN_MAP["<pad>"]


def decode(token_ids):
    return "".join(TOKENS[i] for i in token_ids)

def encode(s, *, eos=True):
    if s.startswith("<bos>"):
        s = s[5:]

    output = [BOS]
    output.extend(TOKEN_MAP[c] for c in s)

    if eos:
        output.append(EOS)

    return torch.tensor(output, device=device)


def generate_instance(n, *, value_min=1, value_max=9):
    current_numbers = [random.randint(value_min, value_max) for _ in range(n)]
    current_expressions = [[str(v) for v in current_numbers]]
    current_fresh = [True for _ in current_numbers]

    while len(current_numbers) > 1:
        next_numbers = []
        next_expressions = [[] for _ in range(len(current_expressions) + 1)]
        next_fresh = []

        i = 0
        while i < len(current_numbers):
            can_merge = (i + 1 < len(current_numbers)) and (current_fresh[i] or current_fresh[i + 1])
            if can_merge and random.random() < 0.5:
                next_numbers.append(current_numbers[i] + current_numbers[i + 1])

                next_expressions[0].append(str(next_numbers[-1]))
                for j in range(len(current_expressions)):
                    next_expressions[j + 1].append(f"({current_expressions[j][i]}+{current_expressions[j][i + 1]})")

                next_fresh.append(True)
                i += 2
            else:
                next_numbers.append(current_numbers[i])

                next_expressions[0].append(str(next_numbers[-1]))
                for j in range(len(current_expressions)):
                    next_expressions[j + 1].append(current_expressions[j][i])

                next_fresh.append(False)
                i += 1

        if len(next_numbers) < len(current_numbers):
            current_numbers = next_numbers
            current_expressions = next_expressions
            current_fresh = next_fresh

    output = '='.join(e[0] for e in reversed(current_expressions))
    return encode(output)
    
def make_batch(*args, batch_size=64, **kwargs):
    seqs = [generate_instance(*args, **kwargs) for _ in range(batch_size)]

    batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD)
    x = batch[:, :-1]
    y = batch[:, 1:]

    return x.to(device), y.to(device)

def causal_mask(T):
    m = torch.full((T, T), float("-inf"), device=device)
    m = torch.triu(m, diagonal=1)  # upper triangle is masked
    return m


class MathTransformer(torch.nn.Module):
    """
    A GPT-style decoder-only transformer for sequence modeling of math expressions.
    Uses learned token and positional embeddings, stacked causal self-attention
    blocks (via TransformerEncoder with an upper-triangular mask), and a linear
    language model head projecting hidden states to vocabulary logits. Weights
    are initialized with small normal distributions (std=0.02), following GPT
    conventions.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_ff=512, max_len=100, dropout=0.2):
        """
        Initialize the MathTransformer.
        Args:
            d_model (int): Dimensionality of token and positional embeddings,
                and all hidden states throughout the model. Default: 128.
            nhead (int): Number of attention heads in each TransformerEncoderLayer.
                Must evenly divide d_model. Default: 4.
            num_layers (int): Number of stacked TransformerEncoderLayer blocks.
                Default: 4.
            dim_ff (int): Hidden dimensionality of the feed-forward sublayer
                within each TransformerEncoderLayer. Default: 256.
            max_len (int): Maximum sequence length supported by the positional
                embedding table. Default: 64.
            dropout (float): Dropout probability applied within each
                TransformerEncoderLayer. Default: 0.1.
        """
        super().__init__()
        self.d_model = d_model # dimensions of the token, position embeddings 
        self.max_len = max_len # maximum sequence length supported by the positional embedding table 

        vocab_size = len(TOKENS) 

        # token + position embeddings
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD) # Embedding are learnable parameters, 
        self.pos_emb = torch.nn.Embedding(max_len, d_model)

        # Define a TransformerEncoderLayer with the specified parameters.
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )

        # Create a TransformerEncoder with the TransformerEncoderLayer defined
        # and the specified number of layers.
        self.blocks = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

        # The head of the transformer is linear layer with d_model size
        # input and vocab_size output.
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

        # initialize the weights of the token and position embeddings,
        # the linear head, and the bias of the linear head.
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.lm_head.bias)

    def forward(self, x):
        """
        Run a forward pass through the MathTransformer.
        Combines learned token and positional embeddings (with token embeddings
        scaled by sqrt(d_model)), then passes the result through stacked causal
        self-attention blocks using both a causal mask (to prevent attending to
        future positions) and a key padding mask (to ignore PAD tokens). The
        final hidden states are projected to vocabulary logits by the language
        model head.
        Args:
            x (torch.Tensor): Integer token index tensor of shape (N, T), where
                N is the batch size and T is the sequence length.
        Returns:
            torch.Tensor: Logits of shape (N, T, vocab_size), where each position
                contains unnormalized scores over the vocabulary. The logit at
                position i represents the model's prediction for the token
                following position i.
        """
        # x: (N, T)
        N, T = x.shape

        # Create position indices [0, 1, 2, ..., T-1] as a (1, T) tensor.
        # The .unsqueeze(0) adds the batch dimension so it can broadcast across
        #  all N sequences.
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        # Token embeddings scaled by sqrt(d_model) and added to positional
        # embeddings. This combines the token and positional information to
        # form the input to the transformer encoder.
        # h shape: (N, T, d_model)
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)

        # key padding mask: -inf where PAD, 0.0 elsewhere (float, additive — same type as attn_mask)
        key_padding_mask = torch.zeros(N, T, device=x.device)
        key_padding_mask = key_padding_mask.masked_fill(x == PAD, float('-inf'))

        # causal mask for self-attention (float, -inf above diagonal)
        attn_mask = causal_mask(T) # (T, T)

        # Pass the input through the transformer encoder.
        # The encoder applies self-attention with the causal mask and ignores
        # PAD tokens.
        h = self.blocks(
            h,
            mask=attn_mask,                         # causal
            src_key_padding_mask=key_padding_mask   # pad masking
        )

        # Project the hidden states to vocabulary logits.
        logits = self.lm_head(h)  # (N, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, prefix_ids, max_new_tokens=64):
        """
        Autoregressively generate tokens following a given prefix.
        Starting from the provided prefix, repeatedly runs a forward pass,
        takes the logit at the last position, and greedily appends the
        highest-scoring token. Stops early if all sequences in the batch
        produce an EOS token, or if the sequence length reaches max_len.

        Args:
            prefix_ids (torch.Tensor): Integer token index tensor of shape
                (N, T0), where N is the batch size and T0 is the prefix length.
            max_new_tokens (int): Maximum number of new tokens to generate
                beyond the prefix. Default: 64.
        Returns:
            torch.Tensor: Integer token index tensor of shape (N, T0 + K),
                where K <= max_new_tokens is the number of tokens actually
                generated before hitting the EOS or max_len stopping condition.
        """
        self.eval()
        x = prefix_ids.clone().to(next(self.parameters()).device)  # (N, T0)
        for _ in range(max_new_tokens):
            if x.size(1) >= self.max_len:
                break
            logits = self.forward(x)[:, -1, :]   # (N, V)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
            x = torch.cat([x, next_id], dim=1)
            if (next_id == EOS).all():
                break
        return x



if __name__ == "__main__": 
    input_file_name = sys.argv[1]
    model = MathTransformer() 
    model.load_state_dict(torch.load("math.pt", map_location=device))
    model.eval()
    print("Hello!") 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda!") 
    else:
        device = torch.device("cpu")

    with open(input_file_name, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
   
    for line in lines:
        prefix = encode(line, eos=False).unsqueeze(0)  # (1, T)
        output_ids = model.generate(prefix)[0].tolist()
        # decode, strip special tokens
        result = decode([t for t in output_ids if t not in (BOS, EOS, PAD)])
        print(result)