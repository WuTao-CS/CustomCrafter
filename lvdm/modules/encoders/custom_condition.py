from packaging import version
import torch
import torch.nn as nn
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
import open_clip
from lvdm.common import autocast
from utils.utils import count_params
from torch.utils.checkpoint import checkpoint
from open_clip.tokenizer import SimpleTokenizer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedderWrapper(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, modifier_token, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.modifier_token = modifier_token
        if '+' in self.modifier_token:
            self.modifier_token = self.modifier_token.split('+')
        else:
            self.modifier_token = [self.modifier_token]

        self.add_token()
        self.freeze()

    def add_token(self):
        self.modifier_token_id = []
        token_embeds1 = self.transformer.get_input_embeddings().weight.data
        for each_modifier_token in self.modifier_token:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token)
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token)
            self.modifier_token_id.append(modifier_token_id)

        self.transformer.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.transformer.get_input_embeddings().weight.data
        token_embeds[self.modifier_token_id[-1]] = torch.nn.Parameter(token_embeds[42170], requires_grad=True)
        if len(self.modifier_token) == 2:
            token_embeds[self.modifier_token_id[-2]] = torch.nn.Parameter(token_embeds[47629], requires_grad=True)
        if len(self.modifier_token) == 3:
            token_embeds[self.modifier_token_id[-3]] = torch.nn.Parameter(token_embeds[43514], requires_grad=True)

    def custom_forward(self, hidden_states, input_ids):
        r"""
        Returns:
        """
        input_shape = hidden_states.size()
        bsz, seq_len = input_shape[:2]
        if version.parse(transformers.__version__) >= version.parse('4.21'):
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
        else:
            causal_attention_mask = self.transformer.text_model._build_causal_attention_mask(bsz, seq_len).to(
                hidden_states.device
            )

        encoder_outputs = self.transformer.text_model.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)

        return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.transformer.text_model.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.final_layer_norm.parameters():
            param.requires_grad = False
        for param in self.transformer.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)

        indices = tokens == self.modifier_token_id[-1]
        for token_id in self.modifier_token_id:
            indices |= tokens == token_id

        indices = (indices*1).unsqueeze(-1)

        input_shape = tokens.size()
        tokens = tokens.view(-1, input_shape[-1])

        hidden_states = self.transformer.text_model.embeddings(input_ids=tokens)
        hidden_states = (1-indices)*hidden_states.detach() + indices*hidden_states

        z = self.custom_forward(hidden_states, tokens)

        return z

    def encode(self, text):
        return self(text)

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, modifier_token, initializer_token, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'))
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.modifier_token = modifier_token
        if '+' in self.modifier_token:
            self.modifier_token = self.modifier_token.split('+')
        else:
            self.modifier_token = [self.modifier_token]
        self.initializer_token = initializer_token
        if '+' in self.initializer_token:
            self.initializer_token = self.initializer_token.split('+')
        else:
            self.initializer_token = [self.initializer_token]
        self.tokenizer = SimpleTokenizer(special_tokens=self.modifier_token)
        self.add_token()
        self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.model.token_embedding.parameters():
            param.requires_grad = True

    def add_token(self):
        self.modifier_token_id = self.tokenizer.all_special_ids[2:]
        self.initializer_token_id = [self.tokenizer.encode(t)[0] for t in self.initializer_token]
        new_embedding = torch.zeros((self.tokenizer.vocab_size, self.model.token_embedding.weight.data.shape[1]))
        new_embedding[:self.model.vocab_size] = self.model.token_embedding.weight.data
        for modifier_id, initializer_id in zip(self.modifier_token_id, self.initializer_token_id):
            new_embedding[modifier_id] = self.model.token_embedding.weight.data[initializer_id]
        self.model.token_embedding.weight.data = new_embedding

    def forward(self, text):
        self.device = self.model.positional_embedding.device
        tokens = self.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z
    
    def tokenize(self, texts, context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def _cal_modifier_indx(self, tokens, modifier_token_ids):
        indices = tokens == modifier_token_ids[-1] # [1, 77] # whether each token is the modifier token
        for token_id in modifier_token_ids:
            indices |= tokens == token_id
        indices = (indices*1).unsqueeze(-1) # [1, 77, 1] # 1 indicates modifier token
        return indices
    
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        indices = self._cal_modifier_indx(text, self.modifier_token_id)
        x = (1-indices)*x.detach() + indices*x # detach other tokens except the modifier token
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

class FrozenOpenCLIPEmbedderWrapper(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, modifier_token=None, initializer_token=None, token_init_mode="init_token",
                 arch="ViT-H-14", 
                 version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last", is_modifier_token_learnable=True
                 ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'))
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual

        self.model = model
        # self.tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer = CLIPTokenizer.from_pretrained("./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer.pad_token_id = 0
        
        self.model = self.model.to(device)

        self.device = device
        self.max_length = max_length
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        
        self.modifier_tokens = modifier_token
        self.initializer_token = initializer_token
        self.token_init_mode = token_init_mode
        self.num_timesteps = 1000
        self.is_modifier_token_learnable = is_modifier_token_learnable

        if '+' in self.modifier_tokens: # multiple tokens
            self.modifier_tokens = self.modifier_tokens.split('+') # e.g. [<new1>, <new2>]
        else:
            self.modifier_tokens = [self.modifier_tokens]
        
        if '+' in self.initializer_token:
            self.initializer_token = self.initializer_token.split('+')
        else:
            self.initializer_token = [self.initializer_token]

        self._add_token()
        
        if freeze:
            self.freeze()
        
    def _add_token(self):
        self.modifier_token_id = []
        self.initializer_token_id = []
        token_embeds = self.model.token_embedding.weight.data # [49408, 1024]
        # token_embeds1 = self.transformer.get_input_embeddings().weight.data # [49408, 768]
        
        # add new token (in the last position), 
        # resize, and get new embedding table
        for each_modifier_token in self.modifier_tokens:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token) # add <new1>
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token) # 49408
            self.modifier_token_id.append(modifier_token_id)
        
        for each_initializer_token in self.initializer_token:
            token_ids = self.tokenizer.encode([each_initializer_token], add_special_tokens=False)
            self.initializer_token_id.append(token_ids[0])
        
        # set a new token embedding table with a new size
        self._resize_token_embeddings(len(self.tokenizer))
        # self.transformer.resize_token_embeddings(len(self.tokenizer)) # Embedding(49409, 768)
        
        # initialize the new token embeddings
        token_embeds = self.model.token_embedding.weight.data # [49409, 768]
        print("self.is_modifier_token_learnable ",self.is_modifier_token_learnable )
        if self.token_init_mode == "special_token":
            token_embeds[self.modifier_token_id[-1]] = torch.nn.Parameter(token_embeds[42170], requires_grad=self.is_modifier_token_learnable)
            if len(self.modifier_tokens) == 2:
                token_embeds[self.modifier_token_id[-2]] = torch.nn.Parameter(token_embeds[47629], requires_grad=self.is_modifier_token_learnable)
            if len(self.modifier_tokens) == 3:
                token_embeds[self.modifier_token_id[-3]] = torch.nn.Parameter(token_embeds[43514], requires_grad=self.is_modifier_token_learnable)
        elif self.token_init_mode == "init_token":
            for x, y in zip(self.modifier_token_id, self.initializer_token_id):
                token_embeds[x] = torch.nn.Parameter(token_embeds[y], requires_grad=self.is_modifier_token_learnable)
        else:
            raise NotImplementedError
        self.model.token_embedding.weight.data = token_embeds
        return
        
    def _resize_token_embeddings(self, new_num_tokens):
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end.

        The new token embeddings are randomly initialized in normal distribition.
        """
        old_embeddings = self.model.token_embedding
        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        old_num_tokens, old_embedding_dim = old_embeddings.weight.data.shape

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        factor = 1
        new_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)

        # copy old embeddings
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.model.token_embedding = new_embeddings

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.is_modifier_token_learnable:
            for param in self.model.token_embedding.parameters():
                param.requires_grad = True
        return
    
    def _cal_modifier_indx(self, tokens, modifier_token_ids):
        indices = tokens == modifier_token_ids[-1] # [1, 77] # whether each token is the modifier token
        for token_id in modifier_token_ids:
            indices |= tokens == token_id
        indices = (indices*1).unsqueeze(-1) # [1, 77, 1] # 1 indicates modifier token
        return indices
    
    def forward(self, text, t=None, **kwargs):
        # tokens = open_clip.tokenize(text).to(self.device) # [b,77]
        tokens = self.tokenizer(text, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids'].to(self.device)
        # encode_with_transformer
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        
        # custom diffusion
        indices = self._cal_modifier_indx(tokens, self.modifier_token_id)
        x = (1-indices)*x.detach() + indices*x # detach other tokens except the modifier token
        

        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        z = self.model.ln_final(x)
        return z
    
    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x
    
    def encode(self, text, **kwargs):
        return self(text, **kwargs)


class FrozenOpenCLIPVisualEmbedderWrapper(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, modifier_token=None, initializer_token=None, token_init_mode="init_token",
                 arch="ViT-H-14", 
                 version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last", is_modifier_token_learnable=False
                 ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'))
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual

        self.model = model
        # self.tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer = CLIPTokenizer.from_pretrained("./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer.pad_token_id = 0
        
        self.model = self.model.to(device)
        self.is_modifier_token_learnable = is_modifier_token_learnable
        self.device = device
        self.max_length = max_length
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        
        self.modifier_tokens = modifier_token
        self.initializer_token = initializer_token
        self.token_init_mode = token_init_mode

        if '+' in self.modifier_tokens: # multiple tokens
            self.modifier_tokens = self.modifier_tokens.split('+') # e.g. [<new1>, <new2>]
        else:
            self.modifier_tokens = [self.modifier_tokens]
        
        if '+' in self.initializer_token:
            self.initializer_token = self.initializer_token.split('+')
        else:
            self.initializer_token = [self.initializer_token]

        self._add_token()
        
        if freeze:
            self.freeze()
        
    def _add_token(self):
        self.modifier_token_id = []
        self.initializer_token_id = []
        token_embeds = self.model.token_embedding.weight.data # [49408, 1024]
        # token_embeds1 = self.transformer.get_input_embeddings().weight.data # [49408, 768]
        
        # add new token (in the last position), 
        # resize, and get new embedding table
        for each_modifier_token in self.modifier_tokens:
            num_added_tokens = self.tokenizer.add_tokens(each_modifier_token) # add <new1>
            modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token) # 49408
            self.modifier_token_id.append(modifier_token_id)
        
        for each_initializer_token in self.initializer_token:
            token_ids = self.tokenizer.encode([each_initializer_token], add_special_tokens=False)
            self.initializer_token_id.append(token_ids[0])
        
        # set a new token embedding table with a new size
        self._resize_token_embeddings(len(self.tokenizer))
        # self.transformer.resize_token_embeddings(len(self.tokenizer)) # Embedding(49409, 768)
        
        # initialize the new token embeddings
        token_embeds = self.model.token_embedding.weight.data # [49409, 768]
        print("self.is_modifier_token_learnable ",self.is_modifier_token_learnable )
        if self.token_init_mode == "special_token":
            token_embeds[self.modifier_token_id[-1]] = torch.nn.Parameter(token_embeds[42170], requires_grad=self.is_modifier_token_learnable)
            if len(self.modifier_tokens) == 2:
                token_embeds[self.modifier_token_id[-2]] = torch.nn.Parameter(token_embeds[47629], requires_grad=self.is_modifier_token_learnable)
            if len(self.modifier_tokens) == 3:
                token_embeds[self.modifier_token_id[-3]] = torch.nn.Parameter(token_embeds[43514], requires_grad=self.is_modifier_token_learnable)
        elif self.token_init_mode == "init_token":
            for x, y in zip(self.modifier_token_id, self.initializer_token_id):
                token_embeds[x] = torch.nn.Parameter(token_embeds[y], requires_grad=self.is_modifier_token_learnable)
        else:
            raise NotImplementedError
        self.model.token_embedding.weight.data = token_embeds
        return
        
    def _resize_token_embeddings(self, new_num_tokens):
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end.

        The new token embeddings are randomly initialized in normal distribition.
        """
        old_embeddings = self.model.token_embedding
        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        old_num_tokens, old_embedding_dim = old_embeddings.weight.data.shape

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        factor = 1
        new_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)

        # copy old embeddings
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.model.token_embedding = new_embeddings

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return
    
    def _cal_modifier_indx(self, tokens, modifier_token_ids):
        indices = tokens == modifier_token_ids[-1] # [1, 77] # whether each token is the modifier token
        for token_id in modifier_token_ids:
            indices |= tokens == token_id
        indices = (indices*1).unsqueeze(-1) # [1, 77, 1] # 1 indicates modifier token
        return indices
    
    def forward(self, text, t=None, **kwargs):
        # tokens = open_clip.tokenize(text).to(self.device) # [b,77]
        tokens = self.tokenizer(text, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids'].to(self.device)
        # encode_with_transformer
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
    
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        z = self.model.ln_final(x)
        return z, tokens
    
    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x
    
    def encode(self, text, **kwargs):
        return self(text, **kwargs)
    
if __name__ == "__main__":
    model = FrozenOpenCLIPEmbedder('<new1>','toy')