from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import os
import json
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from transformers import AutoTokenizer, AutoModel

MODEL_LATENT_DIM_DICT = {
    'michiyasunaga/BioLinkBERT-large': 1024
}

class TextEncoder(nn.Module):
    def __init__(self, 
                 text_embedding_dim: int = 1024,
                 embedding_dim: int = 512, 
                 depth: int = 4,
                 embedding_tokens: int = 1,
                 norm_layer: nn.Module = None):
        super().__init__()
        size = embedding_dim * embedding_tokens
        n = embedding_tokens
        d = embedding_dim

        norm_layer = norm_layer(size) if norm_layer is not None else nn.Identity()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_embedding_dim, size),
                norm_layer,
                Rearrange('b (n d) -> b n d', n=n, d=d),
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        return [layer(x) for layer in self.proj]


class EmbeddingGenerator(ABC):
    def __init__(self, unique_cond_dict: Dict[str, list] = None, unique_cond_list: list = None,
                 model: str = 'michiyasunaga/BioLinkBERT-large', 
                 requires_grad: bool = False, savedir: str = "./ontology_resources", 
                 tensor_fname: str = "cl-emb.pt", data_emb_fname: str = "HLCA_sub-cl-emb.pt",):
        self.model = model
        self.requires_grad = requires_grad
        assert unique_cond_dict is not None or unique_cond_list is not None
        if unique_cond_list is None:
            unique_cond_list = sorted(self.dict_to_list_of_tuples(unique_cond_dict))
        self.unique_cond = unique_cond_list
        self.unique_cond_seq = self.generate_text(self.unique_cond)
        self.save_and_load(savedir, tensor_fname)
        self.emb_dict_to_tensor_and_save(savedir, data_emb_fname)

    def get_embeddings(self):
        text_emb_list = self.extract_class_token_from_pretrained(self.unique_cond_seq)
        self.emb = {
            self.unique_cond[idx]: text_emb_list[idx]
            for idx in range(len(self.unique_cond))
        }
    
    def dict_to_list_of_tuples(self, input_dict):
        if len(list(input_dict)) > 1:
            input_list = [input_dict[k] for k in input_dict.keys()]
            return list(map(tuple, zip(*input_list)))
        else:
            return input_dict[list(input_dict)[0]]
    
    def extract_class_token_from_pretrained(self, input_seqs: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        pretrained_model = AutoModel.from_pretrained(self.model)
        inputs = tokenizer(input_seqs, padding=True, return_tensors="pt")            
        outputs = pretrained_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        class_token_emb = last_hidden_states[:, 0, :] # extract first CLS token embedding
        return class_token_emb.detach().requires_grad_(False)
    
    def save_and_load(self, savedir, tensor_fname):
        assert tensor_fname.endswith('.pt')
        if not os.path.exists(f"{savedir}/{tensor_fname}"):
            self.get_embeddings()
            torch.save(self.emb, f"{savedir}/{tensor_fname}")
        else:     
            self.emb = torch.load(f"{savedir}/{tensor_fname}")
    
    def emb_dict_to_tensor_and_save(self, savedir, tensor_fname):
        emb_list = []
        for key in self.unique_cond:
            emb_list.append(self.emb[key])
        self.emb_tensor = torch.stack(emb_list)
        if not os.path.exists(f"{savedir}/{tensor_fname}"):
            assert tensor_fname.endswith('.pt')
            torch.save(self.emb_tensor, f"{savedir}/{tensor_fname}")
    
    @abstractmethod
    def generate_text(self, cond_list: List = None):
        ...

    def __call__(self, cond_dict: Dict[str, list] = None, cond_list: List[str] = None):
        assert cond_dict is not None or cond_list is not None
        if cond_list is None:
            cond_list = self.dict_to_list_of_tuples(cond_dict)
        out_tensor = torch.stack([self.emb[x] for x in cond_list])
        if not self.requires_grad:
            out_tensor = out_tensor.detach().requires_grad_(False)
        return out_tensor


class SimpleEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, *args, savedir="./ontology_resources", tensor_fname="simple-emb.pt", sep=", ", **kwargs):
        self.sep = sep
        super().__init__(*args, savedir=savedir, tensor_fname=tensor_fname, **kwargs)
    
    def generate_text(self, cond_list: List[tuple] = None):
        return [self.sep.join(x) for x in cond_list]

# TODO: update the descriptions of CD4-positive, alpha-beta T cell and CD8-positive, alpha-beta T cell
class CLEmbeddingGenerator(EmbeddingGenerator):
    CL_URL = "https://github.com/obophenotype/cell-ontology/releases/download/v2023-08-24/cl-full.json"
    CL_DESCRIPTION_BY_GPT = {
        'CL:0000319': ' '.join('A secretory cell specialized in the production and secretion of mucus. \
            These cells are typically found in various mucosal epithelia and contribute to the \
            protection and lubrication of the epithelial surfaces. Mucus-secreting cells are \
            characterized by the presence of mucin-containing granules, which release mucin \
            glycoproteins into the extracellular space upon stimulation.'.split()),
        'CL:1001568': ' '.join('An endothelial cell that is part of the pulmonary artery, responsible \
            for lining the inner surface of the pulmonary artery walls. These endothelial cells \
            play a crucial role in regulating blood flow, vascular tone, and gas exchange in the \
            pulmonary circulation. They are essential for maintaining pulmonary vascular \
            homeostasis and facilitating the exchange of oxygen and carbon dioxide in the lungs.'.split()),       
    }
    NULL_DESCRIPTION = {'null': 'A cell'}

    def __init__(self, *args, savedir="./ontology_resources", tensor_fname="cl-emb.pt", 
                 data_emb_fname="HLCA_sub-cl-emb.pt", null_flag=False, **kwargs):
        self.download_and_read(savedir, null_flag)
        super().__init__(*args, savedir=savedir, tensor_fname=tensor_fname, data_emb_fname=data_emb_fname, **kwargs)
    
    def download_and_read(self, savedir, null_flag=False):
        if not os.path.exists(f"{savedir}/cl-full.json"):
            import wget 
            wget.download(self.CL_URL, out=savedir)
        with open(f"{savedir}/cl-full.json") as f:
            self.cl = json.load(f)
        self.cl_to_def = {
            ':'.join(i['id'].split("/")[-1].split('_')): i['meta']['definition']['val'] 
            for i in self.cl['graphs'][0]['nodes'] 
            if 'meta' in i and 'definition' in i['meta']
        }
        self.cl_to_def.update(self.CL_DESCRIPTION_BY_GPT)
        if null_flag:
            self.cl_to_def.update(self.NULL_DESCRIPTION)
    
    def generate_text(self, cond_list: List[str] = None):
        return [self.cl_to_def[x] for x in cond_list]