import gc
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import log as logger
from utils.functions import tokenizer
from models.layers import encode_input

cache = {}

class NodesEmbedding:

    def __init__(self, nodes_dim: int, model_path=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        from transformers import RobertaTokenizer, RobertaModel
        candidates = []
        if model_path is not None:
            candidates.append(model_path)
        project_root = Path(__file__).resolve().parents[2]
        candidates.extend([project_root / 'models' / 'unixcoder', 'microsoft/unixcoder-base'])
        last_error = None
        for candidate in candidates:
            try:
                self.tokenizer_bert = RobertaTokenizer.from_pretrained(str(candidate))
                self.bert_model = RobertaModel.from_pretrained(str(candidate)).to(self.device)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError(f'Unable to load a CodeLM model for node embeddings. Last error: {last_error}')
        self.nodes_dim = nodes_dim
        assert self.nodes_dim >= 0
        self.target = torch.zeros(self.nodes_dim, self.bert_model.config.hidden_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes, types, codes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()
        types_buffer = [None] * self.nodes_dim
        codes_buffer = [None] * self.nodes_dim
        types_buffer[:len(types)] = types
        codes_buffer[:len(codes)] = codes
        self.target[:nodes_tensor.size(0), :] = nodes_tensor
        return (self.target, types_buffer, codes_buffer)

    def embed_nodes(self, nodes):
        embeddings = []
        types = []
        codes = []

        def get_cached_embedding(tokenized_code):
            code_str = ' '.join(tokenized_code)
            if code_str in cache:
                return cache[code_str]
            return None

        def set_cached_embedding(tokenized_code, embedding):
            code_str = ' '.join(tokenized_code)
            cache[code_str] = embedding
        for n_id, node in nodes.items():
            node_type = node.type
            node_code = node.get_code()
            tokenized_code = tokenizer(node_code, True)
            cached = get_cached_embedding(tokenized_code)
            if cached is not None:
                source_embedding = cached
                valid_embedding = True
            else:
                valid_embedding = False
            while not valid_embedding:
                try:
                    length = len(tokenized_code)
                    if length == 0:
                        raise ValueError('Empty tokenized code')
                    input_ids, attention_mask = encode_input(tokenized_code, self.tokenizer_bert)
                    cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
                    source_embedding = np.mean(cls_feats.cpu().detach().numpy(), 0)
                    set_cached_embedding(tokenized_code, source_embedding)
                    valid_embedding = True
                    del input_ids, attention_mask, cls_feats
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                except RuntimeError:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if length > 13:
                        tokenized_code = tokenized_code[length // 2 - 6:length // 2 + 7]
                    elif length % 2 == 0:
                        tokenized_code = tokenized_code[1:]
                    else:
                        tokenized_code = tokenized_code[:-1]
                except Exception as e:
                    source_embedding = np.zeros(self.bert_model.config.hidden_size)
                    valid_embedding = True
            embedding = np.concatenate((np.array([node_type]), source_embedding), axis=0)
            embeddings.append(embedding)
            types.append(node_type)
            codes.append(node_code)
        return (np.array(embeddings), types, codes)

    def get_vectors(self, tokenized_code, node):
        vectors = []
        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.vocab:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                vectors.append(np.zeros(self.kv_size))
                if node.label not in ['Identifier', 'Literal', 'MethodParameterIn', 'MethodParameterOut']:
                    msg = f'No vector for TOKEN {token} in {node.get_code()}.'
                    logger.log_warning('embeddings', msg)
        return vectors

class GraphsEmbedding:

    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)
        return torch.tensor(connections).long()

    def nodes_connectivity(self, nodes):
        coo = [[], []]
        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception('节点顺序错误')
            for e_id, edge in node.edges.items():
                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)
                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)
        return coo

def nodes_to_input(nodes, target, nodes_dim, edge_type):
    graphs_embedding = GraphsEmbedding(edge_type)
    edge_index = graphs_embedding(nodes)
    if len(edge_index[0]) + len(edge_index[1]) == 0:
        return None
    nodes_embedding = NodesEmbedding(nodes_dim)
    x, types, codes = nodes_embedding(nodes)
    label = torch.tensor([target]).float()
    py_data = Data(x=x, edge_index=edge_index, y=label, types=types, codes=codes)
    print(f'=== PyTorch Geometric Data - sample: {py_data} ===')
    return py_data
