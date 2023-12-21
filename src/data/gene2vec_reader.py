import torch
import numpy as np
import sys
from torch.distributions.normal import Normal

from utils.json_utils import JsonUtils
ju = JsonUtils()

from utils.config_loader import Config
config = Config()

def load_pretrained_embeddings(file_path):
    gene_to_idx = {}
    embeddings_list = []

    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            tokens = line.strip().split()
            gene_symbol = tokens[0]
            embedding_vector = np.array([float(value) for value in tokens[1:]], dtype=np.float32)
            
            gene_to_idx[gene_symbol] = idx
            embeddings_list.append(embedding_vector)

    # Convert the list of numpy arrays to a single numpy array
    embeddings_array = np.array(embeddings_list)
    pretrained_embeddings = torch.tensor(embeddings_array)
    return gene_to_idx, pretrained_embeddings

def load_pretrained_embeddings_add_special_embeddings(file_path, number_of_special_embeddings):
    gene_to_idx = {}
    embeddings_list = []
    gene_list = []

    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            tokens = line.strip().split()
            gene_symbol = tokens[0]
            embedding_vector = np.array([float(value) for value in tokens[1:]], dtype=np.float32)

            gene_to_idx[gene_symbol] = idx + number_of_special_embeddings  # Shift all indices
            embeddings_list.append(embedding_vector)
            gene_list.append(gene_symbol)

    embeddings_array = np.array(embeddings_list)
    mean = embeddings_array.mean(axis=0)
    std = embeddings_array.std(axis=0)

    m = Normal(torch.tensor([mean]), torch.tensor([std]))
    special_embeddings = m.sample((number_of_special_embeddings,)).squeeze().numpy()

    extended_embeddings_array = np.concatenate((special_embeddings, embeddings_array))

    special_genes = [f"SPECIAL_EMB{i+1}" for i in range(number_of_special_embeddings)]
    extended_gene_list = special_genes + gene_list

    gene_to_idx = {f"SPECIAL_EMB{i+1}": i for i in range(number_of_special_embeddings)} | gene_to_idx

    pretrained_embeddings = torch.tensor(extended_embeddings_array)

    return gene_to_idx, pretrained_embeddings, extended_gene_list

# Usage:
file_path = config.get("gene2vec_file_path")
number_of_special_embeddings = config.get("number_of_special_embeddings")
#gene_to_idx, pretrained_embeddings = load_pretrained_embeddings(file_path)
gene_to_idx, pretrained_embeddings, extended_gene_list = load_pretrained_embeddings_add_special_embeddings(file_path, number_of_special_embeddings)
torch.save(pretrained_embeddings, config.get("gene2vec_embeddings_pyn_path"))
ju.save_data_to_file(gene_to_idx, config.get("gene2vec_gene_to_idx_json_path"))
# Saving embeddings to TSV file
with open(config.get("gene2vec_embeddings_pyn_path") + '.tsv', 'w') as f:
    for gene, emb in zip(extended_gene_list, pretrained_embeddings.numpy()):
        f.write(f'{gene}\t' + '\t'.join(map(str, emb)) + '\n')

