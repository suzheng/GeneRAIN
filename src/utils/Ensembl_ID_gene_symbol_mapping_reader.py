def read_ensembl_to_gene_mapping(file_path):
    ensembl_to_gene = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            ensembl_id, gene_symbol = line.strip().split('\t')
            ensembl_to_gene[ensembl_id] = gene_symbol
    
    return ensembl_to_gene