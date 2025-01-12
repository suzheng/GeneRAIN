# About

This dataset contains various files essential for understanding and employing the GeneRAIN models, as described in the accompanying manuscript. GeneRAIN models use bulk RNA-seq data and a 'Binning-By-Gene' normalization method. These models aim to improve upon existing methods in understanding biological information and include a vector representation of genes called GeneRAIN-vec. After thorough testing, these models have shown their effectiveness in predicting a wide range of biological characteristics, including for long non-coding RNAs. This shows their usefulness and potential in bioinformatics and computational biology. Below is a description and usage guide for the dataset:


## Initial Setup

### Download and Extract the Data
Obtain the `data.tar.gz` file and place it in the home directory of the GeneRAIN project. Use the following command to extract its contents:


	tar -xzvf data.tar.gz

Move the downloaded files `human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_*.npy` into folder `data/external/ARCHS/`
## Dataset Description and Usage

### Model Checkpoint Files
The following checkpoint files correspond to different GeneRAIN models:

	data/models/GeneRAIN.GPT_protein_coding_lncRNAs.pth
	data/models/GeneRAIN.GPT_Binning_By_Gene.pth
	data/models/GeneRAIN.BERT_Pred_Expr_Binning_By_Gene.pth
	data/models/GeneRAIN.BERT_Pred_Genes_Binning_By_Gene.pth

### Gene Index Mapping Files
Json files for gene to index mapping information, which can be used for tokenization of genes. Please note that they include genes with small mean expression values, which means some genes in the json files may not present in the model input dataset.

	data/embedding/coding_lncrna_gene_to_idx.json
	data/embedding/gene2vec_gene_to_idx.json
	
### ARCHS Human Bulk RNA-seq Data
The ARCHS human bulk RNA-seq h5 file `human_gene_v2.2.h5` (the file has to be downloaded from their official [website](https://maayanlab.cloud/archs4/download.html)),  along with the meta data files (generated by `notebooks/data/gen_gene_emb.ipynb`). 

	data/external/ARCHS/human_gene_v2.2.h5
	data/external/ARCHS/human_gene_v2.2.h5.genes_meta.tsv
	data/external/ARCHS/human_gene_v2.2.h5.samples_meta.tsv

### Binning-By-Gene Normalized Data
ARCHS dataset normalized by 'Binning-By-Gene' method, it can be used for training `GPT_Binning_By_Gene`, `BERT_Pred_Expr_Binning_By_Gene` and `BERT_Pred_Genes_Binning_By_Gene` models. Dataset is split into 5 chunks by samples.

	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_0.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_1.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_2.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_3.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_4.npy

### Gene Expression Statistics
File containing the mean expression values of genes in the ARCHS dataset, and their boolean flags which can be used for filtering duplicate gene symbols.

	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes.gene_stat_filt_on_z_dup.tsv

### Binning Boundary Files
Files can be used by notebook `notebooks/anal/normalize_expr_mat.ipynb` to determine the binning boundaries in Binning-By-Gene normalization of new expression data.

	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.gene_symbols.txt
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_gene2vec_0.005_subsampled.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_coding_lncrna_0.005_subsampled.npy
	data/external/ARCHS/normalize_each_gene/human_gene_v2.2_with_zero_expr_genes_bin_tot2000_coding_lncrna_0.005_subsampled.gene_symbols.txt

### Example Input and Output
Example input and output for the `notebooks/anal/anal_dataset.ipynb`.

	data/examples/ReplogleWeissman2022_K562_essential.h5ad.mean_agg.pickle
	data/examples/ReplogleWeissman2022_K562_essential.h5ad.mean_agg.coding_lncrna.binned.tsv
	data/examples/ReplogleWeissman2022_K562_essential.h5ad.mean_agg.coding_lncrna.binned.pkl

### Gene Attribute Prediction Results
All the coding and lncRNA gene attribute prediction results can be found in `genes_clf_pred_results.parquet` , which has the same format as `Supplementary Table 7`, except that it lists the predicted probabilities of all genes.

	data/results/genes_clf_pred_results.parquet
