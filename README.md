# GeneRAIN

## About GeneRAIN

GeneRAIN models, based on BERT and GPT Transformer architectures, are trained on an extensive dataset of 410K human bulk RNA-seq samples. These models focus on analyzing gene network correlations and developing robust gene representations. By leveraging bulk RNA-seq data, GeneRAIN distinguishes itself from conventional models that primarily use single-cell RNA-seq data. The combination of varied model architectures with the 'Binning-By-Gene' normalization method allows GeneRAIN to effectively decode a broad range of biological information. This repository serves as a platform providing the necessary code and instruction for dataset preparation, training of the GeneRAIN models, and their application to new samples, utilizing this specialized normalization technique.

## Repository Contents
- **Data Preparation Scripts**: Tools to prepare and preprocess the dataset for training the GeneRAIN models.
- **Model Training Framework**: Scripts and guidelines for training the GeneRAIN models on the prepared datasets.
- **Normalization Tools**: Implementation of the 'Binning-By-Gene' normalization method for processing new expression data and preparing it for model input.
- **Utilization of Model Checkpoints**: Using pre-trained models and checkpoints for applying GeneRAIN to new datasets.


## Installation and Setup

To successfully install and set up this project, ensure that you have a Linux environment equipped with CUDA-capable GPUs. Additionally, the corresponding NVIDIA drivers and CUDA toolkit must be properly installed to fully leverage the GPU acceleration capabilities required by the project.

1. **Clone the Repository**:

	```bash
	git clone https://github.com/suzheng/GeneRAIN.git
	```
2. **Set Up a Virtual Environment**:
Before installing the package, it's recommended to set up a virtual environment. This will keep the project's dependencies isolated from your global Python environment.

	```
	# Navigate to the project directory
	cd [project-directory]
	
	# Create the folders for training output
	mkdir -p results/eval_data results/logs  results/models results/debugging/eval_data results/debugging/logs  results/debugging/models
	
	# Create a virtual environment named 'generain'
	python -m venv generain
	
	# Activate the virtual environment
	source generain/bin/activate
	```

3. **Install Dependencies**:
Once the virtual environment is activated, install the project's dependencies.

	```
	pip install -r requirements.txt
	```

4. **Install the Package**:
Install the project package within the virtual environment. 

	```
	pip install .
	# For development, use command below instead:
	pip install -e .
	```
After installation, the package and its modules can be imported into other Python scripts or notebooks.


5. **Prepare the Data**:
	- Download the dataset from Zenodo.
	- Extract the downloaded `tar.gz` file.
	- Move the downloaded `human_gene_v2.2_with_zero_expr_genes_bin_tot2000_final_gene2vec_chunk_*.npy` files to `data/external/ARCHS/normalize_each_gene/` in the extracted folder.
	- Download the ARCHS4 [`human_gene_v2.2.h5`](https://maayanlab.cloud/archs4/download.html) file, and move the `human_gene_v2.2.h5` file to folder `data/external/ARCHS/`


## Train the Models

Once you have set up everything, you are ready to begin training the models. The training process is managed through the script [`src/train/pretrain.py`](src/train/pretrain.py), which accepts three parameters:

- `--epoch_from`: Specifies the starting epoch number of training, beginning from 1. (Type: int, Default: None)
- `--epoch_to`: Specifies the ending epoch number of training. (Type: int, Default: None)
- `--exp_label`: Provides an experiment label for the output. (Type: str, Default: None)
- Please note that, the dataset in the Zenodo repo was normalized by 'Binning-By-Gene' method, and it is only for training `GPT_Binning_By_Gene`, `BERT_Pred_Expr_Binning_By_Gene` and `BERT_Pred_Genes_Binning_By_Gene` models.

### Configuration via JSON File

- All model and training hyperparameters are specified in a JSON file. Please find folders [`jsons`](json) for all the json files used for different GeneRAIN models. 
- The filename of this JSON configuration should be set in the environmental variable `PARAM_JSON_FILE`.

### Debugging

- To run the training in debug mode, set the environment variable `RUNNING_MODE` to `debug`.

### Parallelism with DDP

- The training script utilizes Distributed Data Parallel (DDP) for parallelism.

### Example Scripts
- Example PBS scripts for submitting the training job can be found at [`src/examples`](src/examples).

### Output and Logging

- After training, the model checkpoints will be saved to the directory `results/models/pretrain/`.
- Tensorboard logs of the training loss are saved in `results/logs/pretrain/`. These logs can be viewed with the command `tensorboard --logdir=LOGS_FOLDER_OF_THE_EXPERIMENT`.
- For detailed usage of Tensorboard, please refer to its [official website](https://www.tensorflow.org/tensorboard).

Ensure that you have configured all necessary parameters and environment variables before initiating the training process.

## Normalize New Expression Data and Use with Pretrained Models

### Binning-By-Gene Normalization

If you have your own expression data, you can apply the Binning-By-Gene normalization method to it. Follow these steps:

1. **Normalization Process**:
   - Open and follow the steps in the notebook: [`notebooks/anal/normalize_expr_mat.ipynb`](notebooks/anal/normalize_expr_mat.ipynb).
   - This notebook guides you through the normalization process step-by-step.

### Creating a Dataset for Model Input and Using the Pretrained Models

After normalizing your expression data, the next step is to prepare it for use with the pretrained GeneRAIN models:

2. **Dataset Preparation and Model Inference**:
   - Use the notebook: [`notebooks/anal/anal_dataset.ipynb`](notebooks/anal/anal_dataset.ipynb).
   - This notebook provides steps to create a dataset from your normalized expression data. The generated dataset will be in the correct format for input to the GeneRAIN models.
   - Once you have your dataset ready, you can use it as input to the models.
   - The model checkpoint files, saved from the training process, will be used for making predictions or further analysis.


## ARCHS4 Human Bulk RNA-seq Data Preprocessing

While the dataset normalized by the Binning-By-Gene method, suitable for models of `GPT_Binning_By_Gene`, `BERT_Pred_Expr_Binning_By_Gene`, and `BERT_Pred_Genes_Binning_By_Gene`, is available in our Zenodo repository, you might be interested in how this dataset was prepared. If you're looking to create your own dataset using a similar method, the following steps outline the preprocessing procedure.

#### Steps for Preprocessing:

1. **Metadata Generation**:
   - Start with the Notebook [`notebooks/data/gen_gene_emb.ipynb`](notebooks/data/gen_gene_emb.ipynb).
   - This notebook will help you generate `genes_meta.tsv` and `samples_meta.tsv` files for the ARCHS4 h5 file.

1. **Filter and Normalize**:
   - Exclude single-cell RNA-seq samples and low-quality samples.
   - Perform library size normalization and log-transformation of read count values.
   - Script: `data/normalize_each_gene/1_split_expr_data_by_sample.py`

2. **Calculate Mean and Standard Deviation**:
   - Compute the mean and standard deviation of expressions by gene.
   - Calculate the z-scores for these expressions.
   - Script: `data/normalize_each_gene/2_cal_mean_std_z_on_one_chunk_of_genes.py`

3. **Gene Filtering and Data Splitting**:
   - Remove duplicate gene symbols and filter genes.
   - Combine genes and split into sample chunks for z-score normalized data.
   - Script: `data/normalize_each_gene/3_combine_genes_filter_genes_split_into_sample_chunks.py`

4. **Binning-By-Genes Normalization**:
   - Perform Binning-by-genes normalization for each gene.
   - Script: `data/normalize_each_gene/4_cal_bins_on_one_chunk_of_genes.py`

5. **Combine and Split Normalized Data**:
   - For the results from Binning-By-Genes normalization, combine genes and split into sample chunks.
   - Script: `data/normalize_each_gene/5_combine_genes_filter_genes_split_into_sample_chunks_for_bins.py`

6. **Downsampling for Binning-By-Genes Normalization**:
   - Down-sample the original expression matrix.
   - This step genenate data for finding the bin boundaries for 'Binning-By-Genes' normalization of new samples.
   - Script: `data/normalize_each_gene/6_gen_subsamples_for_each_gene_for_binning_new_sample.py`



