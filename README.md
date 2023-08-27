# PLMTHP：An Ensemble Framework for Tumor Homing Peptide Prediction based on Protein Language Model

 Tumor homing peptides (THPs) play a significant role in recognizing and specifically binding to tumor cells. Traditional experimental methods can accurately identify THPs but often suffer from high measurement costs and long experimental cycles. In-silicon methods can rapidly screen THPs, thereby accelerating the experimental process. Existing THPs prediction methods rely on constructing peptide sequence features and machine learning approaches. These methods require feature engineering and exhibit weak robustness. In this study, we proposed a method called PLMTHP (Protein Language Model of Tumor Homing Peptides) based on protein language model encoding and integrate multiple machine learning models.

</details>

<details open><summary><b>Table of contents</b></summary>


- [PLMTHP：An Ensemble Framework for Tumor Homing Peptide Prediction based on Protein Language Model](#plmthpan-ensemble-framework-for-tumor-homing-peptide-prediction-based-on-protein-language-model)
  - [Installation ](#installation-)
  - [Quick start ](#quick-start-)
  - [Training Your Own Model ](#training-your-own-model-)
  - [Citations ](#citations-)
</details>


## Installation <a name="Installation"></a>
To use the PLMTHP model, make sure you start from an environment with python = 3.7 , ESM-2, and  pytorch installed.
Then clone this repository by calling: 

```bash
git clone https://github.com/Chenyb939/PLMTHP
```
AND navigate to the cloned directory and Install the required dependencies by calling: 
```bash
cd ./PLMTHP
pip install -r requirements.txt
```
Or PLMTHP can be installed using conda by calling: 
```bash
conda install -r environment.yaml
```
## Quick start <a name="quickstart"></a>

1. Train your own PLMTHP model by calling:
```bash
python ./script/Voting_5ML.py --trainpos ./data/Processed_data/THP_train.txt --trainneg ./data/Processed_data/non_THP_train.txt --output_dir [output_dir]
```  
2. Test your model by calling:
```bash
python ./script/Voting_5ML.py --test ./data/Processed_data/test.txt  --output_dir [output_dir]
```  
## Training Your Own Model <a name="ownmodel"></a>
PLMTHP allows you to train your own models using your custom dataset. Follow these steps to train your own model:
1. Prepare your data in fasta format.
2. Train and test your own PLMTHP model useing `Voting_5ML.py`:

```bash
usage: Voting_5ML [-i_pos] FASTA [-i_neg] FASTA [-t] FASTA 
                  [-o] DIR [--model_location MODEL_LOCATION]
                  [--toks_per_batch TOKS_PER_BATCH]
                  [--repr_layers REPR_LAYERS][--include INCLUDE]
                  [--nogpu NOGPU]

optional arguments:
  -h, --help            show this help message and exit
  -i_pos FASTA, --trainpos FASTA
                        Path of pos train data file
  -i_neg FASTA, --trainneg FASTA
                        Path of neg train data file
  -t FASTA, --test FASTA
                        Path of test data file 
  -o DIR, --output_dir DIR     
                        Path to output file directory
  --model_location MODEL_LOCATION
                        PyTorch model file OR name of pretrained model to 
                        download.Default: esm2_t36_3B_UR50D.
  --toks_per_batch TOKS_PER_BATCH
                        Maximum batch size.Default: 8192.
  --repr_layers REPR_LAYERS
                        layers indices from which to extract representations 
                        (0 to num_layers, inclusive).Default: 36.
  --include INCLUDE     Specify which representations to return.Default: mean.                      
  --nogpu NOGPU         Do not use GPU even if available.Default: true.
```  

## Citations <a name="citations"></a>
The work on PLMTHP has not been published yet.