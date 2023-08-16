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
usage: Voting_5ML [-i_pos] FASTA [-o] DIR [--num-recycles NUM_RECYCLES]
                  [--max-tokens-per-batch MAX_TOKENS_PER_BATCH]
                  [--chunk-size CHUNK_SIZE] [--cpu-only] [--cpu-offload]

optional arguments:
  -h, --help            show this help message and exit
  -i_pos FASTA, --trainpos FASTA
                        Path of pos train data file
  -o PDB, --pdb PDB     Path to output PDB directory
  --num-recycles NUM_RECYCLES
                        Number of recycles to run. Defaults to number used in
                        training (4).
  --max-tokens-per-batch MAX_TOKENS_PER_BATCH
                        Maximum number of tokens per gpu forward-pass. This
                        will group shorter sequences together for batched
                        prediction. Lowering this can help with out of memory
                        issues, if these occur on short sequences.
  --chunk-size CHUNK_SIZE
                        Chunks axial attention computation to reduce memory
                        usage from O(L^2) to O(L). Equivalent to running a for
                        loop over chunks of of each dimension. Lower values
                        will result in lower memory usage at the cost of
                        speed. Recommended values: 128, 64, 32. Default: None.
  --cpu-only            CPU only
  --cpu-offload         Enable CPU offloading
```  

## Citations <a name="citations"></a>
The work on PLMTHP has not been published yet.