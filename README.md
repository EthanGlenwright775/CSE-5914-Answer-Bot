# Retrieval Augmented Generation (RAG) Based Document ChatBot
## Deployment Steps
These steps have been tested to work on the Ohio Super Computer (OSC) system. If you are not using OSC, you will have to follow similar yet slightly different steps to get the project running, for example downloading a different version of CUDA-enabled pytorch among other potential differences. The OSC system uses a BASH shell, so if you are using windows os then all the shell commands will be different.
### Synthetic Data Generation Pipeline
1. Clone the GitHub repository
```bash
git clone https://github.com/EthanGlenwright775/CSE-5914-Answer-Bot.git
```
2. Load necessary OSC modules
```bash
module load miniconda3 cuda
```
3. Create a new conda environment for the pipeline
```bash
conda env create -f pipeline_environment.yml
```
4. Request resources from OSC
```bash
sinteractive -A <project_name> -g 1
```
5. Reload the necessary modules on the partition granted by OSC
```bash
module load miniconda3 cuda
```
6. Acitvate the environment
```bash
conda activate pipeline_1
```
7. Modify the arguments in pipeline.sh
- article_index: starting point of article db from which to pull articles
- article_count: number of articles from which to generate questions
- thread_count: number of threads to use
- q_eval_threshold: minimum BERT score necessary for QA pairs to be kept in training data
- ouput_directory: output directory for training data files
- training_file: name of training data file
- validation_file: name of validation data file
- testing_file: name of testing data file
- article_db: article database from which to pull articles (cnn_news, daily_mail, cc_news)
- qa_gen_method: methodology by which to generate QA pairs (rephrase, summarize)
8. Run pipeline
```bash
bash pipeline.sh
```
9. Review the generated data in the output directory that you specified in step 7
### Model Training and Interface
1. Clone the GitHub repository
```bash
git clone https://github.com/EthanGlenwright775/CSE-5914-Answer-Bot.git
```
2. Load necessary OSC modules
```bash
module load miniconda3 cuda
```
3. Create a new conda environment for the pipeline
```bash
conda env create -f training_environment.yml
```
4. Request resources from OSC
```bash
sinteractive -A <project_name> -g 1
```
5. Reload the necessary modules on the partition granted by OSC
```bash
module load miniconda3 cuda
```
6. Acitvate the environment
```bash
conda activate training_1
```
7. Modify the arguments in trainSeq2Seq.sh
8. Run training
```bash
bash trainSeq2Seq.sh
```
9. After training has completed, modify and then run interfaceSeq2Seq.sh
```bash
bash interfaceSeq2Seq.sh
```
10. Submit articls and ask questions to the model!
## Acknowledgements
Credit to Dr. Eric Fosler-Lussier and Amad Hussain for their guidance throughout this project.
Additional credit to Amad Hussain for providing some portions of the repository including lightning_t5_trainer.py, training_environment.yml, and trainSeq2Seq.sh