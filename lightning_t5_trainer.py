# Fine-tune huggingface generative LMs using pyotrch-lightning, lora, and peft

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, TrainingArguments, AutoModelWithLMHead
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from peft import PeftModel, PeftConfig
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from typing import Optional
import argparse
from pdb import set_trace as bp


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

class CosiDataset(Dataset):
	def __init__(self, data, args=None):
		self.data = data
		self.args = args

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

def rawDataToDataSplits(args):
	numTurns = args['num_turns']
	evidenceFile = args['evidence_data']

	with open(args['raw_examples'], 'r') as fin:
			examplesDict = json.load(fin)

	evidenceDf = pd.read_csv(evidenceFile, sep='\t')
		
	outArray = []
	exampleEncoded = 0
	for arbLabel, arbGroup in examplesDict.items():
		for convoNum, convoTurns in arbGroup.items():
			for turnIdx, turn in enumerate(convoTurns):
				user = turn[0]
				if 'Guide' in user and turnIdx > 0:
					speaker, response, evidence = turn
					target = speaker + ": " + response
					trueTurns = numTurns*2
					aggText = [turn[0] + ": " + turn[1] for turn in convoTurns[max(0, turnIdx-trueTurns):turnIdx]]
					context = " >> ".join(aggText)

					path = arbLabel + "/" + str(convoNum) + "/" + str(turnIdx)

					# Grabs gold evidence ids
					matchedEvidenceIds = []
					matchedEvidenceText = []
					for evidenceIdx, evidenceVal in enumerate(evidence):
						if type(evidenceVal) == list:
							evidenceTitle = evidence[evidenceIdx-1].replace("_", "-")
							for idVal in evidenceVal:
								outId = evidenceTitle + "_" + str(idVal)
								matchedEvidenceIds.append(outId)
								if args['use_topics']:
									matchedEvidenceText.append(" ".join(evidenceDf[evidenceDf['document'] == evidenceTitle]['text']))
								else:
									matchedEvidenceText.append(evidenceDf[evidenceDf['combinedId'] == outId]['text'].values[0])

					# TODO: Modify context and targets as needed
					outArray.append({"context": context, "target": target, "goldEvidenceIds": matchedEvidenceIds, 'evidenceText': matchedEvidenceText, 'examplePath': path})

	# Split into train, dev, test
	np.random.seed(args['data_seed'])
	np.random.shuffle(outArray)
	trainRatio = 1 - (2*args['dev_ratio'])
	trainData = outArray[:int(trainRatio*len(outArray))]
	devData = outArray[int(trainRatio*len(outArray)):int((trainRatio + args['dev_ratio'])*len(outArray))]
	testData = outArray[int((trainRatio + args['dev_ratio'])*len(outArray)):]

	trainDf = pd.DataFrame(trainData)
	devDf = pd.DataFrame(devData)
	testDf = pd.DataFrame(testData)
	
	trainDf.to_csv(args['train_data'], sep='\t', index=False)
	devDf.to_csv(args['dev_data'], sep='\t', index=False)
	testDf.to_csv(args['test_data'], sep='\t', index=False)
	return trainData, devData, testData

class seqTrainer(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.save_hyperparameters()
		self.args = args
		modelName = self.args['model_name']
		self.tokenizer = AutoTokenizer.from_pretrained(modelName)
		self.model = AutoModelWithLMHead.from_pretrained(modelName, load_in_8bit=True, device_map='auto')
		self.metricsDict = dict()
		self.trainDs, self.devDs, self.testDs = self.get_datasets()
		lora_config = LoraConfig(
			r=16,
			lora_alpha=32,
			target_modules=["q", "v"],
			lora_dropout=0.05,
			bias="none",
			task_type=TaskType.SEQ_2_SEQ_LM
			)
		self.model = prepare_model_for_int8_training(self.model, lora_config)
		self.model = get_peft_model(self.model, lora_config)

	def get_datasets(self):
		if self.args['generate_data_splits']:
			trainData, devData, testData = rawDataToDataSplits(self.args)
		else:
			trainData = pd.read_csv(self.args['train_data'], sep='\t').to_dict('records')
			devData = pd.read_csv(self.args['dev_data'], sep='\t').to_dict('records')
			testData = pd.read_csv(self.args['test_data'], sep='\t').to_dict('records')
		
		trainDs = CosiDataset(trainData, args=self.args)
		devDs = CosiDataset(devData, args=self.args)
		testDs = CosiDataset(testData, args=self.args)
		return trainDs, devDs, testDs

	def stage_model(self):
		modelName = self.args['model_name']

		tokenizer = AutoTokenizer.from_pretrained(modelName)
		model = AutoModelWithLMHead.from_pretrained(modelName, load_in_8bit=True, device_map='auto')
		return model, tokenizer

	def on_train_start(self):
		self.logger.log_hyperparams(self.args)
		self.logger.log_hyperparams(self.model.config.to_dict())
		
		#lora_config = LoraConfig(
		#	r=16,
		#	lora_alpha=32,
		#	target_modules=["q", "v"],
		#	lora_dropout=0.05,
		#	bias="none",
		#	task_type=TaskType.SEQ_2_SEQ_LM
		#	)
#
		#self.model = prepare_model_for_int8_training(self.model, lora_config)
		#self.model = get_peft_model(self.model, lora_config)
	
	# Used only for inference
	def forward(self, input_ids, attention_mask, labels=None):
		outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
		return outputs

	def collator(self, batch):
		tokenizedContext = self.tokenizer([x['context'] for x in batch], padding=True, truncation=True, max_length=self.args['max_tkn_length'], return_tensors='pt')
		input_ids = tokenizedContext['input_ids']
		attention_mask = tokenizedContext['attention_mask']
		labels = self.tokenizer([x['target'] for x in batch], padding=True, truncation=True, max_length=self.args['max_tkn_length'], return_tensors='pt')['input_ids']
		return input_ids, attention_mask, labels

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch
		outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs.loss

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch
		outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs.loss

		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, labels = batch
		outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs.loss

		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
		return optimizer

	def train_dataloader(self):
		return DataLoader(self.trainDs, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args['num_workers'], collate_fn=self.collator)

	def val_dataloader(self):
		return DataLoader(self.devDs, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers'], collate_fn=self.collator)

	def test_dataloader(self):
		return DataLoader(self.testDs, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args['num_workers'], collate_fn=self.collator)

	def on_save_checkpoint(self, checkpoint):
		self.model.save_pretrained(self.args['model_path'])
		self.tokenizer.save_pretrained(self.args['model_path'])
		print("SAVED MODEL")
	
	def load_peft_checkpoint(self, checkpoint):
		config = PeftConfig.from_pretrained(self.args['model_path'])
		model = AutoModelWithLMHead.from_pretrained(self.args['model_name'], load_in_8bit=True, device_map='auto')
		self.model = PeftModel.from_pretrained(model, self.args['model_path'], device_map='auto')
		self.tokenizer = AutoTokenizer.from_pretrained(self.args['model_path'])
		return self.model, self.tokenizer

def run_training(model, args):
	pl.seed_everything(args['train_seed'])

	wandb_logger = WandbLogger(name=args['logger_name'], project='cosi')
	wandb_logger.experiment.config.update(args)

	trainer = pl.Trainer(
		accelerator='auto',
		devices = [args['gpu_num']] if device == 'cuda' else None,
		max_epochs=args['max_epochs'],
		callbacks=[
			TQDMProgressBar(refresh_rate=20),
			EarlyStopping(monitor='val_loss', mode='min', patience=args['patience'], verbose=True),
			ModelCheckpoint(monitor='val_loss', dirpath=args['model_directory'],  filename=args['logger_name'] + '-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min')
		],
		logger = wandb_logger,
		deterministic=True
	)

	print("Starting training...")
	trainer.fit(model)

	print("Starting testing...")
	model.load_peft_checkpoint(args['model_path'])
	trainer.test(model)

	return model

def run_generation(model, args):
	model.eval()
	model.freeze()

	generationDf = pd.read_csv(args['generation_input'], sep='\t')
	contexts = generationDf['context'].values.tolist()

	generations = []
	batchSize = args['batch_size']
	for batchIdx in range(0, len(contexts), batchSize):
		batch = contexts[batchIdx:batchIdx+batchSize]
		tokenizedContext = model.tokenizer(batch, padding=True, truncation=True, max_length=args['max_tkn_length'], return_tensors='pt')
		input_ids = tokenizedContext['input_ids'].to(device)
		attention_mask = tokenizedContext['attention_mask'].to(device)
		outputs = model.model.generate(input_ids=input_ids, max_new_tokens=150)
		generations.extend([model.tokenizer.decode(x, skip_special_tokens=True) for x in outputs])
	generationDf['generation'] = generations
	return generationDf

def run_interface(model, args):
	model.eval()
	model.freeze()

	doc = input("Enter your document: ")
	question = input("Enter your question: ")
	while True:
		context = "Use the Article to answer the Question:" + question + " Article: " + doc
		tokenizedContext = model.tokenizer(context, padding=True, truncation=True, max_length=args['max_tkn_length'], return_tensors='pt')
		input_ids = tokenizedContext['input_ids'].to(device)
		attention_mask = tokenizedContext['attention_mask'].to(device)
		outputs = model.model.generate(input_ids=input_ids, max_new_tokens=150)
		outputString = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
		print(outputString)
		doc = input("Enter your document: ")
		question = input("Enter your question: ")

def getParameters():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_directory', type=str, default='models/',
						help='location of the model directory for storing checkpoints')
	parser.add_argument('--model_path', type=str, default='models/',
						help='location of the model checkpoint')
	parser.add_argument('--train_data', type=str, default='data/train.tsv',
						help='location of the training data, should be a tsv file from rawDataToDataSplits')
	parser.add_argument('--dev_data', type=str, default='data/dev.tsv',
						help='location of the validation data, should be a tsv file from rawDataToDataSplits')
	parser.add_argument('--test_data', type=str, default='data/test.tsv',
						help='location of the test data, should be a tsv file from rawDataToDataSplits')
	parser.add_argument('--generation_input', type=str, default='',
						help='location of the generation prompts file, should be a tsv file')
	parser.add_argument('--generation_output', type=str, default='results/generation_output.tsv',
						help='location of the generation output file, should be a tsv file')
	parser.add_argument('--evidence_data', type=str, default='data/evidence.tsv',
						help='location of the evidence data')
	parser.add_argument('--num_turns', type=int, default=2,
						help='number of conversational turns to use for context')
	parser.add_argument('--use_topics', action='store_true',
						help='whether to use topics for evidence')
	parser.add_argument('--generate_data_splits', action='store_true',
						help='whether to generate data splits from raw data')
	parser.add_argument('--model_name', type=str, default='t5-base',
						help='name of base huggingface model')
	parser.add_argument('--raw_examples', type=str, default='data/COSI_pilot_study_dataset.json',
						help='location of the raw examples in json format')
	parser.add_argument('--data_seed', type=int, default=42,
						help='random seed for splitting data into train and dev')
	parser.add_argument('--train_seed', type=int, default=42,
						help='random seed for training')
	parser.add_argument('--dev_ratio', type=float, default=0.1,
						help='fraction of train-dev data to use for dev')
	parser.add_argument('--max_epochs', type=int, default=40,
						help='upper epoch limit')
	parser.add_argument('--lr', type=float, default=.0001,
						help='initial learning rate')
	parser.add_argument('--batch_size', type=int, default=32,
						help='batch size')
	parser.add_argument('--max_tkn_length', type=int, default=512,
						help='maximum token length')
	parser.add_argument('--num_workers', type=int, default=8,
						help='number of workers for data loader')
	parser.add_argument('--patience', type=int, default=5,
						help='patience for early stopping and learning rate decay if used')
	parser.add_argument('--logger_name', type=str, default='test_t5',
						help='name for wandB logger')
	parser.add_argument('--notes', type=str, default='',
						help='notes for wandB logger')
	parser.add_argument('--gpu_num', type=int, default=1,
						help='gpu number to use')
	parser.add_argument('--run_training', action='store_true',
						help='train model')
	parser.add_argument('--run_generation', action='store_true',
						help='generate from model')
	parser.add_argument('--run_interface', action='store_true',
						help='generate from model')
	parser.set_defaults(run_training=False)
	parser.set_defaults(run_generation=False)
	parser.set_defaults(run_interface=False)
	parser.set_defaults(use_topics=False)
	parser.set_defaults(generate_data_splits=False)
	parser.set_defaults(generate_only=False)
	args = parser.parse_args()
	return vars(args)

def main():
	args = getParameters()
	if args['run_training']:
		model = seqTrainer(args)
		model = run_training(model, args)
	if args['run_generation']:
		model = seqTrainer(args)
		model.load_peft_checkpoint(args['model_path'])
		generationDf = run_generation(model, args)
		generationDf.to_csv(args['generation_output'], sep='\t', index=False)
	if args['run_interface']:
		model = seqTrainer(args)
		model.load_peft_checkpoint(args['model_path'])
		run_interface(model, args)

	return 0

if __name__ == '__main__':
	main()