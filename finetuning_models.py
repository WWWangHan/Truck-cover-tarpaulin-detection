import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser


class Logger(object):
	
	def __init__(self, filename='default.log', stream=sys.stdout):
		self.path = 'tarp_detect/log/'
		self.terminal = stream
		self.log = open(self.path + filename, 'a') # specific where to save the log
	
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)
	
	def flush(self):
		pass


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

	since = time.time()

	val_acc_history = []
	test_acc_history = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					if is_inception and phase == 'train':
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4 * loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)
			
					# torch.max(a, 1): returns the element of the largest value in each row and returns its column index
					_, preds = torch.max(outputs, 1)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data) 
			
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print('{} Loss: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc >= best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
				
				if not os.path.exists('tarp_detect/'):
					os.mkdirs('tarp_detect/model/')
					os.mkdirs('tarp_detect/input/photo/')
					os.mkdirs('tarp_detect/log/')
				torch.save(best_model_wts, './tarp_detect/model/tarp.ckpt')
			
			if phase == 'val':
				val_acc_history.append(epoch_acc)

			if phase == 'test':
				test_acc_history.append(epoch_acc)

		print()

		time_elapsed = time.time()  - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		model.load_state_dict(best_model_wts)
		
		return model, val_acc_history, test_acc_history 


def set_parameter_requires_grad(model, feature_extracting):
	
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

	'''
		you can download the corresponding model from, eg: https://download.pytorch.org/models/resnet18-5c106cde.pth
		place the pth file in /home/your_name/.cache/torch/checkpoints/resnet18-5c106cde.pth
	'''

	model_ft = None
	input_size = 0

	if model_name == 'resnet':
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224
	
	elif model_name == 'alexnet':
		model_ft = models.alexnet(pretrained=use_pretrained)	
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == 'vgg':	
		model_ft = models.vgg11_bn(pretrained=use_pretrained)	
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == 'squeezenet':
		model_ft = models.squeezenet1_0(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == 'densenet':	
		model_ft = models.densenet121(pretrained=use_pretrained)	
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features
		model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224
	
	elif model_name == 'inception':
		# (299, 299) sized images have auxiliary output
		model_ft = models.inception_v3(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		# handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features
		model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
		# handle the primary net
		num_ftrs = model_ft.fc.in_features
		model_fc.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 299

	else:
		print('Invalid model name, existing...')
		exit()
	
	return model_ft, input_size


if __name__ == '__main__':
	
	# get configurations from config.ini
	cp = ConfigParser()
	cp.read('./cfg/config.ini', encoding='utf-8')

	data_dir = cp.get('parameter', 'data_dir')
	model_name = cp.get('parameter', 'model_name')
	feature_extract = cp.get('parameter', 'feature_extract')
	num_classes = int(cp.get('parameter', 'num_classes'))
	batch_size = int(cp.get('parameter', 'batch_size'))
	num_epochs = int(cp.get('parameter', 'num_epochs'))

	out_log_name = str(os.path.basename(__file__).split('.')[0] + '_' + str(model_name) + '.log')
	err_log_name = str(os.path.basename(__file__).split('.')[0] + '_' + str(model_name) + '_error.log')
	sys.stdout = Logger(out_log_name, sys.stdout)
	sys.stderr = Logger(err_log_name, sys.stderr)

	torch.multiprocessing.freeze_support()
	
	# initialize the model
	model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True) 

	print(model_ft)

	data_transforms = {
		
		'train': transforms.Compose(
			[
				transforms.RandomResizedCrop(input_size),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
		),
		'val': transforms.Compose(
			[
				transforms.Resize(input_size),
				transforms.CenterCrop(input_size),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
		),
		'test': transforms.Compose(
			[
				transforms.Resize(input_size),
				transforms.CenterCrop(input_size),
				transforms.ToTensor(),	
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]
		),	
	}

	# the default data set has been automatically divided into different folders according to the type to be allocated 
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

	dataloaders_dict = {
		x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
		for x in ['train', 'val', 'test']
	}
	
	device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
	
	model_ft = model_ft.to(device)
	params_to_update = model_ft.parameters()

	if feature_extract:
		params_to_update = []
		for name, param in model_ft.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print('\t', name)
	else:
		for name, param in model_ft.named_parameters():
			if param.requires_grad == True:
				print('\t', name)

	optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
	criterion = nn.CrossEntropyLoss()

	model_ft, val_hist, test_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=='inception'))

	# painting
	ohist = [h.cpu().numpy() for h in val_hist]
	ohist2 = [h.cpu().numpy() for h in test_hist]

	plt.title('val and test acc vs. epoch numbers')
	plt.xlabel('training epoch')
	plt.ylabel('validation acc')
	plt.plot([range(1, num_epochs + 1)], ohist, label='Val')
	plt.plot([range(1, num_epochs + 1)], ohist2, label='Test')
	plt.ylim((0, 1.))
	plt.xticks(np.arange(1, num_epochs + 1, 1.0))
	plt.legend()
	plt.show()

	print('Program finished.')



























