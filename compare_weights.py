import torch

def compare_weigts(file1, file2):
	w1 = torch.load(file1).items()
	w2 = torch.load(file2).items()

	models_differ = 0
	for key_item_1, key_item_2 in zip(w1, w2):
		if torch.equal(key_item_1[1], key_item_2[1]):
			pass
		else:
			models_differ += 1
			if (key_item_1[0] == key_item_2[0]):
				#print('Mismtach found at', key_item_1[0])
				pass
			else:
				raise Exception("Weights shape is not the same ...")
	if models_differ == 0:
		print('Models match perfectly! :)')
	else:
		print("We found", models_differ, "differences.")