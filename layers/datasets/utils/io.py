import pickle
import json

def readPKL(path):
	with open(path, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	return data

def writePKL(file, data):
	with open(file, 'wb') as f:
		pickle.dump(data, f)

def readJSON(path):
	with open(path, 'r') as f:
		data = json.load(f)
	return data

def writeJSON(file, data, **kargs):
	with open(file, 'w') as f:
		json.dump(data, f, **kargs)