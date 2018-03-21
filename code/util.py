from random import shuffle

def getKeyset(map):
	keyset=[]
	n=1
	for k,v in enumerate(map):
	  keyset[n]=k
	  n=n+1

	return keyset



def semiSortedMiniBatches(dataset, mini_batch_size, trimToBatchSize):

	#round down so that minibatches are the same size
	trimmedAns = []
	if trimToBatchSize:
		nTemp = len(dataset)
		maxNum = nTemp - (nTemp % mini_batch_size)
		shuffled = shuffle(getKeyset(dataset))
		for i,s in ipairs(shuffled):
			if i <= maxNum:
                trimmedAns.append(dataset[s])
				table.insert(trimmedAns, dataset[s])


	else
		trimmedAns = dataset;


	def compare(a,b)
	  return a['n_answers'] < b['n_answers']

	table.sort(trimmedAns, compare)

	miniBatches = {}
	for j=1,#trimmedAns,mini_batch_size do
		miniBatch = {}
		for k = j, j + mini_batch_size - 1 do
			table.insert(miniBatch, trimmedAns[k])

		table.insert(miniBatches, miniBatch)


	shuffledBatches = {}
	for i, s in ipairs(shuffle(getKeyset(miniBatches))) do
		table.insert(shuffledBatches, miniBatches[s])


	return shuffledBatches
