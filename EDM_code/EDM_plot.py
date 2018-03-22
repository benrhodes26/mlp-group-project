import numpy as np
import matplotlib.pyplot as plt

outfile_path = './student_model_out_file'
outfile = open(outfile_path, "r")

train_auc = []
test_auc = []
previous_line =""

while True:
    line = outfile.readline()
    if not line: break
    splitted_line = line.split('auc: ')
    if len(splitted_line)>1:
        auc_val = float(splitted_line[1].split(' ')[0])
        if previous_line.find('Train')>-1:
            print(previous_line,auc_val)
            train_auc.append(auc_val)
        elif previous_line.find('Test')>-1:
            print(previous_line,auc_val)
            test_auc.append(auc_val)
    previous_line=line
print(len(train_auc), len(test_auc))
first_train_auc = train_auc[0]
for i, auc in enumerate(train_auc):
    print(i, auc)
train_auc=[auc for i, auc in enumerate(train_auc) if (i+1)%5==0]
train_auc.insert(0, first_train_auc)
test_auc.insert(0, 0.566)
e = np.arange(len(train_auc))



plt.figure()
train_plt, = plt.plot(e, train_auc)
valid_plt, = plt.plot(e, test_auc)
plt.legend([train_plt, valid_plt], ['train', 'valid'])
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC per epoch')
plt.savefig('./auc.png')

outfile.close()
