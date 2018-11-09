import numpy as np

exp_size=5 #1-5
antenna_size=2 #1-2
act_size=6 #0-5
sub_size=40 #0-39

act_label_file=open('activityId.txt')
sub_label_file=open('subjectId.txt')
data_file=[]
for exp_no in xrange(exp_size):
	data_file.append([])
	for ant_no in xrange(antenna_size):
		data_file[exp_no].append(open('./exp%d/antenna%d.txt'%(exp_no+1, ant_no+1)))

act_label_set=np.zeros((act_size,act_size))
for i in xrange(act_size):
	act_label_set[i,i]=1.0
sub_label_set=np.zeros((sub_size,sub_size))
for i in xrange(sub_size):
	sub_label_set[i,i]=1.0

datall=[]
act_labelall=[]
sub_labelall=[]

lineno=0
for act_label_line in act_label_file:
	lineno+=1
	sub_label_line=sub_label_file.readline()
	act_label=act_label_set[int(act_label_line)]
	sub_label=sub_label_set[int(sub_label_line)]
	for exp_no in xrange(exp_size):
		data_piece=[]
		for ant_no in xrange(antenna_size):
			data_line=data_file[exp_no][ant_no].readline()
			data_seg=data_line.strip().split(',')
			mid=(len(data_seg)+1)/2
			data_list=list(map(float, data_seg))
			data_piece.append(np.array(data_list[:mid]).reshape((128, 30)))
			data_piece.append(np.array(data_list[mid:]).reshape((128, 30)))
		data_piece=np.transpose(np.array(data_piece), (1,2,0))
		datall.append(data_piece)
		act_labelall.append(act_label)
		sub_labelall.append(sub_label)
	if lineno%100==0:
		print 'lineno:',lineno

act_label_file.close()
sub_label_file.close()
for exp_no in xrange(exp_size):
	for ant_no in xrange(antenna_size):
		data_file[exp_no][ant_no].close()

datall=np.array(datall)
act_labelall=np.array(act_labelall)
sub_labelall=np.array(sub_labelall)
print 'datall:',datall.shape
print 'act_labelall:', act_labelall.shape
print 'sub_labelall:', sub_labelall.shape

outfile=open('databel.dat','wb')
np.save(outfile, datall)
np.save(outfile, act_labelall)
np.save(outfile, sub_labelall)
outfile.close()
