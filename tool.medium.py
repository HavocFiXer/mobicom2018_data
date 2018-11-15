import numpy as np
import sys

keyword=sys.argv[1]#'3cnn-2fc.cnn-bn.order0'
length=int(sys.argv[2])#30

result=[]

infile=open(keyword)
for line in infile:
	segs=line.strip().split(',')
	result.append((int(segs[0]),float(segs[1])))
infile.close()

s=0.0
for i in xrange(length):
	s+=result[i][1]
s=s/length

ave=[]
pos=[]
ave.append(s)
pos.append((result[0][0], result[length-1][0]))
for i in xrange(length, len(result)):
	s+=(result[i][1]-result[i-length][1])/length
	ave.append(s)
	pos.append((result[i-length+1][0],result[i][0]))
print 'max:',np.max(ave)
maxpos=np.argmax(ave)
print 'max range:', pos[maxpos]
print '0.9:',np.quantile(ave, 0.9)
print 'med:',np.median(ave)
