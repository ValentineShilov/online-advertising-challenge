#!/usr/bin/pypy
#score from only this script 0.01520

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from pymmh3 import hash



train = 'train_p.csv'  # path to training file
test = 'test_p.csv'  # path to testing file

logbatch = 100000
dotest = True

D = 2 ** 28    # number of weights use for learning

signed = False    # Use signed hash? Set to False for to reduce number of hash calls

interaction = True

lambda1 = 0.
lambda2 = 0.

if interaction:
    alpha = .004  # learning rate for sgd optimization 0.004
else:
    alpha = .05   # learning rate for sgd optimization
adapt = 1.        # Use adagrad, sets it as power of adaptive factor. >1 will amplify adaptive measure and vice versa
fudge = .5        # Fudge factor



# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-17), 10e-17)        # The bounds
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    fullind = []
    for key, value in csv_row.items():
        #print(csv_row)
        #return
        s = key + '=' + value
        fullind.append(hash(s) % D) # weakest hash ever ?? Not anymore :P

    if interaction == True:
        indlist2 = []
        for i in range(len(fullind)):
            for j in range(i+1,len(fullind)):
                indlist2.append(fullind[i] ^ fullind[j]) # Creating interactions using XOR
        fullind = fullind + indlist2

    x = {}
    x[0] = 1  # 0 is the index of the bias term
    for index in fullind:
        if(index not in x):
            x[index] = 0
        if signed:
            x[index] += (1 if (hash(str(index))%2)==1 else -1) # Disable for speed
        else:
            x[index] += 1
    
    return x  # x contains indices of features that have a value as number of occurences


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i, xi in x.items():
        wTx += w[i] * xi  # w[i] * x[i]
    return 1. / (1. + exp(-max(min(wTx, 50.), -50.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, g, x, p, y):
    for i, xi in x.items():
        # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        delreg = (lambda1 * ((-1.) if w[i] < 0. else 1.) + lambda2 * w[i]) if i != 0 else 0.
        delta = (p - y) * xi + delreg
        if adapt > 0:
            g[i] += delta ** 2
        w[i] -= delta * alpha / (sqrt(g[i]) ** adapt)  # Minimising log loss
    return w, g


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
g = [fudge] * D  # sum of historical gradients

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
f = open(train)
header = f.readline()
header = header.strip()
header = header.replace("I", "i")
header = header.replace("C", "c")
header = header.split(",")
for t, row in enumerate(DictReader(f, header, delimiter=',')):
    y = 1. if row['Label'] == '1' else 0.

    del row['Label']  
    del row['timestamp']
    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    # step 2, get prediction
    p = get_p(x, w)

    # for progress validation
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.

    # step 3, update model 
    w, g = update_w(w, g, x, p, y)

if not dotest:
    exit()

with open('submission_lr.csv', 'w') as submission:
    submission.write('Id,Click\n')
    f = open(test)
    f.readline()
    for t, row in enumerate(DictReader(f, header, delimiter=',')):
        del row['timestamp']
        del row['Label']
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%d,%f\n' % (1+int(t), p))
