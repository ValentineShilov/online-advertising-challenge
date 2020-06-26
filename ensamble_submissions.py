files = ['submission_lr.csv', 'submission_ffm.csv'] 
weights = [0.6, 0.4]

fds = []
for file in files:
    fds.append(open(file, "r"))

for fd in fds:
    header = fd.readline().strip()
    
with open("final8.csv", "w") as g:
    g.write(header + "\n")
    
    flag=True
    
    while(flag):
        final_score = 0.0
        scores = [0.0]*len(weights)
        for i,fd in enumerate(fds):
            try:
                line = fd.readline().strip()
                parts = line.split(',')
                if len(parts)>1:
                    uid,score = int(parts[0]), float(parts[1])
                    scores[i] = (score)
                else:
                    flag=False
                    break
            except Exception as e:
                flag=False
                print(repr(e))
                break
        if not flag:
            break
        for i,score in enumerate(scores):
            final_score += weights[i] * score
        g.write(str(uid) + "," + str(final_score) + "\n")
        if uid % 1000000==0:
            print(uid)
