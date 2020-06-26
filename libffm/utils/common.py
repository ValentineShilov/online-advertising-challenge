import hashlib, csv, math, os, pickle, subprocess



def open_with_first_line_skipped(path, skip=True):
    f = open(path)
    if not skip:
        return f
    next(f)
    return f

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_feats(row):
    feats = []
    for j in range(1, 3):
        field = 'I{0}'.format(j)
        value = row[field]
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value))**2)
            else:
                value = 'SP'+str(value)
        key = field + '-' + str(value)
        feats.append(key)
    for j in range(1, 28): # ValentineS
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats

def read_freqent_feats(threshold=10):
    frequent_feats = set()
    for row in csv.DictReader(open('fc.trva.t10.txt')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats.add(row['Field']+'-'+row['Value'])
    return frequent_feats

def split(path, nr_thread, has_header):

    def open_with_header_witten(path, idx, header):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if not has_header:
            return f 
        f.write(header)
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        if not has_header:
            nr_lines += 1 
        return math.ceil(float(nr_lines)/nr_thread)

    header = open(path).readline()

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_witten(path, idx, header)
    for i, line in enumerate(open_with_first_line_skipped(path, has_header), start=1):
        if i%nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_witten(path, idx, header)
        f.write(line)
    f.close()

def parallel_convert(cvt_path, arg_paths, nr_thread):

    workers = []
    for i in range(nr_thread):
        cmd = '{0}'.format(os.path.join('.', cvt_path))
        for path in arg_paths:
            cmd += ' {0}'.format(path+'.__tmp__.{0}'.format(i))
        worker = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        workers.append(worker)
    for worker in workers:
        worker.communicate()

def cat(path, nr_thread):
    
    if os.path.exists(path):
        os.remove(path)
    for i in range(nr_thread):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

def delete(path, nr_thread):
    
    for i in range(nr_thread):
        os.remove('{0}.__tmp__.{1}'.format(path, i))

