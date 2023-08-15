import re, os, sys
import numpy as np
import torch

# Read protein sequences in file
def read_protein_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        fasta_sequences.append([name, sequence])
    return fasta_sequences


# Extract the AAC feature
def AAC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    AAComposition = [aa for aa in AA]
    header = [] + AAComposition

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 20
        for j in range(len(sequence) - gap):
            c = sequence.count(sequence[j])
            tmpCode[AADict[sequence[j]]] = float(c) / len(sequence)
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header


# Extract the DPC feature
def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1 - gap):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + gap + 1]]] = tmpCode[AADict[sequence[j]] * 20 +
                                                                                        AADict[sequence[
                                                                                            j + gap + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header


# Load data and feature extraction
def extract_train(pospath, negpath):
    datapos = read_protein_sequences(pospath)
    dataneg = read_protein_sequences(negpath)

    pos, header = AAC(datapos, 0)
    aac_pos = torch.tensor(pos)
    torch.save(aac_pos, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aac_pos.pt'))

    neg, header = AAC(dataneg, 0)
    aac_neg = torch.tensor(neg)
    torch.save(aac_neg, os.path.join(os.path.dirname(os.path.abspath(__file__)),  'aac_neg.pt'))

    pos, header = DPC(datapos, 0)
    dpc_pos = torch.tensor(pos)
    torch.save(dpc_pos, os.path.join(os.path.dirname(os.path.abspath(__file__)),  'dpc_pos.pt'))

    neg, header = DPC(dataneg, 0)
    dpc_neg = torch.tensor(neg)
    torch.save(dpc_neg, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpc_neg.pt'))

# Load data and feature extraction of test
def extract_test(path):
    data=read_protein_sequences(path)

    pos, header = AAC(data, 0)
    aac = torch.tensor(pos)
    torch.save(aac, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aac.pt'))


    pos, header = DPC(data, 0)
    dpc = torch.tensor(pos)
    torch.save(dpc, os.path.join(os.path.dirname(os.path.abspath(__file__)),  'dpc.pt'))



