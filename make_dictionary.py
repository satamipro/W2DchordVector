

def makedictionary(traindata, chordlist):
    vocabulary = {}

    for i in range(len(traindata)):
        for j in range(len(traindata[i]) - 4):
            vocabulary.setdefault(chordlist[i][j], traindata[i][j])

    return  vocabulary