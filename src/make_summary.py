


def process_eval07(filename):
    
    istream = open(filename)
    las = istream.readline()
    uas = istream.readline()
    istream.close()
    
    las = las.split('=')[1]
    las = float(las[:-1])
    uas = uas.split('=')[1]
    uas = float(uas[:-1])
    return (las,uas)


print(process_eval('/tmp/eval.tmp'))
