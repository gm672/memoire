import dl
import dlm
import conll3
import csv
import argparse
import textwrap


def create(fileName1):
    # store the dependency tree in a dict
    T = conll3.conllFile2trees(fileName1)
    stats = []
    c=0
    for tree in T:
        c+=1
        l = len(tree) # length 
        d = dl.DL_T(tree) # observed sentence
        #true random
        r = dl.true_random(tree)
        dr = dl.DL_L(r,tree)
        #optimal
        linearization = dlm.optimal_linearization(tree)
        dmin = dl.DL_L(linearization,tree)
        omega = dl.omega(dmin,dr,d,l) # Omega
        gamma = dl.gamma(dmin,d) # Gamma
        mdd = dl.MDD(d,l) # MDD
        stats.append([l,d,dr,dmin,omega,gamma,mdd]) # Create the csv line
    return stats





parser = argparse.ArgumentParser(description='Get a csv file with DL measure from a CONLL file. --help for more information',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('file', 
                    help= textwrap.dedent('''\
                        The file to be analysed
                        '''))
                        
parser.add_argument('output', 
                    help= textwrap.dedent('''\
                        Output file.
                        The CSV file does not have headers.
                        It is formated as :
                        length, actual dependency length, random DL , minimum DL, omega, gamma, MDD
                        '''))




args = parser.parse_args()
print(args.file)
print(args.output)
stats = create(args.file)
with open(args.output, 'w+',newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for line in stats:
        spamwriter.writerow(line)
csvfile.close()






