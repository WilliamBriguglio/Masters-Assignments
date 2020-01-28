import pandas as pd
import numpy as np
import sys
import re

def SampPadding(sample): #determine the padding of 00's that are in the sample's input tensor
    for n, i in enumerate(sample.flatten()):
        if i != 0:
            return n

def bytesPadandStartIndicie(data): #determine the padding of 00's that are are already in the sample's bytes file
    for n, i in enumerate(data):
        if i != 0:
            return n

def findCodeSnips(addr, asmFP, ngram):

    padding=3
    nGramSize=6
    addrLength=8

    #read asm file
    fp = open(asmFP, "r", errors='replace')
    asm = fp.read()
    fp.close()

    potentialAddrs = []

    #get addresses of lines which potentially contain ngram
    offset = 0
    while(offset < nGramSize):
        straddr = str(hex((int(addr, 16) + offset)))[2:].upper()
        potentialAddrs.append("0" * (addrLength-len(straddr)) + straddr)
        offset+=1
    

    snippet = ""
    lastMatch = ""

    s = 1 #being set to 1 indicates we are looking for start of ngram and its preceeding padding
    m = 0 #being set to 0 indicates we haven't found the start of the ngram
    nf = 0 #flag indicating part of the ngram is on the line before the first found potential address
    for i in potentialAddrs:
        if m == 1: #if we found the start of the ngram, add lines which contain remaining potential addresses
            searchReg = r"\.[A-Za-z]+:" +i+ r".*\n" #matching the line with the address i
            match = re.search(searchReg, asm)
            if match: #add previously matched line only when we match another line, else it isnt added until after the loop
                snippet += lastMatch 
                lastMatch = match.group()

        if s == 1: #if looking for start of 6-gram (i.e. first potential address not matched)
            
            #(".*\n" * padding)         grab [padding] lines before the first line containing part of the ngram. 
            #(nf * ".*\n")              grab 1 line extra if part of the ngram is found on the line before the first matched address
            #"\.[A-Za-z]+:" +i+ ".*\n   matching the line with the address i
            searchReg = (r".*\n" * padding) + (nf * r".*\n") + r"\.[A-Za-z]+:" +i+ r".*\n"

            nf = 1 #if first potential address is not found, then set nf flag if not already set
            match = re.search(searchReg, asm)
            if match:
                snippet += match.group()
                s = 0 #being set to 0 indicates the start of the n-gram and padding before ngram were found
                m = 1 #indicates we are now looking for lines which contain the rest of the ngram


    if snippet != "": #if code snippit found #add the last line and remaining padding
        searchReg = lastMatch + (r".*\n" * padding)
        match = re.search(searchReg, asm)
        snippet += match.group()
    else: #if code snippet not found, this indicates the entire ngram does not appear at the beginning of a line, but is on one line so we can search as so...
        searchReg = (r".*\n" * padding) + r'\.[A-Za-z]+:.*' + ngram + r'.*\n' + (r'.*\n' * padding) #grab pre padding, matched ngram, and post padding
        match = re.search(searchReg, asm)
        if match:
            snippet += match.group()
        else: #if still not found, return error signal
            return int(-1)
    return snippet

print("\n")


relFP = "interpret/A_2.npy" #the file containg the relvances of the features of 0AnoOZDNbPXIr2MRBSCJ
SampFP = "interpret/X_2.npy" #the file containg the relvances of the input tensor of 0AnoOZDNbPXIr2MRBSCJ
byteFP = "interpret/0AnoOZDNbPXIr2MRBSCJ.bytes"
asmFP = "interpret/0AnoOZDNbPXIr2MRBSCJ.asm"

Sample = np.load(SampFP)
relevances = np.load(relFP)

fp = open(byteFP, "r")
bytes = fp.read()
bytesCopy = bytes
fp.close()

firstaddr = int(bytes[:8], 16) #get the starting address in bytes file

#clean data
bytes = re.sub(r'([0-9]|[A-F]){8}',"",bytes)
bytes = re.sub(r'\n'," ",bytes)
bytes = re.sub(r'  '," ",bytes)
bytes = re.sub(r'\?\?',"00", bytes)
	
#convert to array of hex pixels
data = [int(token, 16) for token in bytes.split(" ") if token != ""]

relevances = np.average(relevances, axis = -1).flatten() #average relevances across 6 channels
maxRels = np.argsort(relevances, axis = None) #sort in ascending order

for n, i in enumerate(np.flip(maxRels)[int(sys.argv[1]):]): #starting at the indicie of the n^th most relevant ngram, where n is spcified by sys.argv[1]
    
    _6gram = ""
    for j in Sample[int(i/183),i%183]: #get 6-gram from Sample and display as hex
        _6gram+=hex(int(j))[2:].upper()+" "
    _6gram = _6gram[:-1]

    padInSamp = SampPadding(Sample) #determine # of bytes of padding in Sample
    padInBytes = bytesPadandStartIndicie(data) #determine # of bytes of 00's in binary
    addr = (i * 6) - (padInSamp - padInBytes) #determine address of 6-gram starting from 0
    addrInBytes = hex(addr + firstaddr).upper()[2:] #add starting addres from bytes file

    print(str(n+1)+") 6gram",_6gram,"at","0x"+addrInBytes)
    print("\thas relevance:",relevances[i])

    #find and print corresponging line in .bytes file
    reg = r'' + addrInBytes[:-1]+r"0" + r"[A-F0-9 ]*" + _6gram + r"[A-F0-9 ]*"
    line = re.search(reg, bytesCopy)
    if line:
        print("   Line in .bytes file:")
        print("\t"+line.group(0))
    else:
        reg = r'' + addrInBytes[:-1]+r"0"+r".*\n.*\n"
        line = re.search(reg, bytesCopy)
        print("   Lines in .bytes file:")
        print("\t"+line.group(0))

    #find corresponding code snippet
    codeSnip = findCodeSnips(addrInBytes, asmFP, _6gram)

    if codeSnip == -1:
        print("   ERROR: NO MATCH FOUND: Address of ngram could be in collapsed section of .asm file\n")
    elif codeSnip != -1:
        print("\n   Lines in .asm file:")
        print(codeSnip)
    
    if n+1 >= int(sys.argv[2]): #stop when a total of sys.argv[2] ngrams are processed
        print("")
        break