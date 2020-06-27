import csv
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tree import ParentedTree
from nltk.classify import NaiveBayesClassifier
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

def dataSet(filePath):
    f = open(filePath, "r", encoding="utf-8")
    rdr = csv.reader(f, delimiter="\t")
    dataDict = []
    for i, line in enumerate(rdr):
        if i == 0:
            continue
        dataDictElem = {
            "id": line[0],
            "text": line[1],
            "pn": line[2],
            "pnOfs": int(line[3]),
            "A": line[4],
            "aOfs": int(line[5]),
            "aCor": True if line[6] == "TRUE" else False,
            "B": line[7],
            "bOfs": int(line[8]),
            "bCor": True if line[9] == "TRUE" else False,
            "url": line[10]
        }
        dataDict.append(dataDictElem)
    f.close()
    return dataDict

def makeConstTree(sent, predictor):
    constResult = predictor.predict(
        sentence = sent
    )
    return constResult["trees"]

def makeTrees(sTokens, predictor):
    constTrees = []
    for i, sent in enumerate(sTokens):
        tree = makeConstTree(sent, predictor)
        constTrees.append(tree)
    return constTrees

def loadTreeList(mode):
    if mode == "test":
        ptfile = open("testparsetree.csv", "r", encoding="utf-8")
    elif mode == "train":
        ptfile = open("trainparsetree.csv", "r", encoding="utf-8")
    else:
        ptfile = open("validateparsetree.csv", "r", encoding="utf-8")
    ptreader = csv.reader(ptfile)
    rawTree = []
    for line in ptreader:
        rawTree.append(line)
    return rawTree

def makeOutput(dataDict, outputFileName):
    output = open(outputFileName, "w", encoding="utf-8", newline="")
    wr = csv.writer(output)
    for e in dataDict:
        wr.writerow([e["id"]])
        wr.writerow([e["text"]])
        wr.writerow([e["pn"], e["pnOfs"]])
        wr.writerow([e["a"], e["aOfs"], e["aCor"]])
        wr.writerow([e["b"], e["bOfs"], e["bCor"]])
        wr.writerow([])
    output.close()

    # return constResult["trees"]
'''
1. Pronoun이 주어일 경우 앞 문장의 주어가 coref일 가능성 높음
2. Pronoun이 전치사구로 시작하는 문장의 전치사구에 있을 경우 해당 문장의 주어가 coref일 가능성 높음
3. Pronoun이 목적어일 경우 재귀형태가 아닌 이상 해당 문장의 주어는 coref가 아님
4. Pronoun이 있는 문장에 선행사 후보가 하나일 경우 해당 선행사가 coref일 가능성 높음
5. A and C ~~~~~. Pronoun is ~~~~~일 경우 A는 coref가 아님. 오히려 앞문장의 목적어가 coref일 가능성 있음

'''

def findLocation(ofs, sTokens):
    j = 0
    sumlen = len(sTokens[j])
    while ofs > sumlen:
        j += 1
        if j == len(sTokens):
            return j-1
        else:
            sumlen += len(sTokens[j])
    return j

def isContain(sent, word):
    token = word.split(" ")
    if word in sent:
        return True
    for t in token:
        if t in sent:
            return True
    return False

def findMoreLocation(sTokens, EXAMPLE_DATA):
    locList = [findLocation(EXAMPLE_DATA["pnOfs"], sTokens), findLocation(EXAMPLE_DATA["aOfs"], sTokens), findLocation(EXAMPLE_DATA["bOfs"], sTokens)]
    result = []
    if locList[1] > locList[0]:
        loc = locList[1]
        for j, sent in enumerate(sTokens):
            if j > locList[0]:
                break
            contain = isContain(sent, EXAMPLE_DATA["A"])
            if contain:
                result.append(True)
                break
        if len(result) == 0:
            result.append(False)
    else:
        result.append(True)

    if locList[2] > locList[0]:
        loc = locList[2]
        for j, sent in enumerate(sTokens):
            if j > locList[0]:
                break
            contain = isContain(sent, EXAMPLE_DATA["B"])
            if contain:
                result.append(True)
                break
        if len(result) == 1:
            result.append(False)
    else:
        result.append(True)
    return locList, result    

def isSubject(candidate, treeStr):
    try:
        constTree = ParentedTree.fromstring(treeStr)
    except:
        return False
    else:
        nps = []
        for subtree in constTree:
            if type(subtree) != ParentedTree:
                break
            elif subtree.label() == "VP":
                break
            else:
                if subtree.label() == "NP":
                    nps.append(subtree)
        subject = [" ".join(elem.leaves()) for elem in nps if len(nps) > 0]
        for s in subject:
            if candidate in s:
                return True
        return False

def findSmallestPhrase(tree, pn):
    st = [subtree for subtree in tree.subtrees() if type(subtree) == ParentedTree and subtree.label() == "NP" and pn in subtree.leaves()]
    leaves = [len(" ".join(leaf.leaves())) for leaf in st]
    if len(leaves) == 0:
        st = [subtree for subtree in tree.subtrees() if type(subtree) == ParentedTree and subtree.label().startswith("PRP") and pn in subtree.leaves()]
        leaves = [len(" ".join(leaf.leaves())) for leaf in st]
    idx = leaves.index(min(leaves))
    return st[idx]

def inSamePhrase(treeStr, pn, candidate):
    try:
        constTree = ParentedTree.fromstring(treeStr)
    except:
        return False
    else:
        phrase = findSmallestPhrase(constTree, pn)
    if candidate in " ".join(phrase.leaves()):
        return True
    else:
        return False

def makeSamePhraseFeat(featureDict, treeStr, pn, candidate):
    if featureDict["pntype"] != "p":
        featureDict["samePhrase"] = False
        return featureDict
    result = inSamePhrase(treeStr, pn, candidate)
    if featureDict["dist"] != 0:
        result = False
    featureDict["samePhrase"] = result
    return featureDict

def isInPP(candidate, treeStr):
    try:
        constTree = ParentedTree.fromstring(treeStr)
    except:
        return False
    else:
        firstPP = constTree[0] if type(constTree) == ParentedTree and (constTree[0].label() == "PP" or constTree[0].label().startswith("S")) else None
        if type(firstPP) == ParentedTree and candidate in " ".join(firstPP.leaves()):
            return True
        else:
            return False

def makepnPPFeat(featureDict, sentTree, candidate):
    result = isInPP(candidate, sentTree)
    featureDict["pnPP"] = result
    return featureDict

def checkpnPPSubject(aCor, bCor, ft_pa, ft_pb):
    ispnPP = ft_pa["pnPP"]
    isASub = ft_pa["candidateSubject"]
    isBSub = ft_pb["candidateSubject"]
    if ispnPP == True and isASub == True and ft_pa["dist"] == 0:
        aCor = True
    if ispnPP == True and isBSub == True and ft_pb["dist"] == 0:
        bCor = True
    if aCor == True and bCor == True:
        aCor = None
        bCor = None
    return aCor, bCor

def pronounSubject(subjectList):
    if subjectList[0]:
        if subjectList[1] and subjectList[2]:
            return "plural"
        elif subjectList[1]:
            return "a"
        elif subjectList[2]:
            return "b"
        else:
            return "c"
    else:
        return "c"

def nominate(candidate, locList, subjectList):
    isNominate = candidate.lower() in ["he", "she"]
    if isNominate:
        if locList[0] == locList[1] and subjectList[1]:
            return "a"
        elif locList[0] == locList[2] and subjectList[2]:
            return "b"
        else:
            return "c"
    else:
        return "c"

def pronounPP(ppList):
    if ppList[0]:
        if ppList[1] and ppList[2]:
            return "a"
        elif ppList[3] and ppList[4]:
            return "b"
        else:
            return "c"
    else:
        return "c"

def generalCase(text, pn, candidateType, candidate, ofsList):
    betWords = ""
    if candidateType == "A":
        if ofsList[0] > ofsList[1]:
            betWords = text[ofsList[1]+len(candidate):ofsList[0]]
        else:
            betWords = text[ofsList[0]+len(pn):ofsList[1]]
    elif candidateType == "B":
        if ofsList[0] > ofsList[2]:
            betWords = text[ofsList[2]+len(candidate):ofsList[0]]
        else:
            betWords = text[ofsList[0]+len(pn):ofsList[2]]
    gender = "F" if pn.lower() in ["she", "her"] else "M"
    postag = pos_tag(word_tokenize(betWords))
    # print (postag)
    nps = [w for w, p in postag if (gender == "F" and (p in ["NNP"] or w.lower() in ["she", "her"])) or (gender == "M" and (p in ["NNP"] or w.lower() in ["he", "his", "him"]))]
    return len(nps)

def makeGeneralFeat(featureDict, text, pn, candidate, ofsList, candidateType):
    generalFeat = generalCase(text, pn, candidateType, candidate, ofsList)
    featureDict["general"] = generalFeat
    return featureDict

def makeDistFeat(featureDict, locList, isPrevious, candidateType):
    if candidateType == "A":
        featureDict["dist"] = locList[1]
        featureDict["previous"] = isPrevious[0]
    else:
        featureDict["dist"] = locList[2]
        featureDict["previous"] = isPrevious[1]
    featureDict["PN_dist"] = locList[0]
    return featureDict

#s: subject p: possessive o: objective
def pronounType(pn, EXAMPLE_TEXT, pnOfs):
    if pn.lower() in ["he", "she"]:
        return "s"
    elif pn.lower() == "his":
        return "p"
    elif pn.lower() == "him":
        return "o"
    else:
        subtext = EXAMPLE_TEXT[pnOfs+len(pn):]
        if subtext[0] != " ":
            return "o"
        else:
            subtext = subtext[1:]
            tag = pos_tag(subtext.split(" "))
            if tag[0][1] in ["IN", "TO", "DT", "CC", "WDT", "WP", "WP$", "WRB"]:
                return "o"
            else:
                return "p"

def makepronounType(featureDict, pnType):
    featureDict["pntype"] = pnType
    return featureDict

def makePnSubjectFeat(featureDict, sentTree, pn):
    checkSubject = isSubject(pn, sentTree)
    if pn[0].isupper():
        checkSubject = True
    featureDict["pnSubject"] = checkSubject
    return featureDict

def makeSubjectFeat(featureDict, sentTree, candidate):
    checkSubject = isSubject(candidate, sentTree)
    featureDict["candidateSubject"] = checkSubject
    return featureDict

def makeOrderFeat(featureDict, locList):
# def makeOrderFeat(featureDict, ofsList):
    pnLoc = locList[0]
    aLoc = locList[1]
    bLoc = locList[2]
    # pnLoc = ofsList[0]
    # aLoc = ofsList[1]
    # bLoc = ofsList[2]
    order = "Undefined"
    if aLoc == bLoc and pnLoc >= aLoc:
    # if pnLoc > aLoc and pnLoc > bLoc:
        order = "LastLoc"
    elif aLoc == bLoc and bLoc == pnLoc:
    # elif pnLoc > bLoc and pnLoc > aLoc:
        order = "FirstLoc"
    elif (pnLoc == aLoc and pnLoc > bLoc) or (pnLoc == bLoc and pnLoc > aLoc):
    # elif (pnLoc < aLoc and pnLoc > bLoc) or (pnLoc < bLoc and pnLoc > aLoc):
        order = "MiddleLoc"
    featureDict["order"] = order
    return featureDict

#page-context
def makeEntityFeat(featureDict, candidate, url):
    token = url.split("/")[-1]
    if candidate.lower().strip() in token.lower().strip():
        featureDict["entity"] = True
    else:
        featureDict["entity"] = False
    return featureDict

# candidate: "A" or "B"
def makeAllFeats(candidate, EXAMPLE_DATA, loadedTreeList, num):
    EXAMPLE_ID = EXAMPLE_DATA["id"]
    EXAMPLE_TEXT = EXAMPLE_DATA["text"]
    EXAMPLE_PN = EXAMPLE_DATA["pn"]
    EXAMPLE_POFS = EXAMPLE_DATA["pnOfs"]
    EXAMPLE_A = EXAMPLE_DATA["A"]
    EXAMPLE_AOFS = EXAMPLE_DATA["aOfs"]
    EXAMPLE_ACOR = EXAMPLE_DATA["aCor"]
    EXAMPLE_B = EXAMPLE_DATA["B"]
    EXAMPLE_BOFS = EXAMPLE_DATA["bOfs"]
    EXAMPLE_BCOR = EXAMPLE_DATA["bCor"]
    ofsList = [EXAMPLE_POFS, EXAMPLE_AOFS, EXAMPLE_BOFS]
    featureDict = {}

    sTokens = sent_tokenize(EXAMPLE_TEXT)
    locList, isPrevious = findMoreLocation(sTokens, EXAMPLE_DATA)
    featureDict = makeDistFeat(featureDict, locList, isPrevious, candidate)
    pntype = pronounType(EXAMPLE_DATA["pn"], EXAMPLE_DATA["text"], EXAMPLE_DATA["pnOfs"])
    featureDict = makepronounType(featureDict, pntype)
    featureDict = makeSamePhraseFeat(featureDict, loadedTreeList[num][locList[0]], EXAMPLE_DATA["pn"], EXAMPLE_DATA[candidate])
    featureDict = makePnSubjectFeat(featureDict, loadedTreeList[num][locList[0]], EXAMPLE_DATA["pn"])
    if candidate == "A":
        featureDict = makeSubjectFeat(featureDict, loadedTreeList[num][locList[1]], EXAMPLE_DATA[candidate]) 
    else:
        featureDict = makeSubjectFeat(featureDict, loadedTreeList[num][locList[2]], EXAMPLE_DATA[candidate])
    featureDict = makeGeneralFeat(featureDict, EXAMPLE_DATA["text"], EXAMPLE_DATA["pn"], EXAMPLE_DATA[candidate], ofsList, candidate)
    featureDict = makepnPPFeat(featureDict, loadedTreeList[num][locList[0]], EXAMPLE_DATA["pn"])
    featureDict = makeOrderFeat(featureDict, loadedTreeList[num][locList[0]])
    # featureDict = makeOrderFeat(featureDict, ofsList)

    #page-context
    featureDict = makeEntityFeat(featureDict, EXAMPLE_DATA[candidate], EXAMPLE_DATA["url"])

    return featureDict

def combineFeature(ft_pa, ft_pb):
    featureDict = {}
    featureDict["PN_dist"] = ft_pa["PN_dist"]
    featureDict["A_dist"] = ft_pa["dist"]
    featureDict["B_dist"] = ft_pb["dist"]
    featureDict["pntype"] = ft_pa["pntype"]
    featureDict["A_samePhrase"] = ft_pa["samePhrase"]
    featureDict["B_samePhrase"] = ft_pb["samePhrase"]
    featureDict["pnSubject"] = ft_pa["pnSubject"]
    featureDict["A_subject"] = ft_pa["candidateSubject"]
    featureDict["B_subject"] = ft_pb["candidateSubject"]
    featureDict["A_general"] = ft_pa["general"]
    featureDict["B_general"] = ft_pb["general"]
    featureDict["pnPP"] = ft_pa["pnPP"]
    featureDict["order"] = ft_pa["order"]

    #page-context
    featureDict["A_entity"] = ft_pa["entity"]
    featureDict["B_entity"] = ft_pb["entity"]
    return featureDict

def combineCor(ACOR, BCOR):
    output = ""
    if ACOR == True:
        output += "T"
    else:
        output += "F"
    if BCOR == True:
        output += "T"
    else:
        output += "F"
    return output

def disassembleResult(result):
    if result == "TT":
        return [True, True]
    elif result == "TF":
        return [True, False]
    elif result == "FT":
        return [False, True]
    else:
        return [False, False]

def init():
    traindataDict = dataSet("./gap-development.tsv")
    validatedataDict = dataSet("./gap-validation.tsv")
    testdataDict = dataSet("./gap-test.tsv")
    loadedTreeList = loadTreeList("train")
    loadedValidateList = loadTreeList("validate")
    trainset = []
    for num, EXAMPLE_DATA in enumerate(traindataDict):
        print (num+1)
        EXAMPLE_ACOR = EXAMPLE_DATA["aCor"]
        EXAMPLE_BCOR = EXAMPLE_DATA["bCor"]
        # ft_base = makeBaseFeats(EXAMPLE_DATA)
        ft_pa = {}
        ft_pb = {}
        ft_pa = makeAllFeats("A", EXAMPLE_DATA, loadedTreeList, num)
        ft_pb = makeAllFeats("B", EXAMPLE_DATA, loadedTreeList, num)
        trainelem = (combineFeature(ft_pa, ft_pb), combineCor(EXAMPLE_ACOR, EXAMPLE_BCOR))
        trainset.append(trainelem)

    for num, EXAMPLE_DATA in enumerate(validatedataDict):
        print (num+1)
        EXAMPLE_ACOR = EXAMPLE_DATA["aCor"]
        EXAMPLE_BCOR = EXAMPLE_DATA["bCor"]
        # ft_base = makeBaseFeats(EXAMPLE_DATA)
        ft_pa = {}
        ft_pb = {}
        ft_pa = makeAllFeats("A", EXAMPLE_DATA, loadedValidateList, num)
        ft_pb = makeAllFeats("B", EXAMPLE_DATA, loadedValidateList, num)
        trainelem = (combineFeature(ft_pa, ft_pb), combineCor(EXAMPLE_ACOR, EXAMPLE_BCOR))
        trainset.append(trainelem)

    classifier = NaiveBayesClassifier.train(trainset)

    numTotal = 0
    numTrue = 0
    loadedTestTree = loadTreeList("test")
    testset = []
    outputLine = []

    for num, EXAMPLE_DATA in enumerate(testdataDict):
        print (num+1)
        numTotal += 1
        EXAMPLE_ACOR = EXAMPLE_DATA["aCor"]
        EXAMPLE_BCOR = EXAMPLE_DATA["bCor"]
        # ft_base = makeBaseFeats(EXAMPLE_DATA)
        ft_pa = makeAllFeats("A", EXAMPLE_DATA, loadedTestTree, num)
        ft_pb = makeAllFeats("B", EXAMPLE_DATA, loadedTestTree, num)
        result = classifier.classify(combineFeature(ft_pa, ft_pb))
        outputLine.append([EXAMPLE_DATA["id"]] + disassembleResult(result))

    outputFile = open('CS372_HW5_page_output_20170441.tsv', 'w', encoding='utf-8', newline='')
    outputWriter = csv.writer(outputFile, delimiter='\t')
    for line in outputLine:
        outputWriter.writerow(line)
    outputFile.close()

init()
