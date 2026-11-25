def main():
    with open("WSJ_02-21.pos", "r") as f:
        words, pos = parseFile(f, False) # Words and parts of speech arrays
    with open("WSJ_24.pos", "r") as f:
        words2, pos2 = parseFile(f, False) # Adding development corpus
    words = words + [""] + words2
    pos = pos + [""] + pos2
    posLiklihood, transition, setOfWords = countFrequencies(words, pos) # Getting dictionaries of dictionaries with frequencies
    posProbTable, transitionProbTable = getProbabilities(posLiklihood, transition) # Getting probability tables as dictionaries

    with open("WSJ_23.words", "r") as f:
        testWords = parseFile(f, True) # Words to tag

    sentences = [] # Splitting test words into sentences, each sentence is a list of words
    current = [] # Words in the sentence being built
    for word in testWords:
        if word == "": # End of sentence
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(word)
    if current: # Last sentence when there's no blank line
        sentences.append(current)

    with open("submission.pos", "w") as output:
        for sentence in sentences: # Tagging each sentence
            tags = transduceSentence(posProbTable, transitionProbTable, setOfWords, sentence)
            for i in range(len(sentence)):
                output.write(sentence[i] + "\t" + tags[i] + "\n")
            output.write("\n") # Blank line between sentences

# This function determines if the file path in question is a file with only words or a file with words and parts of speech depending on the boolean onlyWords.
# This function parses a tagged file into a list of words or a list with two sub-lists of words and parts of speech.
# The function marks sentence boundaries with an empty string in the list or each sublist.
def parseFile(f, onlyWords):
    words = []
    if (onlyWords): # Case where file only has words
        for line in f:
            word = line.strip()
            if not word : # Blank line sentence boundaries
                words.append("")
                continue
            words.append(word)
        return words
    else: # Case where file has both words and tags
        pos = []
        for line in f:
            wordAndPOS = line.strip().split()
            if not wordAndPOS : # Blank line sentence boundaries
                words.append("")
                pos.append("")
                continue
            words.append(wordAndPOS[0])
            pos.append(wordAndPOS[1])
        return words, pos

# This is a helper function to get three tables (posLiklihood, transition, and setOfWords).
# The function takes the raw lists of words and POS tags as input.
def countFrequencies(words, pos):
    posLiklihood = {} # Table of frequencies of words that occur with a POS
    transition = {} # Table of frequencies of following states 
    setOfWords = {} # Words (and their frequencies) in corpus that appear once or more than once

    for i in range(len(pos)): # Filling in posLiklihood
        tag = pos[i]
        word = words[i]
        if tag != "":
            if tag not in posLiklihood:
                posLiklihood[tag] = {}
            if word not in posLiklihood[tag]:
                posLiklihood[tag][word] = 1
            else:
                posLiklihood[tag][word] += 1

    prev = "Begin_Sent" # Starting state, beginning a sentence
    for i in range(len(pos)): # Filling in transition
        curr = pos[i]
        if curr == "": # Blank line is end of sentence
            if prev != "Begin_Sent":
                if prev not in transition:
                    transition[prev] = {}
                if "End_Sent" not in transition[prev]:
                    transition[prev]["End_Sent"] = 1
                else:
                    transition[prev]["End_Sent"] += 1
            prev = "Begin_Sent"
        else:
            if prev not in transition:
                transition[prev] = {}
            if curr not in transition[prev]:
                transition[prev][curr] = 1
            else:
                transition[prev][curr] += 1
            prev = curr
    if prev != "Begin_Sent": # Ending state, end of sentence
        if prev not in transition:
            transition[prev] = {}
        if "End_Sent" not in transition[prev]:
            transition[prev]["End_Sent"] = 1
        else:
            transition[prev]["End_Sent"] += 1

    for word in words: # Filling in setOfWords
        if word != "":
            if word not in setOfWords:
                setOfWords[word] = 1
            else:
                setOfWords[word] += 1

    return posLiklihood, transition, setOfWords

# This helper function converts the frequency tables into probability tables.
# The function takes posLiklihood and transition and returns two new tables with probabilities.
def getProbabilities(posLiklihood, transition):
    posProbTable = {} # Table of probabilities of each word for each POS tag
    transitionProbTable = {} # Table of probabilities of each state given previous state

    for tag in posLiklihood: # Getting probabilities for posLiklihood
        tagTotal = 0
        for word in posLiklihood[tag]: # Adding up all frequencies for this tag
            tagTotal += posLiklihood[tag][word]
        posProbTable[tag] = {}
        for word in posLiklihood[tag]: # Getting probability of word by tag
            posProbTable[tag][word] = posLiklihood[tag][word] / tagTotal

    for prev in transition: # Getting probabilities for transition
        prevTotal = 0
        for state in transition[prev]: # Adding up all frequencies for the previous state
            prevTotal += transition[prev][state]
        transitionProbTable[prev] = {}
        for state in transition[prev]: # Dividing each frequency by the previous state's total
            transitionProbTable[prev][state] = transition[prev][state] / prevTotal

    return posProbTable, transitionProbTable

# This is a helper function that will run the Viterbi algorithm for one sentence.
# It uses the previously obtained posProbTable, transitionProbTable, and setOfWords, and also takes the sentence in as input.
def transduceSentence(posProbTable, transitionProbTable, setOfWords, sentence):
    posTags = list(posProbTable.keys()) # Getting all POS tags
    states = ["Begin_Sent"] + posTags + ["End_Sent"]
    viterbiChart = [] # 2-D array, first index is the state, and second index is the column
    backtrack = [] # 2-D array that stores, for each cell of a column, the best row in previous column
    
    for row in states: # Adding a row for every state, and adding the length of the sentence + 2 columns to each row
        viterbiChart.append([0] * (len(sentence) + 2))
        backtrack.append([None] * (len(sentence) + 2)) # Appending blank rows to be later filled in
    viterbiChart[0][0] = 1 # Beginning state is probability 1
    
    firstWord = sentence[0] # Doing position 1
    for row in range(1, len(states) - 1): # Skipping Begin_Sent and End_Sent
        state = states[row]
        prob = getProb(posProbTable, setOfWords, firstWord, state)
        transitionProb = transitionProbTable.get("Begin_Sent", {}).get(state, 1/1000) # Tiny probability if pair doesn't exist
        viterbiChart[row][1] = 1 * transitionProb * prob
        backtrack[row][1] = 0 # Came from Begin_Sent, which is row 0
        
    for column in range(2, len(sentence) + 1): # Doing positions 2 to end
        word = sentence[column - 1]
        for row in range(1, len(states) - 1):
            state = states[row]
            prob = getProb(posProbTable, setOfWords, word, state)
            bestScore = 0
            bestPrevRow = None
            for prevRow in range(len(states)): # Trying every cell in the previous column
                prevScore = viterbiChart[prevRow][column - 1]
                prevState = states[prevRow]
                transitionProb = transitionProbTable.get(prevState, {}).get(state, 1 / 10000000) # Tiny probability if pair doesn't exist
                score = prevScore * transitionProb * prob
                if score > bestScore: # Getting row with best score
                    bestScore = score
                    bestPrevRow = prevRow
            viterbiChart[row][column] = bestScore
            backtrack[row][column] = bestPrevRow # Row number of the best previous state
        columnMax = max(viterbiChart[r][column] for r in range(len(states))) # Rescaling the column so its best score is 1,
        if columnMax > 0: # otherwise long sentences (100+ words) underflow to 0; every cell is divided equally, so the best path is unchanged
            for r in range(len(states)):
                viterbiChart[r][column] /= columnMax

    endRow = len(states) - 1 # Doing End_Sent row
    endColumn = len(sentence) + 1
    bestScore = 0
    bestPrevRow = None
    for prevRow in range(len(states)):
        prevScore = viterbiChart[prevRow][endColumn - 1]
        prevState = states[prevRow]
        transitionProb = transitionProbTable.get(prevState, {}).get("End_Sent", 1 / 1000) # Transition to End_Sent
        score = prevScore * transitionProb * 1 # End state likelihood is 1
        if score > bestScore:
            bestScore = score
            bestPrevRow = prevRow
    viterbiChart[endRow][endColumn] = bestScore
    backtrack[endRow][endColumn] = bestPrevRow

    return getTagsFromBacktrack(backtrack, states, sentence)
    
# This helper function returns the probability of a word given a POS tag.
# The function takes posProbTable, a word, a tag, and setOfWords and returns the probability for a word given a tag, so long as the word is in the set of words.
# If the word is out of vocabulary, it returns a probability of 1 / 1000.
def getProb(posProbTable, setOfWords, word, tag):
    if word in setOfWords:
        return posProbTable[tag].get(word, 0) # 0 if word never seen with this tag
    return 1 / 1000

# This helper function walks backwards through the backtrack chart, starting at End_Sent.
# It runs through the backtrack array from the end, appends the optimal rows, and returns the particular tags as mapped to the states table.
# It returns the optimal tags for a sentence in the order of a sentence.
def getTagsFromBacktrack(backtrack, states, sentence):
    tags = []
    row = len(states) - 1 # Start at the End_Sent row
    for column in range(len(sentence) + 1, 1, -1): # Ending at position 2 (first word), going backwards
        row = backtrack[row][column] # Getting rows of best states
        tags.append(states[row])
    tags.reverse() # Reversing list to get sentence order
    return tags

if __name__ == "__main__": # Only run the homework pipeline when executed directly, not on import
    main()