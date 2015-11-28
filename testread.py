import json
import os
import collections
import sys
import praw
import pandas
import requests.exceptions

def findKbest(k, subreddit):
    if type(k) is int and type(subreddit) is str:
        subreddit = subreddit.lower().strip()
        try: #if has been pickled already, use pickle; if not, create pickle
            sims = pandas.read_pickle('similarities.pkl')
        except:
            sims = createSimilarities()
        try:
            desiredSub = sims[subreddit]
        except KeyError:
            sys.exit("Subreddit not found in records.")
        #create ordered list with subreddit names
        orderedSims = {}
        for row, sim in enumerate(desiredSub.tolist()):
            orderedSims[(list(sims.index.values)[row])] = sim  #sub:sim
        kBestSims = sorted(orderedSims.values(), reverse = True)
        kBest = [orderedSims.keys()[orderedSims.values().index(sim)] for sim \
            in kBestSims if (sim in kBestSims and sim > 0)]
        kBest = list(set(kBest))[:k]  #sets inherently disallow duplicates; provides up to k subreddits
        return kBest
    else:
        sys.exit("K must be an integer and subreddit must be a string.")

def createSimilarities():
    r = praw.Reddit("kNN Subreddit Recommendation Engine")
    subreddits = {}
    users = collections.defaultdict(list)
    #to iterate over each file of json
    for text in os.listdir(os.getcwd()):
        if text.endswith(".txt"):
            jsonFile = open(text);
            blobs = jsonFile.read()
            #to iterate over each comment
            rawBlocks = blobs.split("\n")[5500:5505] #gives ~250k comments total
            for blob in rawBlocks:
                try: #valid json?
                    comm = json.loads(blob)
                except ValueError:
                    continue
                if comm['subreddit'] in subreddits:
                    subreddits[comm['subreddit'].lower()] = subreddits[comm['subreddit']] + 1
                else:
                    subreddits[comm['subreddit'].lower()] = 1
                try: #use praw to fetch past 1k comments, then use dataset
                    user = r.get_redditor(comm['author'])
                    for comment in user.get_comments(limit = 50):
                        if comment.subreddit.display_name.lower() not in users[comm['author']]:
                            users[comm['author']].append(comment.subreddit.display_name.lower())
                except (requests.exceptions.RequestException, praw.errors.HTTPException, praw.errors.NotFound):
                    pass
                if comm['subreddit'].lower() not in users[comm['author']]:
                    users[comm['author']].append(comm['subreddit'].lower())
        else:
            continue
    #only want the subreddits with intersection
    for subreddit in subreddits.keys():
        if subreddits[subreddit] == 1:
            del subreddits[subreddit]
    #create subreddit vectors (columns of dataframe)
    for subreddit in subreddits:
        vector = []
        for user, subscribes in users.iteritems():
            if subreddit in subscribes:
                vector.append(1)
            else:
                vector.append(0)
        subreddits[subreddit] = vector
    dataframeSubreddits = pandas.DataFrame.from_dict(subreddits, orient='columns')
    #compute Jaccard similarities across subreddits;
    #too much computation to do on-the-fly, must be done beforehand
    allSims = []
    currSub = 0
    for subreddit, vector in dataframeSubreddits.iteritems():
        allSims.append([])
        vectorList = vector.tolist()
        for nextSubreddit, nextVector in dataframeSubreddits.iteritems():
            if (nextSubreddit == subreddit): #when subreddit compares to itself
                allSims[currSub].append(0.0)
            else:
                nextVectorList = nextVector.tolist()
                numNext = float(nextVectorList.count(1))
                numOrig = float(vectorList.count(1))
                numBoth = float(len([x for ind, x in enumerate(vectorList) if (x == 1 and nextVectorList[ind] == 1)]))
                similarity = numBoth / (numBoth + numOrig + numNext)
                allSims[currSub].append(similarity)
        currSub = currSub + 1
    similaritySubreddits = pandas.DataFrame(data = allSims, index = subreddits.keys(), columns = subreddits.keys())
    similaritySubreddits.to_pickle("similarities.pkl")
    return similaritySubreddits
