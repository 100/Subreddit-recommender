# Subreddit-recommender

###The data:
I originally found a subset of what is probably my favorite dataset, the collection of all Reddit comments, on Kaggle; it provided May 2015's comments in a SQLite database. However, I wanted some experience reading through files in Python and really dealing with more-difficulty-formatted data, so I googled around, and found another form of this dataset in a plain text file. It was a 30GB text file of JSON blobs.

###Process:
I tried to open the text file, but was unable to due to its large size. Instead, I split the file into 30 smaller text files, which proved to be much more feasible. Then, I iterated through each .txt file in the directory, and obtained each JSON block, delimited by newlines. Some of the blocks were not valid (either due to issues in the original collection process or uneven splitting by me), so I had to ensure that each block was valid and had the required fields. Luckily, very few, if any, of the JSON were invalid.

###The code:
I wanted to do something simple that I could implement from scratch, so I decided to use a very simple machine learning model that used the k-Nearest-Neighbor method. I began by getting a list of all the users in the dataset and all of the subreddits in the union of their subscriptions (only subreddits with 2 or more subscribers were included), operating under the naive assumption that commenting in a subreddit meant that a user was subscribed and that not commenting meant that a user was not subscribed. To further this method, I then used praw to get that user's past comments. This has clear flaws and gravitates towards comment-oriented subreddits, but under the circumstances, was the most logical assumption to make. I then created vectors for each subreddit, with binary values depending on whether or not each user subscribed to it.

Assuming that there were N users, and M subreddits: I had M [Nx1] vectors.

Then, since the calculations are impossibly time-consuming to do online, I had to index the similarities of every subreddit against every other subreddit. I used Jaccard Similarity in this case, because the vectors were incredibly sparse, and using something such as Euclidean distance or Cosine similarity would simply not capture the intended overlaps.

I therefore created an [MxM] matrix S, where S[i][j] is the similarity between the ith subreddit and the jth subreddit. In order to account for comparing a subreddit with itself, I set every entry where i=j to 0.

Finally, in order to find the kth nearest subreddits to the nth subreddit, I simply needed to retrieve the nth column, and extract the kth highest values and their corresponding row labels.

###After-thoughts:
This method is horribly inefficient, and struggles to run at any scale remotely near the 50 million-large dataset that I have on my personal computer. It was definitely more of a proof-of-concept learning experience for me, but even then, it was only able to process ~30 comments per minute. This could definitely be improved by using more functions of pandas, which are hyperoptimized compared to my implementations, or obviously, by just using scikit-learn's kNN module.
