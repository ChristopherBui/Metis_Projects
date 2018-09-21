# Project 4 Summary:

## Returning Text Data Relevant To An Asked Question

The field of Natural Language Processing (NLP) is an emerging and fast growing one. In essence, it is the extraction, processing, and manipulation of text data in order to predict an output of text or to see similarities between groups of text. Practical applications of this concept can be applied to online customer service platforms, information request services, or even classification of clients based on their profiles. For this project, its aim is to provide the most relevant Wikipedia articles based on an input question. When researching a topic, people many times do not get the information they are interested in, and as a result, has to spend further unneccessary time researching. This project hopes to establish the foundation for a scalable product that can ultimately reduce this searching time within the realm of NLP. 

## Data

Conveniently, Stanford University has a [public dataset](https://rajpurkar.github.io/SQuAD-explorer/) of 442 Wikipedia articles provided as part of an ongoing competition to see which competitor's model can return a correct answer to a question asked.  The training dataset has on average about 500 questions and answers for each article. The questions are generally asked within context to the article, with answers that are highly embedded within the text of the article. For example, the question, *"What was Beyoncé's debut album?"* has an answer that can be found within the article's lines *"...the release of her debut album, Dangerously in Love (2003)"* . There is a small proportion within the dataset where some questions do not have answers based on the 442 articles, and the answers have been labeled as "no answer". For the sake of building an MVP, such questions will not be considered in building the model.

## Tools

Text data requires that it be processed and transformed numerically, typically in a document term matrix of some sort so that the computer can make sense of the data and run an algorithm. 

For data retrieval, python's JSON library was used. Tokenization, filtering of stop words, and ensuring words abided to the ASCII format, Python's Natural Language Tool Kit (NLTK) was utilized. Pandas and Numpy used to format numerical transformations of the text data and Sklearn was used to implement NLP algorithms.

## Model

The intention of this model is to return the most relevant wikipedia article either to a random question chosen from the training data, or to a custom question that we can ask. After preprocessing of text data, each word in the entire set of aritcles was ran through a Term Frequency–Inverse Document Frequency algorithm **(TF-IDF)**. The result is a document term matrix with numerical TF-IDF values for each word for each document. In order to compare apples to apples, the question of interest is treated as an article and included in the same document matrix. 

In determining the closest related article to the question, **cosine similarity** seems to be a good option.  Each article's TF-IDF values for each word can be seen as components as a vector, and similarly so for the question of interest. By applying cosine similarity between all articles' vectors to the the question's vectors, the article-question pair that returns the closest result to 1 has the highest similarity; From this result, we have identified the closest related article based on text content. 

Overall, this model is a rudimentary form of unsupervised learning, as we do not have labeled article's pertaining to a custom input question. 

## Conclusions

Performance of this model was measured based on its ability to recall the correct Wikipedia article within the top 5 relevant articles returned for each question. If the correct article was within the top 5, then that was counted as "correct". If the correct article was not among the top 5, then it was "incorrect". The percentage of correct over the total number of questions asked was **84.7%.** This result seems adequate for an MVP model. However, the model failed about 15% of the time, a significant proportion especially since the ultimate goal of this product is save people time and optimize accuracy when performing information gathering. The said results can be attributed to the current methods used. Improvements can be made by changing the modeling process.

## Future Work

Currently, this model has enabled the foundations for a product that has the ultimate goal of returning relevant text data based on an input question. Since the feature space of the TF-IDF document term matrix is enormus (1 feature equates to every unique word among all articles), dimensionality reduction algorithms such as LSA will be highly considered. In addition, implementation of a Recurrent Neural Network (RNN) will also be a factor to try in future work, since other NLP projects has shown that RNN work well with text data.

This product differs from a search engine in its ability to scale and return results based on the content of a relevant document. For example, after returning the most relevant article, the product can return the most relevant paragraphs or sentences within that article, saving time for the end user.

