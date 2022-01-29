#### Import libraries and dataset ####
library(NLP)
library(ggplot2)
library(readr)
library(dplyr)
library(caTools)
library(rpart)
library(rpart.plot)
library(tm)
library(SnowballC)
library(RColorBrewer)
library(stringr)
library(tidytext)
library(wordcloud)

COVID <- read.csv("Corona_NLP_train.csv")

#### Data exploration ####
head(COVID)
COVID$Sentiment <- as.factor(COVID$Sentiment)

COVID[COVID==""] <- NA #Setting all the blank cells as NA
any(is.na(COVID)) 
is.na(COVID)
#Location is the only variable with null variables, but since this variable is not really relevant in the analysis, we will not worry about this.

#Count of sentiment tweets
table(COVID["Sentiment"])
barplot(table(COVID["Sentiment"]), col=brewer.pal(5, "Set2"), xlab = "Sentiment", ylab = "Nº of tweets", main = "Count of different sentiment COVID-19 tweets")


#### Data preparation ####
#The only variables of interest for this project are OriginalTweet and Sentiment, so we only keep these.
COVID<- COVID[, c("OriginalTweet", "Sentiment")]
head(COVID)

#Combining "Extremely Negative" and "Negative" tweets, as well as "Extremely Positive" and "Positive" tweets for ease. 
levels(COVID$Sentiment)[levels(COVID$Sentiment)=="Extremely Negative"] <-"Negative"
levels(COVID$Sentiment)[levels(COVID$Sentiment)=="Extremely Positive"] <-"Positive"
head(COVID$Sentiment)

#Removing urls, html tags, digits, hashtags and mentions
COVID$CleanTweet<- gsub("(RT|via)((?:\\b\\W*@\\w+)+)*","",COVID$OriginalTweet)
COVID$CleanTweet<- gsub("@\\w+","",COVID$CleanTweet)
COVID$CleanTweet<- gsub("http.+","",COVID$CleanTweet)
COVID$CleanTweet<- gsub("https.+","",COVID$CleanTweet)
COVID$CleanTweet<- gsub("[[:digit:]]*","",COVID$CleanTweet)
COVID$CleanTweet <- str_replace(COVID$CleanTweet,"RT @[a-z,A-Z]*: ","")

#Separating different sentiment tweets into 3 dataframes
df_neutral <- dplyr::filter(COVID, Sentiment=='Neutral')
df_positive <- dplyr::filter(COVID, Sentiment=='Positive')
df_negative <- dplyr::filter(COVID, Sentiment=='Negative')

#### Analysis ####
#Tokenization and removal of stop words from the tidytext package for each of the Sentiments.

#NEGATIVE TWEETS
df_clean_NEG <- df_negative %>%
  dplyr::select(CleanTweet) %>%
  unnest_tokens(word, CleanTweet)

data("stop_words")
cleaned_tweet_words_NEG <- df_clean_NEG %>%
  anti_join(stop_words)

#Plotting the top 10 words
cleaned_tweet_words_NEG %>%
  count(word, sort = TRUE) %>%
  top_n(10) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col(fill = "#FF6666") +
  xlab(NULL) +
  coord_flip() +
  labs(y = "Count",
       x = "Unique words",
       title = "Count of unique words found in NEGATIVE tweets")


#NEUTRAL TWEETS
df_clean_NEU <- df_neutral %>%
  dplyr::select(CleanTweet) %>%
  unnest_tokens(word, CleanTweet)

data("stop_words")
cleaned_tweet_words_NEU <- df_clean_NEU %>%
  anti_join(stop_words)

#Plotting the top 10 words
cleaned_tweet_words_NEU %>%
  count(word, sort = TRUE) %>%
  top_n(10) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col(fill = "#FFA500") +
  xlab(NULL) +
  coord_flip() +
  labs(y = "Count",
       x = "Unique words",
       title = "Count of unique words found in NEUTRAL tweets")


#POSITIVE TWEETS
df_clean_POS <- df_positive %>%
  dplyr::select(CleanTweet) %>%
  unnest_tokens(word, CleanTweet)

data("stop_words")
cleaned_tweet_words_POS <- df_clean_POS %>%
  anti_join(stop_words)

#Plotting the top 10 words
cleaned_tweet_words_POS %>%
  count(word, sort = TRUE) %>%
  top_n(10) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col(fill = "#00FF00") +
  xlab(NULL) +
  coord_flip() +
  labs(y = "Count",
       x = "Unique words",
       title = "Count of unique words found in POSITIVE tweets")


#Wordclouds by sentiment

#NEGATIVE 
wordcloud(df_negative$CleanTweet, max.words = 100, random.order = FALSE, colors=brewer.pal(8,"Set1"))

#NEUTRAL
wordcloud(df_neutral$CleanTweet, max.words = 100, random.order = FALSE, colors=brewer.pal(8,"Dark2"))

#POSITIVE
wordcloud(df_positive$CleanTweet, max.words = 100, random.order = FALSE, colors=brewer.pal(8,"Set2"))


#Converting to data frame
corpus <- VCorpus(VectorSource(COVID$CleanTweet))
tdm <- DocumentTermMatrix(corpus, control = 
                            list(tolower = TRUE,
                                 removeNumbers = TRUE,
                                 stopwords = TRUE,
                                 removePunctuation = TRUE,
                                 stemming = TRUE))

dim(tdm)
inspect(tdm)

findFreqTerms(tdm, lowfreq = 200) 
Words <- removeSparseTerms(tdm, 0.99)


Words <- as.data.frame(as.matrix(Words)) 
colnames(Words) <- make.names(colnames(Words)) 
str(Words)
Words$Class <- COVID$Sentiment

#### Modelling and evaluation ####
#Decision Tree algorithm
set.seed(123)
split <- sample.split(Words$Class, SplitRatio = 0.70)
train <- subset(Words, split == T)
test <- subset(Words, split == F)

table(test$Class)

#Decision Tree algorithm
TrainingTree <- rpart(Class ~ ., data = train, method = "class", minbucket = 35)
prp(TrainingTree)
print(TrainingTree)
PredictingTree <- predict(TrainingTree, test, type = "class")
table(test$Class, PredictingTree)


rpart.accuracy.table <- as.data.frame(table(test$Class, PredictingTree))
print(paste("Accuracy of the Decision Tree Model: ", 100*round(((rpart.accuracy.table$Freq[1]+rpart.accuracy.table$Freq[4])/nrow(test)), 4), "%"))
