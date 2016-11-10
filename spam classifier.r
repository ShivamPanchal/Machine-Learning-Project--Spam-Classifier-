#############################
#                           #
#     SMS-SPAM CLASSIFIER   #
#                           #
#############################
#   NAIVE BAYES ALGORITHM   #  
############################

# Machine Learning: Naive Bayes Classifiation
# A simple R implementation of classification using Naive Bayes
# Shivam Panchal


# STEP 1

# loading the dataset into the Console
sms_raw <- read.csv("sms_spam.csv" , stringsAsFactors = FALSE)
str(sms_raw)

# Two Variables- Type and Text
# the type variable is a character vector , since it is a categorical variable.
# So, conver it into a factor
sms_raw$type <- factor(sms_raw$type) #exploration of data
str(sms_raw$type)

# Converting it into a table
table(sms_raw$type)


# STEP 2

# Data prepartion for analysis
#The tm text mining package can be installed via the install.packages("tm") command and loaded with library(tm).

library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:5])


# Implementing the transformation mapping, tm_map command

corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

corpus_clean <- tm_map(corpus_clean, PlainTextDocument)

# Now, the data is processed to our liking 
# Creating a document Term matrix

sms_dtm <- DocumentTermMatrix(corpus_clean) 


# Now, we will generate wordcloud for both indices "ham" and "spam" individually

spam_indices <- which(sms_raw$type == "spam")
ham_indices <- which(sms_raw$type == "ham")

library(wordcloud)

wordcloud(corpus_clean[spam_indices], min.freq = 10)
wordcloud(corpus_clean[ham_indices], min.freq = 10)


# STEP 3

# Data Preparation- creating training and test data datasents

# We will begin by splitting the raw data frame
sms_raw_train <- sms_raw[1:4169,]
sms_raw_test <- sms_raw[4170:5559,]
# then, document term marix
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]
# and, finally the corpus
sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]


# To check that subsets are representative of the complete set
# let's compare the proportion of spam in train and test data set 
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

# Vizualizing text data- wordcloud for both data sets

spam <- subset(sms_raw_train , type == "spam")
ham <- subset(sms_raw_train , type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# Data Preparation - creating indicator features for frequent words


#############################################################################################
#                                                                                           #
#  The final step in the data preparation process is to transform the sparse matrix         #
# into a data structure that can be used to train a naive Bayes classifier. Currently,      #
#  the sparse matrix includes over 7,000 features a feature for every word that appears     #
#  in at least one SMS message. It's unlikely that all of these are useful for              #
#  classification. To reduce the number of features, we will eliminate any words            #
#  that appear in less than five SMS messages, or less than about 0.1 percent of records    #
#  in the training data. Finding frequent words requires use of the findFreqTerms()         #
#  function in the tm package. This function takes a document term matrix and returns a     #
#  character vector containing the words appearing at least a specified number of times.    #
#                                                                                           #
#############################################################################################

#Get the description of the function 

# findFreqTerms(x, lowfreq = 0, highfreq = Inf)

five_times_words <- findFreqTerms(sms_dtm_train, 5)

sms_train <- DocumentTermMatrix(sms_corpus_train, control = list(dictionary = five_times_words))
sms_test <- DocumentTermMatrix(sms_corpus_test, control = list(dictionary = five_times_words))

# Now, the training and test data contains less words which are more relevant/frequent

#######################################################################################
#                                                                                     #
# The naive Bayes classifier is typically trained on data with categorical features.  #
# This poses a problem since the cells in the sparse matrix indicate a count of the   #
# times a word appears in a message. We should change this to a factor variable that  #
# simply indicates yes or no depending on whether the word appears at all.            #
# The following code defines a convert_counts() function to convert counts to factors #
#                                                                                     #
#######################################################################################

convert_counts <- function(x) {
  y <- ifelse(x>0, 1, 0)
  y <- factor(x, levels = c(0,1), labels = c("Yes","No"))
  return(y)
}

##############################################################################################
#                                                                                            #
#   The first line defines the function. The statement ifelse(x > 0, 1, 0) will transform    #
#   the values in x so that if the value is greater than 0, then it will be replaced with 1, # 
#   otherwise it will remain at 0. The factor command simply transforms the 1 and 0 values   #
#   to a factor with labels No and Yes. Finally, the newly-transformed vector x  is returned.#
#                                                                                          ``#
##############################################################################################

##############################################################################################
#                                                                                            #
# The apply() function is part of a family of functions including lapply() and sapply() that #
# perform operations on each element of an R data structure. These functions are one of key  # 
# idioms of the R language. Experienced R coders use these functions rather than using loops # 
# such as for or while as you would in other programming languages because they result in    # 
# more readable (and sometimes more efficient) code.                                         #
#                                                                                            #
# The apply() function allows a function to be used on each of the rows/columns in a matrix. # 
# It uses a MARGIN parameter to specify either rows or columns. Here, we'll use MARGIN = 2   #
# since we're interested in the columns (MARGIN = 1 is used for rows).                       #
#                                                                                          ``#
##############################################################################################


sms_train <- apply(sms_train , MARGIN = 2, convert_counts)
sms_test <- apply(sms_test , MARGIN = 2, convert_counts)

# The result will be two matrixes, each with factor type columns indicating Yes or No for whether 
# each column's word appears in the messages comprising the rows. 

# Now, we have transformed the raw data into a format that can be represented
# by a statistical model, its time to apply Naive Bayes Algorithm

# install.packages("e1071") and library(e1071) before continuing.
# Also available in NaiveBayes() function in KlaR package


#  STEP 4

# Training a model on the data
library(e1071)
sms_classifier <- naiveBayes(sms_train, factor(sms_raw_train$type))


# STEP 5

# Evaluating model performance on sms_test, class labels spam or ham is stored in matrix name sms_raw_test

sms_test_pred <- predict(sms_classifier, newdata=sms_test)

table(sms_test_pred, sms_raw_test$type)
# sms_test_pred  ham  spam
#     ham       1202    31
#    spam         5    152


# To compare the predicted values to the actual values, we'll use the CrossTable() function in the gmodels package

library(gmodels)
#  we'll add some additional parameters to eliminate unnecessary cell proportions, 
#  and use the dnn parameter (dimension names) to relabel the rows and columns, as shown in the following code:

CrossTable(sms_test_pred, sms_raw_test$type, 
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c("predicted", "actual"))


#Cell Contents
#  |-------------------------|
#  |                       N |
#  |           N / Row Total |
#  |           N / Col Total |
#  |-------------------------|
  

#  Total Observations in Table:  1390 

#
#                | actual 
#  predicted     |       ham |      spam | Row Total | 
#  --------------|-----------|-----------|-----------|
#  ham           |      1202 |        31 |      1233 | 
#                |     0.975 |     0.025 |     0.887 | 
#                |     0.996 |     0.169 |           | 
#  ------------- |-----------|-----------|-----------|
#  spam          |         5 |       152 |       157 | 
#                |     0.032 |     0.968 |     0.113 | 
#                |     0.004 |     0.831 |           | 
#  --------------|-----------|-----------|-----------|
#  Column Total  |      1207 |       183 |      1390 | 
#                |     0.868 |     0.132 |           | 
#  ------------- |-----------|-----------|-----------|
#  