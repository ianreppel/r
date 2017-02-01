#######################################################################################################################
# SETUP: SEE http://theoryno3.blogspot.de/2010/12/installing-topicmodels-r-package.html
#######################################################################################################################

# Initialize JVM for RWeka (on Linux)
require(rJava) 
options(java.parameters = "-Xmx4g")

# Define and load required packages
libs = c("ggplot2", "extrafont", "gridExtra",
         "tm", "SnowballC",
         "randomForest", "caret",
         "psych", "reshape",
         "topicmodels", "RWeka")
lapply(libs, require, character.only=TRUE)

rm(libs)

#######################################################################################################################
# DATA LOAD
#######################################################################################################################

# Load the CSV files
setwd("/home/megatron/Downloads")
NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
NewsTest  = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

# Combine the training and test data
NewsTest$Popular = NA
News             = rbind(NewsTrain, NewsTest)

#######################################################################################################################
# FUNCTIONS
#######################################################################################################################

# Remove leading and training white space
trim = function(str) {
  return(gsub("^\\s+|\\s+$", "", str))
}

# Remove diacritics from letters
replaceHTML = function(str) {
  return(gsub("&([[:alpha:]])(acute|cedil|circ|grave|uml);", "\\1", str))
}

# Remove special characters, such as dashes, quotes, and ellipses
ignoreHTML = function(str) {
  return(gsub("&((m|n)dash|hellip|(l|r)squo;)", "", str))
}

# Extract the text between HTML tags
extractText = function(str) {
  return(gsub("<.*?>", "", str))
}

# Clean up: extract, remove, replace, and trim
cleanupText = function(str) {
  return(trim(replaceHTML(ignoreHTML(extractText(str)))))
}

# Replace multiple string patterns
mgsub = function(x, pattern, replacement, ...) {
  if (length(pattern) != length(replacement)) {
    stop("pattern and replacement do not have the same length.")
  }
  result = x
  for (i in 1:length(pattern)) {
    result = gsub(pattern[i], replacement[i], result, ...)
  }
  return(result)
}

# Analyse frequency and popularity of string patterns
analyseTerms = function(terms, ignoreCase=TRUE)
{
  idxHeadTrain = which(grepl(terms, NewsTrain$Headline, ignore.case=ignoreCase))
  idxTextTrain = which(grepl(terms, NewsTrain$Text,     ignore.case=ignoreCase))
  idxHeadTest  = which(grepl(terms, NewsTest$Headline,  ignore.case=ignoreCase))
  idxTextTest  = which(grepl(terms, NewsTest$Text,      ignore.case=ignoreCase))

  avgPopHeadTrain = mean(as.numeric(as.character(NewsTrain$Popular[idxHeadTrain])))
  numArtHeadTrain = length(idxHeadTrain)
  avgPopTextTrain = mean(as.numeric(as.character(NewsTrain$Popular[idxTextTrain])))
  numArtTextTrain = length(idxTextTrain)
  numArtHeadTest  = length(idxHeadTest)
  numArtTextTest  = length(idxTextTest)

  return(c(avgPopHeadTrain, numArtHeadTrain,
           avgPopTextTrain, numArtTextTrain,
           numArtHeadTest, numArtTextTest))
}

# Calculate the accuracy and AUC of the model on the training set
calcAUC = function(model, truth)
{
  suppressPackageStartupMessages(require(ROCR))

  model.pred = predict(model, type="prob")
  model.accu = tr(table(truth, model.pred[, 2]>0.5))/nrow(model.pred)
  model.auc  = as.numeric(performance(prediction(model.pred[, 2], truth), "auc")@y.values)

  return(c(model.accu, model.auc))
}

# N-Gram tokenizer for mono-, bi-, and tri-grams
NGTokenizer = function(x) NGramTokenizer(x, Weka_control(min = 1, max = 3))

# Prepare a CSV file with predictions for submission
generateSubmission = function(predictions) {
  fileName = paste0("Submission_", deparse(substitute(predictions)), ".csv")
  submission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = predictions)
  write.csv(submission, fileName, row.names=FALSE)
}

#######################################################################################################################
# DATA TRANSFORMATIONS AND CLEANSING
#######################################################################################################################

News$Headline = cleanupText(News$Headline)
News$Summary  = ifelse(nchar(cleanupText(News$Snippet)) > nchar(cleanupText(News$Abstract)),
                       cleanupText(News$Snippet),
                       cleanupText(News$Abstract))

originalText    = c("new york times", "new york city", "new york", "silicon valley", "times insider",
                    "fashion week", "white house", "international herald tribune archive", "president obama", "hong kong",
                    "big data", "golden globe")
replacementText = c("NYT", "NYC", "NewYork", "SiliconValley", "TimesInsider",
                    "FashionWeek", "WhiteHouse", "IHT", "Obama", "HongKong",
                    "BigData", "GoldenGlobe")

News$Headline = mgsub(News$Headline, originalText, replacementText, ignore.case=TRUE)
News$Summary  = mgsub(News$Summary,  originalText, replacementText, ignore.case=TRUE)
News$Text     = paste(News$Headline, News$Summary)

News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Arts", "Culture", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Business Day", "Business", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Health", "Science", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Multimedia", "", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="N.Y. / Region", "Metro", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Open", "Technology", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Opinion", "OpEd", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Technology", "Business", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="Travel", "Travel", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="U.S.", "National", News$NewsDesk)
News$NewsDesk = ifelse(News$NewsDesk=="" & News$SectionName=="World", "Foreign", News$NewsDesk)

idx = which(News$SectionName=="Crosswords/Games")
News$NewsDesk[idx]       = "Styles"
News$SectionName[idx]    = "Puzzles"
News$SubsectionName[idx] = ""

idx = which(News$NewsDesk=="Styles" & News$SectionName=="U.S.")
News$NewsDesk[idx]       = "Styles"
News$SectionName[idx]    = "Style"
News$SubsectionName[idx] = ""

News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="Culture", "Arts", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="Foreign", "World", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="National", "U.S.", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="OpEd", "Opinion", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="Science", "Science", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="Sports", "Sports", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="Styles", "Style", News$SectionName)
News$SectionName = ifelse(News$SectionName=="" & News$NewsDesk=="TStyle", "Magazine", News$SectionName)

idx = which(News$NewsDesk == "" & News$SectionName == "" & News$SubsectionName == "" &
              grepl("^(first draft|lunchtime laughs|politics helpline|today in politics|verbatim)",
                    News$Headline, ignore.case=TRUE))
News$NewsDesk[idx]       = "National"
News$SectionName[idx]    = "U.S."
News$SubsectionName[idx] = "Politics"

idx = which(News$SectionName=="" &
              grepl(paste0("white house|democrat|republican|tea party|",
                           "obama|biden|boehner|kerry|capitol|senat|",
                           "sen\\.|congress|president|washington|politic|",
                           "rubio|palin|clinton|bush|limbaugh|rand paul|",
                           "christie|mccain|election|poll|cruz|constitution|",
                           "amendment|federal|partisan|yellen|govern|",
                           "gov\\.|legislat|supreme court|campaign|",
                           "primary|primaries|justice|jury"),
                    News$Text, ignore.case=TRUE))
News$NewsDesk[idx]       = "National"
News$SectionName[idx]    = "U.S."
News$SubsectionName[idx] = "Politics"

idx = which(News$SectionName=="" &
              grepl(paste0("PAC|GOP|G\\.O\\.P\\.|NRA|N\\.R\\.A\\."),
                    News$Text, ignore.case=FALSE))
News$NewsDesk[idx]       = "National"
News$SectionName[idx]    = "U.S."
News$SubsectionName[idx] = "Politics"

News$NewsDesk[which(News$NewsDesk=="")]             = "Missing"
News$SectionName[which(News$SectionName=="")]       = "Missing"
News$SubsectionName[which(News$SubsectionName=="")] = "Missing"

rm(idx)
rm(originalText)
rm(replacementText)

#######################################################################################################################
# FEATURE ENGINEERING
#######################################################################################################################

News$PubDate = strptime(News$PubDate, "%Y-%m-%d %H:%M:%S")
News$PubDay  = as.Date(News$PubDate)
News$Weekday = News$PubDate$wday
News$Hour    = News$PubDate$hour
News$LogWC   = log(1 + News$WordCount)

News$HeadlineCharCount = nchar(News$Headline)
News$SummaryCharCount  = nchar(News$Summary)
News$HeadlineWordCount = sapply(gregexpr("\\W+", gsub("[[:punct:]]", "", News$Headline)), length) + 1
News$SummaryWordCount  = sapply(gregexpr("\\W+", gsub("[[:punct:]]", "", News$Summary)),  length) + 1

News$Pop                         = News$Popular
News$Pop[which(is.na(News$Pop))] = "N/A"

News$SEO         = as.factor(ifelse(News$HeadlineCharCount<=48, 1, 0))
News$Question    = as.factor(ifelse(grepl("\\?", News$Headline), 1, 0))
News$Exclamation = as.factor(ifelse(grepl("!", News$Headline), 1, 0))
News$HowTo       = as.factor(ifelse(grepl("^how to", News$Headline, ignore.case=TRUE), 1, 0))
News$Negative    = as.factor(ifelse(grepl("\\<(never|do not|dont|don't|stop|quit|worst)\\>",
                                          News$Headline, ignore.case=TRUE), 1, 0))
News$SpecialWord = as.factor(ifelse(grepl("\\<(strange|incredible|epic|simple|ultimate|great|sex)\\>",
                                          News$Headline, ignore.case=TRUE), 1, 0))


News$NoComment = as.factor(ifelse(grepl(
  paste0("6 q's about the news|daily|fashion week|first draft|in performance|",
         "international arts events happening in the week ahead|",
         "inside the times|lunchtime laughs|pictures of the day|playlist|",
         "podcast|q\\. and a\\.|reading the times|test yourself|",
         "throwback thursday|today in|the upshot|tune in to the times|",
         "tune into the times|under cover|verbatim|walkabout|weekend reading|",
         "weekly news quiz|weekly wrap|what we're (reading|watching)|",
         "what's going on in this picture|word of the day|the daily gift"),
  News$Headline, ignore.case=TRUE), 1, 0))

News$Recurrent = as.factor(ifelse(grepl(
  paste0("ask well|facts & figures|think like a doctor|readers respond|",
         "no comment necessary|quandary|your turn"),
  News$Headline, ignore.case=TRUE), 1, 0))

News$Controversial = as.factor(ifelse(grepl(
  paste0("\\<(gun control|abortion|birth control|",
         "consent|rape|african-american|latino|racis(m|t))\\>"),
  News$Headline, ignore.case=TRUE), 1, 0))

News$Obama       = as.factor(ifelse(grepl("obama|president", News$Headline, ignore.case=TRUE), 1, 0))
News$Republican  = as.factor(ifelse(grepl("republican", News$Headline, ignore.case=TRUE), 1, 0))
News$Congress    = as.factor(ifelse(grepl("\\<(senate|congress)\\>", News$Headline, ignore.case=TRUE), 1, 0))
News$Election    = as.factor(ifelse(grepl("\\<(election|campaign|poll(s|))\\>", News$Headline, ignore.case=TRUE), 1, 0))

holidays = c(as.POSIXlt("2014-09-01 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-10-13 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-10-31 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-11-11 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-11-27 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-12-24 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-12-25 00:00", format="%Y-%m-%d %H:%M"),
             as.POSIXlt("2014-12-31 00:00", format="%Y-%m-%d %H:%M"))

News$Holiday       = as.factor(ifelse(News$PubDate$yday %in% holidays$yday, 1, 0))
News$BeforeHoliday = as.factor(ifelse(News$PubDate$yday %in% (holidays$yday-1) &
                                        News$PubDate$hour>=17, 1, 0))

News$Current = as.factor(ifelse(grepl(
  "ebola|ferguson|michael brown|cuba|embargo|castro|havana",
  News$Text, ignore.case=TRUE), 1, 0))

News$Recap = as.factor(ifelse(grepl("recap",
                                    News$Headline, ignore.case=TRUE), 1, 0))

News$UN = as.factor(ifelse(grepl("u\\.n\\.|united nations|ban ki-moon|climate",
                                 News$Text, ignore.case=TRUE), 1, 0))

News$Health = as.factor(ifelse(grepl(
  paste0("mental health|depress(a|e|i)|anxiety|schizo|",
         "personality|psych(i|o)|therap(i|y)|brain|autis(m|t)|",
         "carb|diet|cardio|obes|cancer|homeless"),
  News$Headline), 1, 0))

News$Family = as.factor(ifelse(grepl(
  "education|school|kids|child|college|teenager|mother|father|parent|famil(y|ies)",
  News$Headline, ignore.case=TRUE), 1, 0))

News$Tech = as.factor(ifelse(grepl(
  paste0("twitter|facebook|google|apple|microsoft|amazon|",
         "uber|phone|ipad|tablet|kindle|smartwatch|",
         "apple watch|match\\.com|okcupid|social (network|media)|",
         "tweet|mobile| app "),
  News$Headline, ignore.case=TRUE), 1, 0))

News$Security = as.factor(ifelse(grepl("cybersecurity|breach|hack|password",
                                       News$Headline, ignore.case=TRUE), 1, 0))

News$Biz = as.factor(ifelse(grepl(
  paste0("merger|acqui(s|r)|takeover|bid|i\\.p\\.o\\.|billion|",
         "bank|invest|wall st|financ|fund|share(s|holder)|market|",
         "stock|cash|money|capital|settlement|econo"),
  News$Headline, ignore.case=TRUE), 1, 0))

News$War = as.factor(ifelse(grepl(
  paste0("israel|palestin|netanyahu|gaza|hamas|iran|",
         "tehran|assad|syria|leban(o|e)|afghan|iraq|",
         "pakistan|kabul|falluja|baghdad|islamabad|",
         "sharif|isis|islamic state"),
  News$Text, ignore.case=TRUE), 1, 0))

News$Cuba = as.factor(ifelse(grepl("cuba|embargo|castro|havana",
                                   News$Text, ignore.case=TRUE), 1, 0))

News$Holidays = as.factor(ifelse(grepl("thanksgiving|hanukkah|christmas|santa",
                                       News$Text, ignore.case=TRUE), 1, 0))

News$Boring = as.factor(ifelse(grepl(
  paste0("friday night music|variety|[[:digit:]]{4}|photo|today|",
         "from the week in style|oscar|academy|golden globe|diary|",
         "hollywood|red carpet|stars|movie|film|celeb|sneak peek|",
         "by the book|video|music|album|spotify|itunes|taylor swift|",
         "veteran|palin|kerry|mccain|rubio|rand paul|yellen|partisan|",
         "capitol|bush|clinton|senator|congressman|governor|chin(a|e)|",
         "taiwan|tibet|beijing|hongkong|russia|putin"),
  News$Text, ignore.case=TRUE), 1, 0))

rm(holidays)

#######################################################################################################################
# FEATURE CONVERSION: FACTORIZATION
#######################################################################################################################

News$Pop            = as.factor(News$Pop)
News$Popular        = as.factor(News$Popular)
News$NewsDesk       = as.factor(News$NewsDesk)
News$SectionName    = as.factor(News$SectionName)
News$SubsectionName = as.factor(News$SubsectionName)
News$PubDay         = as.factor(News$PubDay)
News$Weekday        = as.factor(News$Weekday)
News$Hour           = as.factor(News$Hour)
News$DayOfWeek      = as.factor(weekdays(News$PubDate))
News$DayOfWeek      = factor(News$DayOfWeek,
                             levels=c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

#######################################################################################################################
# FEATURE ENGINEERING: TEMPORAL COUNTS - MUST BE DONE AFTER FACTORIZATION FOR MERGE TO WORK PROPERLY
#######################################################################################################################

dailyArticles = as.data.frame(table(News$PubDay))
names(dailyArticles) = c("PubDay", "NumDailyArticles")
News = merge(News, dailyArticles, all.x=TRUE)

dailySectionArticles = as.data.frame(table(News$PubDay, News$SectionName))
names(dailySectionArticles) = c("PubDay", "SectionName", "NumDailySectionArticles")
News = merge(News, dailySectionArticles, all.x=TRUE)

hourlyArticles = as.data.frame(table(News$PubDay, News$Hour))
names(hourlyArticles) = c("PubDay", "Hour", "NumHourlyArticles")
News = merge(News, hourlyArticles, all.x=TRUE)

hourMatrix = as.matrix(table(News$Hour, News$Popular))
hourMatrix = cbind(hourMatrix, hourMatrix[, 2]/(hourMatrix[, 1] + hourMatrix[, 2]))
colnames(hourMatrix) = c("Unpopular", "Popular", "PopularDensity")

rm(dailyArticles)
rm(dailySectionArticles)
rm(hourlyArticles)

#######################################################################################################################
# TEXT MINING: CORPORA
#######################################################################################################################

dropWords = c(stopwords("SMART"))

CorpusHeadline = Corpus(VectorSource(News$Headline))
CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusHeadline = tm_map(CorpusHeadline, removeWords, dropWords)
CorpusHeadline = tm_map(CorpusHeadline, stemDocument, language="english")
dtmHeadline    = DocumentTermMatrix(CorpusHeadline)
sparseHeadline = removeSparseTerms(dtmHeadline, 0.995)
sparseHeadline = as.data.frame(as.matrix(sparseHeadline))

CorpusSummary = Corpus(VectorSource(News$Summary))
CorpusSummary = tm_map(CorpusSummary, tolower)
CorpusSummary = tm_map(CorpusSummary, PlainTextDocument)
CorpusSummary = tm_map(CorpusSummary, removePunctuation)
CorpusSummary = tm_map(CorpusSummary, removeWords, dropWords)
CorpusSummary = tm_map(CorpusSummary, stemDocument, language="english")
dtmSummary    = DocumentTermMatrix(CorpusSummary)
sparseSummary = removeSparseTerms(dtmSummary, 0.99)
sparseSummary = as.data.frame(as.matrix(sparseSummary))

freqTerms         = findFreqTerms(dtmHeadline, lowfreq=100)
termFreq          = colSums(as.matrix(dtmHeadline))
termFreq          = subset(termFreq, termFreq>=100)
freqTermsHeadline = data.frame(term=names(termFreq), freq=termFreq)

colnames(sparseHeadline) = make.names(paste0("H",colnames(sparseHeadline)))
colnames(sparseSummary)  = make.names(paste0("S",colnames(sparseSummary)))

NYTWords                         = cbind(sparseHeadline, sparseSummary)
NYTWords$Popular                 = News$Popular
NYTWords$NewsDesk                = News$NewsDesk
NYTWords$SectionName             = News$SectionName
NYTWords$SubsectionName          = News$SubsectionName
NYTWords$LogWC                   = News$LogWC
NYTWords$Weekday                 = News$Weekday
NYTWords$Hour                    = News$Hour
NYTWords$SEO                     = News$SEO
NYTWords$Question                = News$Question
NYTWords$Holiday                 = News$Holiday
NYTWords$BeforeHoliday           = News$BeforeHoliday
NYTWords$NumDailyArticles        = News$NumDailyArticles
NYTWords$NumDailySectionArticles = News$NumDailySectionArticles
NYTWords$NumHourlyArticles       = News$NumHourlyArticles

CorpusText = Corpus(VectorSource(News$Text))
CorpusText = tm_map(CorpusText, tolower)
CorpusText = tm_map(CorpusText, PlainTextDocument)
CorpusText = tm_map(CorpusText, removePunctuation)
CorpusText = tm_map(CorpusText, removeWords, dropWords)
CorpusText = tm_map(CorpusText, stemDocument, language="english")
tdmText    = TermDocumentMatrix(CorpusText,
                                control=list(weighting=weightTfIdf,
                                             tokenize=NGTokenizer))
names(tdmText) = make.names(names(tdmText))
sparseText     = removeSparseTerms(tdmText, 0.99)
freqTerms      = findFreqTerms(sparseText, lowfreq=20)
termFreq       = rowSums(as.matrix(sparseText))
termFreq       = subset(termFreq, termFreq>=20)
freqTermsText  = data.frame(term = names(termFreq), freq = termFreq)

rm(dropWords)
rm(freqTerms)
rm(termFreq)

#######################################################################################################################
# TOPIC CLUSTERING AND MODELLING
#######################################################################################################################

# K-Means clustering based on tf-idf n-grams
NYTNGrams.matrix     = as.matrix(sparseText)
NYTNGrams.distMatrix = dist(scale(NYTNGrams.matrix))
NYTNGrams.clusters   = hclust(NYTNGrams.distMatrix, method="ward.D")

k          = 25
mTranspose = t(sparseText)
KMC        = kmeans(mTranspose, k)

for (i in 1:k) {
  cat(paste("cluster ", i, ": ", sep=""))
  s = sort(KMC$centers[i, ], decreasing=T)
  cat(names(s)[1:15], sep=", ", "\n")
}

News$TopicCluster = as.factor(KMC$cluster)

# LDA based on tf monograms
dtmText      = DocumentTermMatrix(CorpusText)
ldaText      = LDA(dtmText, k=25) # 25 topics
topicTerms   = terms(ldaText, 5)  # first 5 terms for each topic
topicTerms   = apply(topicTerms, MARGIN=2, paste, collapse=", ")
topicsText   = topics(ldaText, 1)
topicsSeries = data.frame(date=as.Date(News$PubDate), topic=topicsText, terms=topicTerms[topicsText])

News$Topic   = as.factor(topicsText)

rm(i)
rm(k)
rm(s)
rm(mTranspose)
rm(NYTNGrams.matrix)
rm(NYTNGrams.distMatrix)
rm(ldaText)
rm(topicTerms)
rm(topicsText)

#######################################################################################################################
# TRAIN/TEST DATA SPLIT
#######################################################################################################################

News = subset(News, select=-c(PubDate, PubDay, Headline, Snippet, Abstract,
                              Summary, Text, WordCount))

NewsTrain = head(News, nrow(NewsTrain))
NewsTest  = tail(News, nrow(NewsTest))

NYTWordsTrain = head(NYTWords, nrow(NewsTrain))
NYTWordsTest  = tail(NYTWords, nrow(NewsTest))

rm(NYTWords)

#######################################################################################################################
# PREDICTIVE MODELS
#######################################################################################################################

# Create a 50-50 partition in the training data for cross-validation and tuning of random forests
# partTrain   = createDataPartition(y=NewsTrain$Popular, p=0.5, list=FALSE)
# rfTrainTune = NewsTrain[partTrain, ]

# Basic RF model with all custom features
#  rfCustomFull.tuned = train(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster -Topic
#                             -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
#                            data=rfTrainTune,
#                            method="rf",
#                            trControl=trainControl(method="cv",number=5),
#                            allowParallel=TRUE)
# print(rfCustomFull.tuned)

rfCustomFullModel = randomForest(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster -Topic
                                 -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
                                 data=NewsTrain, nodesize=5, ntree=1000, importance=TRUE)
calcAUC(rfCustomFullModel, NewsTrain$Popular)

impCustomFull      = melt(importance(rfCustomFullModel, type=1))
impCustomFull$sign = ifelse(impCustomFull$value>=0, "positive", "negative")

# Basic RF model with selected (relevant) custom features
# rfCustomSel.tuned = train(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster -Topic
#                           -Health -Holiday -HowTo -Cuba -Election
#                           -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
#                           data=rfTrainTune,
#                           method="rf",
#                           trControl=trainControl(method="cv",number=5),
#                           allowParallel=TRUE)
# print(rfCustomSel.tuned)

rfCustomSelModel = randomForest(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster -Topic
                                -Health -Holiday -HowTo -Cuba -Election
                                -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
                                data=NewsTrain, nodesize=5, ntree=1000, importance=TRUE)
calcAUC(rfCustomSelModel, NewsTrain$Popular)

impCustomSel      = melt(importance(rfCustomSelModel, type=1))
impCustomSel$sign = ifelse(impCustomSel$value>=0, "positive", "negative")

# RF model with selected (relevant) custom features and k-means topic clusters
# rfCustomSelKMC.tuned = train(Popular ~ . -UniqueID -Pop -DayOfWeek -Topic
#                              -Health -Holiday -HowTo -Cuba -Election
#                              -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
#                              data=rfTrainTune,
#                              method="rf",
#                              trControl=trainControl(method="cv",number=5),
#                              allowParallel=TRUE)
# print(rfCustomSelKMC.tuned)

rfCustomSelKMCModel = randomForest(Popular ~ . -UniqueID -Pop -DayOfWeek -Topic
                                   -Health -Holiday -HowTo -Cuba -Election
                                   -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
                                   data=NewsTrain, nodesize=5, ntree=1000, importance=TRUE)
calcAUC(rfCustomSelKMCModel, NewsTrain$Popular)

impCustomSelKMC      = melt(importance(rfCustomSelKMCModel, type=1))
impCustomSelKMC$sign = ifelse(impCustomSelKMC$value>=0, "positive", "negative")

# RF model with selected (relevant) custom features and LDA topics
# rfCustomSelLDA.tuned = train(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster
#                              -Health -Holiday -HowTo -Cuba -Election
#                              -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
#                              data=rfTrainTune,
#                              method="rf",
#                              trControl=trainControl(method="cv",number=5),
#                              allowParallel=TRUE)

rfCustomSelLDAModel = randomForest(Popular ~ . -UniqueID -Pop -DayOfWeek -TopicCluster
                                   -Health -Holiday -HowTo -Cuba -Election
                                   -HeadlineCharCount -SummaryCharCount -NumHourlyArticles,
                                   data=NewsTrain, nodesize=5, ntree=1000, importance=TRUE)
calcAUC(rfCustomSelLDAModel, NewsTrain$Popular)

impCustomSelLDA      = melt(importance(rfCustomSelLDAModel, type=1))
impCustomSelLDA$sign = ifelse(impCustomSelLDA$value>=0, "positive", "negative")

# Logistic model based on basic RF model with selected custom features
logModel = glm(Popular ~ NewsDesk + SectionName + SubsectionName + Weekday + LogWC +
                 HeadlineWordCount + SummaryWordCount + SEO + Question + Exclamation + Negative + SpecialWord +
                 NoComment + Recurrent + Controversial + Obama + Republican + Congress + BeforeHoliday + Current +
                 Recap + UN + Family + Tech + Security + Biz + War + Holidays + Boring +
                 NumDailyArticles + NumDailySectionArticles,
               data=NewsTrain, family=binomial)

# Bag-of-Words (BoW) RF model with non-text features
rfBoWModel = randomForest(Popular ~ ., data=NYTWordsTrain, nodesize=5, ntree=1000, importance=TRUE)
calcAUC(rfBoWModel, NYTWordsTrain$Popular)

# Predictions for the test set
rfCustomFullPred   = predict(rfCustomFullModel, newdata=NewsTest, type="prob")[, 2]
rfCustomSelPred    = predict(rfCustomSelModel, newdata=NewsTest, type="prob")[, 2]
rfCustomSelKMCPred = predict(rfCustomSelKMCModel, newdata=NewsTest, type="prob")[, 2]
rfCustomSelLDAPred = predict(rfCustomSelLDAModel, newdata=NewsTest, type="prob")[, 2]
rfBoWPred          = predict(rfBoWModel, newdata=NYTWordsTest, type="prob")[, 2]
logPred            = predict(logModel, newdata=NewsTest, type="response")

# Predictions for blended models
ensSelBoWLogPred = (rfCustomSelPred + rfBoWPred + logPred)/3
ensKMCBoWLogPred = (rfCustomSelKMCPred + rfBoWPred + logPred)/3
ensLDABoWLogPred = (rfCustomSelLDAPred + rfBoWPred + logPred)/3

ens2Sel3BoW1LogPred = (2*rfCustomSelPred + 3*rfBoWPred + logPred)/6
ens2KMC3BoW1LogPred = (2*rfCustomSelKMCPred + 3*rfBoWPred + logPred)/6
ens2LDA3BoW1LogPred = (2*rfCustomSelLDAPred + 3*rfBoWPred + logPred)/6

ens3KMC2LDA4BoW1LogPred = (3*rfCustomSelLDAPred + 2*rfCustomSelLDAPred + 4*rfBoWPred + logPred)/10

generateSubmission(rfCustomFullPred)
generateSubmission(rfCustomSelPred)
generateSubmission(rfCustomSelKMCPred)
generateSubmission(rfCustomSelLDAPred)
generateSubmission(rfBoWPred)
generateSubmission(logPred)
generateSubmission(ensSelBoWLogPred)
generateSubmission(ensKMCBoWLogPred)
generateSubmission(ensLDABoWLogPred)
generateSubmission(ens2Sel3BoW1LogPred)
generateSubmission(ens2KMC3BoW1LogPred)
generateSubmission(ens2LDA3BoW1LogPred)
generateSubmission(ens3KMC2LDA4BoW1LogPred)

#######################################################################################################################
# GRAPHICS
#######################################################################################################################

p = ggplot(NewsTrain, aes(x=LogWC, fill=Popular)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of LogWC") +
  xlab("log(1 + WordCount)") +
  theme(axis.title.y = element_blank()) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-dist-logwc.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(NewsTrain, aes(x=LogWC, fill=Popular)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of LogWC") +
  xlab("Log(1 + WordCount)") +
  theme(axis.title.y = element_blank()) +
  facet_wrap( ~ DayOfWeek, ncol=2) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-dist-daily-logwc.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p1 = ggplot(NewsTrain, aes(x=HeadlineCharCount, fill=Popular)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of HeadlineCharCount") +
  xlab("# Characters in Headline") +
  theme(axis.title.y = element_blank()) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
p2 = ggplot(NewsTrain, aes(x=HeadlineWordCount, fill=Popular)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of HeadlineWordCount") +
  xlab("# Words in Headline") +
  theme(axis.title.y = element_blank()) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
p = arrangeGrob(p1, p2, ncol=1, nrow=2)
ggsave(filename="kaggle-dist-headline-cc-wc.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(News, aes(x=NumDailyArticles, fill=Pop)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of NumDailyArticles") +
  xlab("# Daily Articles Published") +
  scale_fill_discrete(name="Popular") +
  theme(axis.title.y = element_blank()) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-dist-daily-articles.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(News, aes(x=NumDailySectionArticles, fill=Pop)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of NumDailySectionArticles") +
  xlab("# Daily Articles Published") +
  scale_fill_discrete(name="Popular") +
  theme(axis.title.y = element_blank()) +
  facet_wrap( ~ SectionName, ncol=3) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-dist-daily-section-articles.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(News, aes(x=NumHourlyArticles, fill=Pop)) +
  geom_density(aes(y=..scaled..), alpha=0.4) +
  ggtitle("Distribution of NumHourlyArticles") +
  xlab("# Hourly Articles Published") +
  scale_fill_discrete(name="Popular") +
  theme(axis.title.y = element_blank()) +
  facet_wrap( ~ Hour, ncol=3) +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-dist-hourly-articles.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(impCustomFull, aes(x=reorder(X1, value, max), y=value, group=X2, fill=sign)) +
  geom_bar(stat="identity") +
  coord_flip() +
  ggtitle("Relative Importance of Predictors") +
  theme(axis.title.y = element_blank()) +
  ylab("Mean Decrease Accuracy") +
  theme(legend.position = "none") +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-imp-rf.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(freqTermsHeadline, aes(x=reorder(term, freq, max), y=freq)) +
  geom_bar(stat="identity") +
  xlab("Terms") +
  ylab("Frequency") +
  coord_flip() +
  ggtitle("Most Common Terms in the Headline") +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-freq-terms-headline.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)

p = ggplot(freqTermsText, aes(x=reorder(term, freq, max), y=freq)) +
  geom_bar(stat="identity") +
  ggtitle("Most Common N-Grams in News$Text") +
  xlab("Terms") +
  ylab("Frequency") +
  coord_flip() +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-n-grams.png", plot=p, type="cairo-png", dpi=300, width=8, height=14)

p = ggplot(topicsSeries, aes(x=date)) +
  geom_density(aes(y=..count.., fill=terms), position="stack") +
  ggtitle("Evolution of the Distribution of Topics") +
  xlab("Publication Date") +
  ylab("Frequency") +
  theme(plot.title = element_text(size=16, face="bold")) +
  theme(text=element_text(family="AvantGarde", size=14))
ggsave(filename="kaggle-topics-evolution.png", plot=p, type="cairo-png", dpi=300, width=8, height=8)