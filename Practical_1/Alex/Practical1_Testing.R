
# Load Functions ==================================

library(RODBC)
library(dplyr)
library(knitr)
library(ggplot2)
library(scales)
library(stringr)
library(rmarkdown)
library(stringr)
library(h2o)

source('G:\\git_adagley\\algu_repo\\R\\au_sourcer.R')
source("G:\\git_adagley\\cogsy\\Cogsy_Core_fns.R")
source("G:\\git_adagley\\cogsy\\Cogsy_Batch_fns.R")


# Initiliaze =====================================

myconn <- odbcConnect("AlexSQL")

localH2O <- auh2$MakeOrConnectToH2O(RAM_GB = 300
                                    , jarPath = "C:\\Program Files\\R\\R-3.2.2\\library\\h2o\\java\\h2o.jar"
                                    , port = 55519
                                    , nThreads = 14)

# Data Setup ==========================================

Bag1 <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Bag1.csv")
Bag1 <- slice(Bag1, 1000000:nrow(Bag1)+1)
Bag2 <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Bag2.csv")
Bag2 <- slice(Bag2, 1000000:nrow(Bag2)+1)
Bag3 <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Bag3.csv")
Bag3 <- slice(Bag3, 1000000:nrow(Bag3)+1)
Bag4 <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Bag4.csv")
Bag4 <- slice(Bag4, 1000000:nrow(Bag4)+1)
Bag5 <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Bag5.csv")
Bag5 <- slice(Bag5, 1000000:nrow(Bag5)+1)
test <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\test.csv")
#train <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\train.csv", nrows=100000)
similarity <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\SimilarityV1.csv")
similarity <- slice(similarity, 1000000:nrow(similarity)+1)

Morgan <- read.csv("I:\\algu\\user\\adagley\\Kaggle\\Morgan2048.csv")
Morgan <- slice(Morgan, 1000000:nrow(Morgan)+1)

Bag1 <- select(Bag1, -X)
Bag2 <- select(Bag2, -X)
Bag3 <- select(Bag3, -X)
Bag4 <- select(Bag4, -X)
Bag5 <- select(Bag5, -X)
Morgan <- select(Morgan, -X)
similarity <- select(similarity, -X)
#test <- select(test, -Id)
train <- select(train, -Id)

train$

Big_df <- inner_join(Bag1, Bag2, by = "smiles") %>%
  inner_join(Bag3, by = "smiles") %>%
  inner_join(Bag4, by = "smiles") %>%
  inner_join(Bag5, by = "smiles") %>%
  inner_join(similarity, by = "smiles") %>%
  inner_join(Morgan, by = "smiles") %>%
  inner_join(test, by = "smiles")
  
Big_df2 <- inner_join(Big_df, Morgan, by = "smiles")
Big_df <- inner_join(Big_df, similarity, by = "smiles")

Big_df <- inner_join(Big_df, train, by  = "smiles")


small_df <- sample_frac(Big_df2, .1, replace = F)
#small_df <- select(small_df, -contains("feat"))

Big_df$isTrain <- auml$DefineIsTrain_Random(Big_df, .8)

X_Names <- as.vector(colnames(select(Big_df, -smiles) %>% select(-gap) %>% select(-isTrain)))
train_df <- filter(Big_df, isTrain == T)
test_df <- filter(Big_df, isTrain == F)
# Train GBM Model ======================================

model_gbm <- h2o.gbm(x = X_Names
                     , y = "gap"
                     , training_frame = as.h2o(train_df)
                     , distribution = 'gaussian'
                     , max_depth = 10
                     , ntrees = 100
                     , min_rows = 100
                     , learn_rate = .04)


# Test Model =================================

Y_Pred_FULL <- h2o.predict(model_gbm, newdata = as.h2o(Big_df2))  


# Measure Error ========================

Big_df$Y_Pred <- as.matrix(Y_Pred_FULL)[,'predict']

Big_df$L2 <- (test_df$Y_Pred - test_df$gap)^2

RMSE <- sqrt(mean(test_df$L2))

ggplot(data = test_df, aes(test_df$gap)) + 
  geom_histogram(col="red"
                 , fill="green"
                 , alpha = .2)  +
  labs(title="Cogsy Elasta vs Current Elasticity Price Distribution") +
  labs(x="Weighted Difference", y="Count") 