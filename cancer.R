#Loading the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data <- read.csv(file = url, header = FALSE,
                 col.names = c("ID","clump_thickness", "uniformity_size", "uniformity_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli","mitoses", "diagnosis"))

#Attaching the required R packages
library(gmodels)
library(ggplot2)
library(reshape)
library(reshape2)
library(corrplot)
library(rpart.plot)
library(randomForest)
library(class)
library(dplyr)
library(C50)
library(caTools)
library(kableExtra)
library(caret)
library(pROC)
library(knitr)

#Data Cleansing
data <- select(data, -1)
data <- data[data$bare_nuclei != "?",] %>% mutate(bare_nuclei = as.integer(as.character((bare_nuclei))))
data <- data %>% mutate(diagnosis = ifelse(diagnosis == 2, 0, 1),
                        diagnosis = as.factor(diagnosis))

#Data Preparation
set.seed(3011) 
train_index <- sample(nrow(data), size = round(0.75 * nrow(data)), replace = FALSE)
train <- data[train_index,]
test <- data[-train_index,]

#DATA EXPLORATION

#Structure
str(data)
dim(data)
names(data)

#Viewing the data
head(data)
summary(data)
sd(data$single_epithelial_cell_size)
summary(data$single_epithelial_cell_size)

#Sum and Mean
tapply(data$clump_thickness, data$diagnosis, sum)
tapply(data$clump_thickness, data$diagnosis, mean)

#CrossTable
CrossTable(data$uniformity_size,data$uniformity_shape)

#Correlation
correlation <- cor(data[,-10])
corrplot(correlation, type = "upper", col = c("#fcbba1", "#b2d2e8"), addCoef.col = "black", tl.col = "black")  

#DATA VISUALISATION

#Segmentation of the diagnosis
qplot(diagnosis, data = data, fill = I("darkblue"))

#Plotting the thickness of the Clump
qplot(clump_thickness, data = data, facets = diagnosis ~ .,bins=30)

#Jitter plot of uniformity size attribute grouped by diagnosis
qplot(diagnosis, uniformity_size, data = data, geom = "jitter")

#Box and Jitter plot of clump thickness attribute grouped by diagnosis
qplot(diagnosis, clump_thickness, data = data, geom = c("boxplot", "jitter"), alpha = I(1/5))

#Density of bland chromatin filling in by mitosis attribute
qplot(bland_chromatin, data = data, fill = mitoses, geom = "density",alpha = I(1/2))

#Modelling Approaches

#DECISION TREE
train_tree <- 0
test_tree <- 0
Dtree <- data.frame(train_tree = numeric(), test_tree = numeric())

set.seed(3011)
tree_parameters <- data.frame(minsplit_para = floor(runif(8, 10, 60)), 
                              maxdepth_para = floor(runif(8, 10, 30)))

for(para_comb in 1:nrow(tree_parameters)){
  decision_tree <- rpart(diagnosis ~ .,  data = train,
                         control = rpart.control(minsplit = tree_parameters[para_comb, "minsplit_para"], 
                                                 maxdepth = tree_parameters[para_comb, "maxdepth_para"])) 
  
  pred_train_tree <- as.data.frame(predict(decision_tree, train, type='prob'))
  train_tree <- roc(train$diagnosis, pred_train_tree$`1`, percent = TRUE, plot = TRUE)
  
  pred_test_tree <- as.data.frame(predict(decision_tree, test, type='prob'))
  test_tree <- roc(test$diagnosis, pred_test_tree$`1`, percent = TRUE, plot = TRUE)
  
  Dtree[para_comb, ] <- c(round(train_tree$auc, 2), round(test_tree$auc, 2))
  train_tree = ifelse(train_tree > train_tree$auc, train_tree, train_tree$auc)
  test_tree = ifelse(test_tree > test_tree$auc, test_tree, test_tree$auc)
}

# Minsplit of 11 and Maxdepth of 10.
best_decision_tree <- rpart(diagnosis ~., data = train,
                            control = rpart.control(minsplit = 11,
                                                    maxdepth = 10))
rpart.plot(x = best_decision_tree, box.palette="RdBu", shadow.col="gray", nn=TRUE, yesno = 2)

#RANDOM FOREST
train_bestrf <- 0
test_bestrf <- 0
rf <- data.frame(train_rf = numeric(), test_rf = numeric()) 

set.seed(160)
rf_parameters <- data.frame(nodesize = round(runif(10,5,20)),
                            sampsize= round(runif(10,1,400)),
                            mtry = round(runif(10,1,10)),
                            ntree = round(runif(10,1,400)))

for(paracomb_rf in 1:nrow(rf_parameters)){
  random_forest <- randomForest(diagnosis ~ ., data = train,
                                nodesize = rf_parameters[paracomb_rf, "nodesize"],
                                sampsize = rf_parameters[paracomb_rf, "sampsize"],
                                mtry = rf_parameters[paracomb_rf, "mtry"],
                                ntree = rf_parameters[paracomb_rf, "ntree"])
  
  pred_train_rf <- as.data.frame(predict(random_forest, train, type='prob'))
  train_rf <- roc(train$diagnosis, pred_train_rf$`1`, percent = TRUE, plot = TRUE)
  
  pred_test_rf <- as.data.frame(predict(random_forest, test, type='prob'))
  test_rf <- roc(test$diagnosis, pred_test_rf$`1`, percent = TRUE, plot = TRUE) 
  
  rf[paracomb_rf, ] <- c(round(AUC_train_rf$au, 2), round(AUC_test_rf$auc, 2))
  train_bestrf = ifelse(train_bestrf > train_rf$auc, train_bestrf, train_rf$auc)
  test_bestrf = ifelse(test_bestrf > test_rf$auc, test_bestrf, test_rf$auc)
}

# nodesize of 9, sampsize of 329, mtry of 7, and ntree of 210.
best_random_forest <- randomForest(diagnosis ~ ., data = train,
                                   nodesize = 9,
                                   sampsize = 329,
                                   mtry = 7,
                                   ntree = 210)
best_random_forest
varImpPlot(best_random_forest)

#KNN Classifier
data$Clump_Thickness=as.numeric(data$clump_thickness)
data$Uniformity_CellSize=as.numeric(data$uniformity_size)
data$Uniformity_CellShape=as.numeric(data$uniformity_shape)
data$Marginal_Adhesion=as.numeric(data$marginal_adhesion)
data$Single_Epithelial_cellSize=as.numeric(data$single_epithelial_cell_size)
data$BareNuclei=as.numeric(data$bare_nuclei)
data$Bland_Chromatin=as.numeric(data$bland_chromatin)
data$Normal_Nucleoli=as.numeric(data$normal_nucleoli)
data$Mitoses=as.numeric(data$mitoses)
data$Diagnosis=as.factor(data$diagnosis)

sub <- sample(nrow(data), floor(nrow(data) * 0.75))

train.class=data[sub,11]
test.class<- data[-sub,11]

predict<-knn(train, test,train.class,k = 10)
table(test.class,predict)

#K means Clustering and C4.5 decision tree classifier

set.seed(100)
malignantdata=subset(data,Diagnosis==0)
benigndata=subset(data,Diagnosis==1)
str(malignantdata)

k=2;
KMC = kmeans(malignantdata[ ,3:11], centers = k, iter.max = 1000)
KMC$cluster

malignantdata$Class=KMC$cluster
benigndata$Class=3

data<-rbind(malignantdata,benigndata)
data
str(data)
data$Class=as.factor(data$Class)
str(data)
treeModel<-C5.0(x=data[,3:11],y=data$Class)
summary(treeModel)

# RESULTS
models_list <- list(Decision_Tree=Dtree, 
                    Random_Forest=rf,
                    KNN=predict,
                    kmc=KMC)                                    
models_results <- resamples(models_list)
summary(models_results)

confusionmatrix_list <- list(
  Decision_Tree=Dtree, 
  Random_Forest=rf,
  KNN=predict,
  kmc=KMC)   
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()

confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)[confusionmatrix_results_max],
                            value=mapply(function(x,y) {confusionmatrix_list_results[x,y]}, 
                                         names(confusionmatrix_results_max), 
                                         confusionmatrix_results_max))
rownames(output_report) <- NULL
output_report



