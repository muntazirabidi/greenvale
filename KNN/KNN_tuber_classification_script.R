#install.packages("sampling")
library(stats)
library(class)
library(sampling)
library(dplyr)
#library(plyr)

# Functions
#function called by yield_sample_metrics for mean, variance and standard deviation
multi.fun <- function(x) {
  c(mean = mean(x), var = var(x), sd = sd(x))
}

folder = "/home/sma74/local/scratch/public/sma74-dropbox/Dropbox/CatalystAI/Greenvale/KNN/"

#read in raw data of tuber weights
df.tw <- read.csv(paste(folder,"MarisPiper_TuberData.csv", sep=""), header=TRUE) #variety = Maris Piper
#df.tw <- read.csv(paste(folder,"Jelly_TuberData.csv",sep=""), header=TRUE) #variety = Jelly
#df.tw <- read.csv(paste(folder,"Soraya_TuberData.csv",sep=""), header=TRUE) #variety = Soraya
#df.tw <- read.csv(paste(folder,"Venezia_TuberData.csv",sep=""), header=TRUE) #variety = Venezia
#df.tw <- read.csv(paste(folder,"Marfona_TuberData.csv",sep=""), header=TRUE) #variety = Marfona
#df.tw <- read.csv(paste(folder,"Orchestra_TuberData.csv",sep=""), header=TRUE) #variety = Orchestra
#df.tw <- read.csv(paste(folder,"LadyBalfour_TuberData.csv",sep=""), header=TRUE) #variety = Lady Balfour
 
# Field Names - by variety
# Maris Piper: 'R21', '582', 'M12', 'Duffy'
# Soraya: 'Somerlayton', 'Year 1 trial field', 'Yr 1 variety trial', 'Field', 'Fairley 1'
# Jelly: 'Franklin', 'Irby Hall Nort', 'Middle Allotment', 'RHP', 'Somerlayton'
# Venezia: 'Chanters Hole', 'Hanger Blackdyke',  'Hanger BlackDyke', 'HS2', 'Lovers Pightle', 'TurnPike',  'wortley', 'NSB Trial 1'
# Marfona:  'APT Farming', 'APT Farming 2', 'APT Farming 3', 'Early Baker', 'EB Field - Dig 1', 'EB Field - Dig 2', 'Park Farm', 'Salmons',
#           'EB Field - Plot 1 Guard 1',  'EB Field - Plot 1 Guard 2', 'EB Field - Plot 12 Guard 1', 'EB Field - Plot 12 Guard 2', 'EB Field - Plot 24 Guard 1', 'EB Field - Plot 24 Guard 2', 
# Lady Balfour: "Commercial Crop 1", "Commercial Crop 2", "Field Sample 1", "Field Sample 2", "Trial Guard Row 1", "Trial Guard Row 2", "Trial Guard Row 3"
# Orchestra: 'Oxford Field', 'Plumbs', 'Water Lane'


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#KNN classification model
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
field = 'R21'
  #split data in training and testing datasets - for per crop analysis
  cv_tr <- df.tw[df.tw$Type != field,]
  cv_tst <- df.tw[df.tw$Type == field,]
  
  #allocate training and testing datasets to matrices for parsing through KNN 
  
  #training dataset
  prc_train_labels <- matrix(cv_tr[,2],ncol=1)
  prc_train <- matrix(cv_tr[,6],ncol=1)
  #testing dataset
  prc_test_labels <- matrix(cv_tst[,2],ncol=1)
  prc_test <- matrix(cv_tst[,6],ncol=1)
  
  #**********************************************************************
  #KNN clasification
  
  #classify training dataset
  prc_train_pred <- knn(prc_train, prc_train, cl=prc_train_labels, k=11)
  #classify testing dataset
  prc_test_pred <- knn(prc_train, prc_test, cl=prc_train_labels, k=11)
  
  inv <- cbind(cv_tst, prc_test_pred)
  
  #**********************************************************************
  #derive error rates
  # training error rate
  tr_err <- mean(prc_train_labels != prc_train_pred)
  # testing error rate
  tst_err <- mean(prc_test_labels != prc_test_pred)
  
  #output training and test error rates
  errTable <- cbind(tr_err, tst_err)

  #**********************************************************************    
  #aggregate output  
  
  #hand graded sample data
  outSample <- ddply(cv_tst, .(MidSize),
                     summarise,
                     tubercount=length(TuberWeight),
                     tuberweight=sum(TuberWeight)/1000)  
  
  #predicted model of sample
  modOut <- as.data.frame(cbind(as.numeric(as.character(prc_test_pred)), cv_tst[,6]),ncol=2) 
  colnames(modOut) <- c("MidSize", "TuberWeight")
  
  outPred <- ddply(modOut, .(MidSize),
                   summarise,
                   tubercount=length(TuberWeight),
                   tuberweight=sum(TuberWeight)/1000)
  
  out <- merge(outSample,outPred, by="MidSize", all=TRUE)
  out[is.na(out)] <- 0
  
  #============================================================================
  #predicted sample stats - hand graded sample
  
  #calculate total weight of sample and total tuber population
  total_tubers <- sum(out[,2])
  total_weight <- sum(out[,3])
  
  frequency <- round(out[,3]/total_weight*100,2)
  
  sizeDist <- rep(out[,1], frequency) ## expands the data by frequency of each size grade
  
  #tuber profile summary (mu, sigma)
  metrics <- multi.fun(sizeDist)
  
  k <- metrics[1]/(total_weight/total_tubers)^(1/3)
  tubCoV <- metrics[3]/metrics[1]*100
  
  sample_metrics_s <- cbind(metrics[1], 
                            metrics[3],
                            tubCoV, 
                            k)
  
  #============================================================================
  #predicted sample stats - model predicted sample
  
  total_tubers <- sum(out[,4])
  total_weight <- sum(out[,5])
  
  frequency <- round(out[,5]/total_weight*100,2)
  
  sizeDist <- rep(out[,1], frequency) ## expands the data by frequency of each size grade
  
  #tuber profile summary (mu, sigma)
  metrics <- multi.fun(sizeDist)
  
  #K parameter (mu/(yield/tuber pop) ^ 1/3)
  
  
  k <- metrics[1]/(total_weight/total_tubers)^(1/3)
  tubCoV <- metrics[3]/metrics[1]*100
  
  sample_metrics_c <-cbind(metrics[1], 
                           metrics[3],
                           tubCoV, 
                           k)
  
  #============================================================================
  #summary output
  
  #1. yield sample categorisation (tuber count and weight per 5 mm size fraction)
  df.sampleSummary <- out
  #2. sample metrics (Mu, CoV, K for hand graded and model predicted classification)
  df.tuberMetrics <- cbind(sample_metrics_s, sample_metrics_c)
    
  
  #**********************************************************************************
  #==================================================================================
  #crop optimisation model
  # function will calculate the optimum mean tuber size for the crop metrics supplied
  
  #set input values from df.tuberMetrics
  yield_sample_CoV <- 17.5
  yield_sample_k <- 120.5
  yield_sample_tuber_population <- (10000/(2.5*0.91)*sum(df.sampleSummary$tubercount.x))/1000 #set tuber population per ha ('000's per ha)
  #Don't change these values
  
  commMin <- 45
  commMax <- 85
  muMax <- 70
  optFlag <- 2
  subFracMin <- 45
  subFracMax <- 85
  subFracPct <- 50
  overPct <- 10
  
  
  #function will return a target mean tuber size (mu) and yield for the given sample parameters and selected commercial specifications
  yield_sample_optimised <- yield_sample_optiMu(yield_sample_CoV, 
                                                yield_sample_k, 
                                                yield_sample_tuber_population, 
                                                commMin, commMax, muMax, optFlag,
                                                subFracMin, subFracMax, subFracPct,
                                                overPct)
  
  
  
  
  
  
  yield_sample_optiMu <- function(inCov, inK, intubPop, inMin, inMax, inMuMax, inSizeFlag, inSizeFracMin, inSizeFracMax, inSizeFracPct, inOver, print=TRUE){
    
    #storage vector
    muSize <- c()
    allYld <- c()
    commFracYld <- c()
    commFrac <- c()
    subSizeFrac <- c()
    pctOver <- c()
    pctUnder <- c()
    
    #set range of mean tuber size to search through for optimum value
    mulower <- 10
    muupper <- inMuMax
    
    #sequence of mu, incremented by 0.1, and length of sequence
    muProc <- seq(mulower,muupper, by=1)
    totLen <- length(seq(mulower,muupper, by=1))
    
    for (i in 1:totLen){
      
      muSize[i] <- round(muProc[i],2)
      # calculate % yield in required commercial size fraction
      commFrac[i] <- round((pnorm(inMax, mean=muProc[i], sd=inCov/100 * muProc[i]) - pnorm(inMin, mean=muProc[i], sd=inCov/100 * muProc[i])) * 100,2)
      #calculate gross yield (t/ha)
      allYld[i] <- round((intubPop*(muProc[i]/inK)^3),2)
      #calculate commercial (net) yield (t/ha)
      commFracYld[i] <- round((intubPop*(muProc[i]/inK)^3) * (commFrac[i]/100),2)
      # calculate % bakers
      if(inSizeFracMin > 0 ){
        subSizeFrac[i] <- round((pnorm(inSizeFracMax, mean=muProc[i], sd=inCov/100 * muProc[i]) - pnorm(inSizeFracMin, mean=muProc[i], sd=inCov/100 * muProc[i])) * 100,2)
      }else{
        subSizeFrac[i] <- -9999
      }
      # % over size
      pctOver[i] <- round((1 - pnorm(inMax, mean=muProc[i], sd=inCov/100 * muProc[i])) * 100,2)
      # % undersize
      pctUnder[i] <- round(1 - ( 1 - (pnorm(inMin, mean=muProc[i], sd=inCov/100 * muProc[i]) * 100)),2)
      
    }   
    
    #combine metrics
    cropSummary <- cbind(muSize, allYld, commFracYld, commFrac, subSizeFrac, pctOver, pctUnder)
    
    #===================================================================================================
    #Optimisation 1 - 50% 65-85mm
    
    if (inSizeFlag == 1){
      
      idx <- which(cropSummary[,6] < inOver)
      #optimisation 1a - aim for 50% 65-85 mm and <10% over size
      #find if model has returned >50% in the 65-85mm range
      if (max(cropSummary[idx,5]) >= inSizeFracPct){
        
        idx <- which(cropSummary[,5] >= inSizeFracPct)
        #find value closest to 50% bakers
        idxOpt <- idx[which.min(abs(cropSummary[idx,5]-inSizeFracPct))]
        OptFlag <- 1
        
        #optimisation 1b - if 1a can't be resolved return 50% 65-85 mm and remove waste restriction
      }else{
        
        idx <- which(cropSummary[,1] <= inMuMax & cropSummary[,5] >= inSizeFracPct)
        if (length(idx) > 0){
          
          idxOpt <- idx[which.min(abs(cropSummary[idx,5]-inSizeFracPct))]
          OptFlag <- 2
          
          #optimisation 1c - if 1b can't be resolved (i.e. <50% 65-85 mm) return maximum 65-85 mm fraction
        }else{
          
          idx <- which(cropSummary[,1] <= inMuMax)
          #find max % commercial fraction
          idxOpt <- which.max(cropSummary[idx,5])
          OptFlag <- 3
          
        }
        
      }
      
      #===================================================================================================  
      #Optimisation 2 - maximise sub fraction yield (e.g. 58-75 mm  of 45-75 mm)  
    }else if (inSizeFlag == 2){
      
      idx <- which(cropSummary[,1] <= inMuMax & round(cropSummary[,6]) <= inOver)
      idxOpt <- which.max(cropSummary[idx,5])
      OptFlag <- 4
      
      #===================================================================================================  
      #Optimisation 3 - minimise variation in the yield between bottom - mid and mid - upper size fractions    
      
    } else if (inSizeFlag == 3){
      
      idx <- which(cropSummary[,1] >= inMin & cropSummary[,1] <= inMuMax &  cropSummary[,4] > 0 &  cropSummary[,5] > 0)
      idxOpt <- idx[which.min(abs(cropSummary[idx,5]-(cropSummary[idx,4] - cropSummary[idx,5])))]
      OptFlag <- 5
      
    }
    
    optiYield <- c(cropSummary[idxOpt,], OptFlag)
    return(optiYield)
    
  }
  
  
  
  
