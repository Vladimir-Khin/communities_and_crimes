library(glmnet)
library(randomForest)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
df = read.csv('communities.data.txt', header = FALSE)

colnames(df) <- c('state','county','community','communityname','fold','population','householdsize',
                  'racepctblack','racepctwhite','racepctasian','racepcthisp','agepct12t21','agepct12t29',
                  'agepct16t24','agepct65up','numburban','pcturban','medincome','pctwwage','pctwfarmself',
                  'pctwinvinc','pctwsocsec','pctwpubasst','pctwretire','medfaminc','percapinc','whitepercap',
                  'blackpercap','indianpercap','asianpercap','otherpercap','hisppercap','numunderpov','pctpopunderpov',
                  'pctless9thgrade','pctnothsgrad','pctbsormore','pctunemployed','pctemploy','pctemplmanu','pctemplprofserv',
                  'PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv','TotalPctDiv',
                  'PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids',
                  'PctWorkMom','NumIlleg','PctIlleg','NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10',
                  'PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam',
                  'PctLargHouseOccup','PersPerOccupHous','PersPerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous',
                  'PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt',
                  'PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal','OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRent',
                  'MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet','PctForeignBorn','PctBornSameState','PctSameHouse85',
                  'PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop',
                  'PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor',
                  'OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg',
                  'LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop','ViolentCrimesPerPop')

# Check for cols where '?' appears in rows and drop that column
for (cols in 1:length(colnames(df))) {
  print(c(colnames(df)[cols], length(df[which(df[,cols] == '?'), cols])))
}

drop_cols = c('LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq',
              'LemasTotReqPerPop','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack',
              'PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz','PolicAveOTWorked',
              'PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','PolicBudgPerPop')

df = df[,!colnames(df) %in% drop_cols]

# Remove Metadata Columns
df = df[-1:-5]

# Set seed for reproducibility and debugging
set.seed(99)

n = dim(df)[1]
p = dim(df)[2] - 1

X = as.data.frame(df[,-101])
y = as.data.frame(df[,101])

## For final run - set iterations to 100
iterations                   = 1

# Resulting data frames
lasso.results                = data.frame(method="LASSO", matrix(ncol=3, nrow=iterations))
ridge.results                = data.frame(method="RIDGE", matrix(ncol=3, nrow=iterations))
randf.results                = data.frame(method="RANDF", matrix(ncol=3, nrow=iterations))
column.Names                 = c("METHOD","R2.TRAIN", "R2.TEST", "FITTING.TIME")
colnames(lasso.results)      = column.Names
colnames(ridge.results)      = column.Names
colnames(randf.results)      = column.Names

lasso.coeff.mat = data.frame(matrix(ncol=length(colnames(X)), nrow=1))
ridge.coeff.mat = data.frame(matrix(ncol=length(colnames(X)), nrow=1))
randf.coeff.mat = data.frame(matrix(ncol=length(colnames(X)), nrow=1))

colnames(lasso.coeff.mat) = c(colnames(X))
colnames(ridge.coeff.mat) = c(colnames(X))
colnames(randf.coeff.mat)    = c(colnames(X))

# Settings for cv.fit plots
par(mar = c(5, 4, 4, 2) + 0.1)
par(mfrow=c(2,1))

## Run for Lasso, Ridge
alphas = c(1, 0)

for (run in seq(iterations)) {
  runStart      = Sys.time()
  # Take 80/20 split
  sample.idx    = sample(nrow(X), size = n * 0.8)

  X.train       = data.matrix(X[sample.idx,])
  X.test        = data.matrix(X[-sample.idx,])
  y.train       = y[sample.idx,]
  y.test        = y[-sample.idx,]
  
  n.train = length(y.train)
  n.test = length(y.test)
  y.bar.train = mean(y.train)
  y.bar.test = mean(y.test)

  # Regression
  for (a in alphas){
    logisticType = ""
    if (a==1)  { logisticType = "LASSO"}
    if (a==0)  { logisticType = "RIDGE"}
  
    # Fitting model
    start         = Sys.time()
    cv.fit        = cv.glmnet(X.train, y.train, alpha=a, nfolds=10, intercept=TRUE)
    fit           = glmnet(X.train, y.train, alpha=a, lambda=cv.fit$lambda.1se, intercept=TRUE)
    end           = Sys.time()
    time          = end - start
    
    beta0.hat     = fit$a0
    beta.hat      = fit$beta
    y.train.hat   = predict(fit, newx = X.train)
    y.test.hat    = predict(fit, newx = X.test)
    
    # Calculate train and test R2
    train.num     <- (1/n.train)*(sum((y.train - y.train.hat)^2))
    train.denom   <- (1/n.train)*(sum((y.train - y.bar.train)^2))
    
    test.num      <- (1/n.test)*(sum((y.test - y.test.hat)^2))
    test.denom    <- (1/n.test)*(sum((y.test - y.bar.test)^2))
    
    r2.train      <- 1 - (train.num / train.denom)
    r2.test       <- 1 - (test.num / test.denom)
    
    # Populate results data frame
    if (logisticType == "LASSO")      {lasso.results[run,2:4]       <- c(r2.train, r2.test, time)}
    if (logisticType == "RIDGE")      {ridge.results[run,2:4]       <- c(r2.train, r2.test, time)}
    
    if (run == iterations && logisticType == "LASSO")      {lasso.coeff.mat[1,]       <- t(beta.hat)}
    if (run == iterations && logisticType == "RIDGE")      {ridge.coeff.mat[1,]       <- t(beta.hat)}

    # Print cv.fit plot for logistic method on last iteration
    if (run == iterations) {
      plot(cv.fit)
      title(main=logisticType, line=3)
    }

    # Screen progress output
    print(sprintf("Run %i: %s Fitting Runtime: %3.4f seconds, Train R2: %.4f Test R2: %.4f", run, logisticType, time, r2.train, r2.test))
  }
  
  # Random forest
  dat.train     = data.frame(X.train, y.train)
  dat.test      = data.frame(X.test, y.test)

  start         = Sys.time()

  # We examined Random Forest up to 100 Trees and found 40 to be the optimal point
  # This is optimal due to the error flattening out
  rf.fit                = randomForest(y.train~., data = dat.train, mtry = sqrt(p), ntree=40, importance=TRUE)
  end                   = Sys.time()
  time                  = end - start

  y.train.hat.rf        = predict(rf.fit, dat.train)
  y.test.hat.rf         = predict(rf.fit, dat.test)

  train.num.rf          <- (1/n.train)*(sum((y.train - y.train.hat.rf)^2))
  train.denom.rf        <- (1/n.train)*(sum((y.train - y.bar.train)^2))

  test.num.rf           <- (1/n.test)*(sum((y.test - y.test.hat.rf)^2))
  test.denom.rf         <- (1/n.test)*(sum((y.test - y.bar.test)^2))

  r2.train.rf           <- 1 - (train.num.rf / train.denom.rf)
  r2.test.rf            <- 1 - (test.num.rf / test.denom.rf)

  randf.results[run,2:4] <- c(r2.train.rf, r2.test.rf, time)
  print(sprintf("Run %i: %s Fitting Runtime: %3.4f seconds, Train R2: %.4f Test R2: %.4f", run, "RANDF", time, r2.train.rf, r2.test.rf))

  if (run == iterations)      {rfImportance     <- as.data.frame(importance(rf.fit))
                               randf.coeff.mat[1,] <- t(rfImportance$`%IncMSE`)
                              }

  # Iteration completed, printing total time for run
  runEnd        = Sys.time()
  runTime       = runEnd - runStart
  print(sprintf("Run %i took %3.4f seconds", run, runTime))
}

## Final dataframe
finalResults = rbind(lasso.results,ridge.results, randf.results)

# R^2 boxplots
library(ggpubr)
par(mfrow = c(2,1))
gg_test = ggplot(finalResults, aes(x=METHOD, y=R2.TEST, fill=METHOD)) +
  geom_boxplot() +
  ylim(0,1) +
  ylab("")+
  guides(fill="none") +
  ggtitle('Test R^2') +
  theme_bw()

gg_train = ggplot(finalResults, aes(x=METHOD, y=R2.TRAIN, fill=METHOD)) +
  geom_boxplot() +
  ylim(0,1) +
  ylab('R^2') +
  guides(fill="none") +
  ggtitle('Train R^2') +
  theme_bw()

ggarrange(gg_train, gg_test,
          ncol = 2, nrow = 1)


# Create coefficient matrix and plot
lasso.coeff.mat         = lasso.coeff.mat[, order(lasso.coeff.mat, decreasing = TRUE)]
ridge.coeff.mat         = ridge.coeff.mat[, order(lasso.coeff.mat, decreasing = TRUE)]
randf.coeff.mat         = randf.coeff.mat[, order(lasso.coeff.mat, decreasing = TRUE)]
all.coeff.mat           = rbind(lasso.coeff.mat, ridge.coeff.mat, randf.coeff.mat)

all.coeff.mat           = cbind(seq(1:200), t(all.coeff.mat))
colnames(all.coeff.mat) = c('Lasso Net Ordered Index', 'Lasso', 'Ridge', 'Random_Forest')

all.coeff.mat           = as.data.frame(all.coeff.mat)

par(mfrow=c(3,1))
par(mar = c(2, 4, 2, 2))
barplot(all.coeff.mat$Lasso, horiz=FALSE, main = 'Lasso', ylab = 'Coefficient Value', col = 'blue')
barplot(all.coeff.mat$Ridge, horiz=FALSE, main = 'Ridge', ylab = 'Coefficient Value', col = 'red')
barplot(all.coeff.mat$Random_Forest, horiz=FALSE, main = 'Random Forest', ylab = 'IncMSE', col = 'green')


# Find median Test AUC and Fit Times
median(finalResults[finalResults$METHOD == 'LASSO',]$R2.TEST)
median(finalResults[finalResults$METHOD == 'RIDGE',]$R2.TEST)
median(finalResults[finalResults$METHOD == 'RANDF',]$R2.TEST)

median(finalResults[finalResults$METHOD == 'LASSO',]$FITTING.TIME)
median(finalResults[finalResults$METHOD == 'RIDGE',]$FITTING.TIME)
median(finalResults[finalResults$METHOD == 'RANDF',]$FITTING.TIME)

sum(finalResults[finalResults$METHOD == 'LASSO',]$FITTING.TIME)
sum(finalResults[finalResults$METHOD == 'RIDGE',]$FITTING.TIME)
sum(finalResults[finalResults$METHOD == 'RANDF',]$FITTING.TIME)

#  Random notes / development
varImpPlot(rf.fit)

o <- order(abs(all.coeff.mat$Random_Forest), decreasing = TRUE)
all.coeff.mat[o, ]

ggplot(finalResults, aes(x=R2.TEST)) +
  geom_histogram(aes(color=METHOD, fill=METHOD), bins=30, position="identity", alpha = 0.4)+
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#373F7A")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#373F7A"))

## 90% R2.TEST Confidence Interval for last run of each method
r2.lasso.se = finalResults %>% filter(METHOD=="LASSO") %>% pull(R2.TEST) %>% sd
r2.ridge.se = finalResults %>% filter(METHOD=="RIDGE") %>% pull(R2.TEST) %>% sd
r2.randf.se = finalResults %>% filter(METHOD=="RANDF") %>% pull(R2.TEST) %>% sd
r2.lasso.ci = c(finalResults[c(100),]$R2.TEST - (1.645*r2.lasso.se), finalResults[c(100),]$R2.TEST + (1.645*r2.lasso.se))
r2.ridge.ci = c(finalResults[c(200),]$R2.TEST - (1.645*r2.ridge.se), finalResults[c(200),]$R2.TEST + (1.645*r2.ridge.se))
r2.randf.ci = c(finalResults[c(300),]$R2.TEST - (1.645*r2.randf.se), finalResults[c(300),]$R2.TEST + (1.645*r2.randf.se))

## 90% R2.TEST Confidence Interval using quantile function
r2.lasso    =  finalResults %>% filter(METHOD=="LASSO") %>% pull(R2.TEST)
r2.ridge    =  finalResults %>% filter(METHOD=="RIDGE") %>% pull(R2.TEST)
r2.randf    =  finalResults %>% filter(METHOD=="RANDF") %>% pull(R2.TEST)
quantile(r2.lasso, probs=c(0.05,0.95))
quantile(r2.ridge, probs=c(0.05,0.95))
quantile(r2.randf, probs=c(0.05,0.95))
