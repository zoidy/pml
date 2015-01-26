#set working directory first!

library(Hmisc)

#loading the data ======================================================

alldata_raw=csv.get('pml-training.csv')

#remove all columns which have NA 
#(keep the cols where the count of NA is zero)
alldata=alldata_raw[,colSums(is.na(alldata_raw)) == 0]

#remove all columns which have ''
#(keep the cols where the count of '' is zero)
alldata=alldata[,colSums(alldata=='') == 0]

column_names=names(alldata)

#get subsets of columns which have all valid data and print their names
idx_forearm=grep('forearm',column_names)
column_names[idx_forearm]

idx_dumbbell=grep('dumbbell',column_names)
column_names[idx_dumbbell]

idx_arm=grep('.arm.',column_names,fixed=TRUE)
column_names[idx_arm]

idx_belt=grep('belt',column_names)
column_names[idx_belt]

# partition data============================================================
library(caret)
idxtrain=createDataPartition(alldata$classe,p=.7,list=FALSE)
datatrain=alldata[idxtrain,]
datatest=alldata[-idxtrain,]

# EDA =====================================================================
summary(datatrain)

#No useful info is gained here.
#All potential predictors have many unique values
nearZeroVar(datatrain,saveMetrics=T)

featurePlot(x=datatrain[,c(8,9,10,11)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(12,13,14,15)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(16,17,18,19)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(20,21,22,23)],y = datatrain$classe,plot="pairs")

featurePlot(x=datatrain[,c(24,25,26,27)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(28,29,30,31)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(32,33,34,35)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(36,37,38,39)],y = datatrain$classe,plot="pairs")

featurePlot(x=datatrain[,c(40,41,42,43)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(44,45,46,47)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(48,49,50,51)],y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c(52,53,54,55)],y = datatrain$classe,plot="pairs")

featurePlot(x=datatrain[,c(56,57,58,59)],y = datatrain$classe,plot="pairs")

featurePlot(x=datatrain[,c('roll.belt','pitch.belt','yaw.belt',
                           'roll.arm','pitch.arm','yaw.arm')],
            y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c('roll.dumbbell','pitch.dumbbell','yaw.dumbbell',
                           'roll.forearm','pitch.forearm','yaw.forearm')],
            y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c('roll.dumbbell','pitch.dumbbell','yaw.dumbbell',
                           'roll.arm','pitch.arm','yaw.arm')],
            y = datatrain$classe,plot="pairs")
featurePlot(x=datatrain[,c('roll.belt','pitch.belt','yaw.belt',
                           'roll.dumbbell','pitch.dumbbell','yaw.dumbbell')],
            y = datatrain$classe,plot="pairs")

#these look promising, maybe for  (Class A), throwing the elbows to the front
#and throwing the hips to the front (Class E)
featurePlot(x=datatrain[,c('total.accel.belt','total.accel.forearm','total.accel.arm',
                           'total.accel.dumbbell',
                           'gyros.dumbbell.x','gyros.dumbbell.y','gyros.dumbbell.z')],
            y = datatrain$classe,plot="pairs")

featurePlot(x=datatrain[,c('total.accel.belt','total.accel.dumbbell',
                           'magnet.arm.x','magnet.arm.y','magnet.arm.z')],
            y = datatrain$classe,plot="pairs")

#magnet.arm.x, gyros.dumbbell.x possible candidates?
featurePlot(x=datatrain[,c('gyros.dumbbell.x','gyros.dumbbell.y','gyros.dumbbell.z',
                           'magnet.belt.x','magnet.belt.y','magnet.belt.z',
                           'magnet.arm.x','magnet.arm.y','magnet.arm.z')],
            y = datatrain$classe,plot="pairs")


#1st attempt===============================================================
library(rattle)

#extract only the desired columns
cols=c('total.accel.belt','total.accel.forearm','total.accel.arm','total.accel.dumbbell',
       'gyros.dumbbell.x','gyros.dumbbell.z','magnet.arm.x','classe')
cols=c('total.accel.belt','total.accel.forearm','total.accel.arm',
       'total.accel.dumbbell','classe',
       'gyros.dumbbell.x','gyros.dumbbell.y','gyros.dumbbell.z')
cols=c('roll.dumbbell',
       'pitch.dumbbell',
       'total.accel.belt',
       'accel.dumbbell.y','accel.arm.y','accel.arm.z',
       'magnet.dumbbell.y',
       'gyros.dumbbell.y','gyros.belt.z','gyros.dumbbell.z',
       'classe')
cols=c('total.accel.belt','total.accel.arm','total.accel.dumbbell',
       'gyros.dumbbell.y','gyros.dumbbell.z',       
       'gyros.forearm.y','gyros.forearm.z',
       'gyros.arm.y','gyros.arm.z',
       'magnet.dumbbell.x','magnet.dumbbell.y','magnet.dumbbell.z',
       'roll.dumbbell',
       'pitch.dumbbell','pitch.arm',
       'classe')
cols=c('total.accel.belt',
       'accel.forearm.y','accel.forearm.z',
       'gyros.dumbbell.y','gyros.dumbbell.z',       
       'gyros.forearm.y','gyros.forearm.z',
       'gyros.arm.y','gyros.arm.z',
       'magnet.dumbbell.x','magnet.dumbbell.y','magnet.dumbbell.z',
       'roll.dumbbell',
       'pitch.dumbbell','pitch.arm',
       'classe')

datatrain_subset=datatrain[,cols]
modFit <- train(classe ~ .,method="rpart",data=datatrain_subset)
pred1=predict(modFit,newdata=datatrain_subset)
confusionMatrix(pred1,datatrain_subset$classe)

fancyRpartPlot(modFit$finalModel)
print(modFit$finalModel)


#2nd attempt - everything=============================================================
cols=c('X','user.name','raw.timestamp.part.1','raw.timestamp.part.2',
       'cvtd.timestamp','new.window','num.window')
datatrain_subset=datatrain[,!names(datatrain) %in% cols] #http://stackoverflow.com/a/5236518
modFit=train(classe ~ .,method="rf",data=datatrain_subset,nodesize=5,
                do.trace=T,sampsize=100,ntree=100)
pred1=predict(modFit,newdata=datatrain)
confusionMatrix(pred1,datatrain$classe)
print(modFit$finalModel)

#2nd attempt - everything w/different params===========================================
cols=c('X','user.name','raw.timestamp.part.1','raw.timestamp.part.2',
       'cvtd.timestamp','new.window','num.window')
datatrain_subset=datatrain[,!names(datatrain) %in% cols] #http://stackoverflow.com/a/5236518

modFit=train(classe ~ .,method="rf",data=datatrain_subset,nodesize=5,
             do.trace=T,sampsize=100,ntree=200)

pred1=predict(modFit,newdata=datatrain)
confusionMatrix(pred1,datatrain$classe)
print(modFit$finalModel)



#4th attempty - pca============================================================
cols=c('X','user.name','raw.timestamp.part.1','raw.timestamp.part.2',
       'cvtd.timestamp','new.window','num.window','classe')
datatrain_subset=datatrain[,!names(datatrain) %in% cols]
pp=preProcess(datatrain_subset,method=c('YeoJohnson','pca'),thresh=.85)
pp$numComp
datatrain_pc=predict(pp,datatrain_subset) #project training data to the pca components
modFit=train(datatrain$classe ~ .,method='rpart',data=datatrain_pc)
pred1=predict(modFit,newdata=datatrain_pc)
confusionMatrix(pred1,datatrain$classe)

#check the testing set ========================================================
pred2=predict(modFit,newdata=datatest)
confusionMatrix(pred2,datatest$classe)

#chek the test cases=========================================
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

#load and clean data
testdata_raw=csv.get('pml-testing.csv')
testdata=testdata_raw[,colSums(is.na(testdata_raw)) == 0]
testdata=testdata[,colSums(alldata=='') == 0]

pred3=predict(modFit,newdata=testdata)
pml_write_files(pred3)
