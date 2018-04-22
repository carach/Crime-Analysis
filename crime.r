View(communities)
myData = communities[,-c(1:5,102:118,122:125,127)]
summary(myData)
myData$V31 = as.numeric(myData$V31)
summary(myData$V128)

# First we will build a linear regression model
# and cross validate it.
# Attach the "boot" package
myData_reg = glm(V128~., data = myData)
cv.out = cv.glm(myData, myData_reg, K=10)
cv.out$delta[1]
# cv.out$delta[1] = 0.01837187

# Next we will build a principle compoment regression
# Attach the "pls" package.
myData.pcr = pcr(V128~., data = myData, scale=TRUE, validation="CV")
myData.pcr$validation$adj
# when 38 comps msep = 0.01832859
validationplot(myData.pcr, val.type="MSEP")
# MSEP value see pcrValiPlot.pdf

# Next we will try Ridge and Lasso. Note we use the built
# in CV function to select the best lambda, and this will
# also give the test MSE error estimate for that lambda.
# Attach the glmnet package first.
# Lasso
x = model.matrix(V128~.,myData)[,-1]
y = myData$V128
cv.out=cv.glmnet(x,y,alpha=0)
names(cv.out)
cv.out$lambda.min
cv.out$cvm
min(cv.out$cvm)
# cv.out$lambda.min = 0.01887681
# min(cv.out$cvm) = 0.01853674

#Ridge
cv.out=cv.glmnet(x,y,alpha=1)
min(cv.out$cvm)
# min(cv.out$cvm) = 0.01839613

# Now let's try KNN. Attach the "FNN" package.
# Let's first do a simple CV. We select a set of training
# rows to use.
train = sample(1:1994,1994*0.75)
train.x = scale(x[train,])
test.x = scale(x[-train,])
train.y = y[train]
test.y = y[-train]
knn.fit = knn.reg(train.x, test.x, train.y, k=5)
mean((test.y - knn.fit$pred)^2)
# [1] 0.01952474

# Somewhat worse, but K was arbitrary and this is simple CV.
# Let's first see if we can find a better k
errs = rep(0,15)
for(i in 1:15){
knn.fit = knn.reg(train.x, test.x, train.y, k=i)
errs[i] = mean((test.y - knn.fit$pred)^2)
}
errs
#  [1] 0.03146613 0.02070466 0.02028110 0.02009069 0.01952474 0.01918457 0.01950221
#  [8] 0.01997255 0.01998732 0.01995839 0.01997244 0.01978598 0.01991306 0.02018330
#  [15] 0.02013273

# Next do k-fold CV
bins = sample(1:10,1994, replace = TRUE)

# Note this vector will assign every row in the data set to a bin
binErrs = rep(0,10)
for(k in 1:10){
train.x = scale(x[bins != k,])
test.x = scale(x[bins == k,])
train.y = y[bins != k]
test.y = y[bins == k]
knn.fit = knn.reg(train.x, test.x, train.y, k=8)
binErrs[k] = mean((test.y - knn.fit$pred)^2)
}
mean(binErrs)
# [1] 0.02110787

# Combine them together
errs = rep(0,15)
for(i in 1:15){
  for(k in 1:10){
  train.x = scale(x[bins != k,])
  test.x = scale(x[bins == k,])
  train.y = y[bins != k]
  test.y = y[bins == k]
  knn.fit = knn.reg(train.x, test.x, train.y, k=i)
  binErrs[k] = mean((test.y - knn.fit$pred)^2)
  }
  errs[i] = mean(binErrs)
}
errs
# [1] 0.03506495 0.02581563 0.02353328 0.02197404 0.02174078 0.02155901 0.02143634
# [8] 0.02110787 0.02080396 0.02074517 0.02069507 0.02069510 0.02090470 0.02096175
# [15] 0.02090486
# No etter than simple CV
# PCR gave the best answer with 38 or more components