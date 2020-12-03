setwd("C:\\Users\\Giovanni\\Desktop")
setwd("/home/dede/Documents/git/midst")
rm(list=ls())
library(dplyr)

# ---------------------------------------------------------------------------- #

data = read.csv2("log.txt", header = TRUE, sep = ",", dec = ".")
data = data[,c(2,4:7,9:11)]
colnames(data) = c("file","algorithm","initialization","centroids",
		"fuzzy_weight","init_time","algorithm_time","iteration")
data$stop = grepl("*", data$iteration, fixed=TRUE)
data$iteration = as.numeric(gsub("[*]", "", data$iteration))
data$tot_time = data$init_time + data$algorithm_time
#View(data)


# ---------------------------------------------------------------------------- #

# Definisco il livello di specificazione con la var. identifier
data$identifier = paste(data$file, data$algorithm, data$initialization,
                        data$centroids, data$fuzzy_weight, sep="_")
id_list = unique(data$identifier)
length(id_list) # 144 = 4 * 3 * (1 * 3 + 3 * 3)
# gruppi = file * init * (1 * centroids + centroids * fuzzy_weight)

df = data.frame(id=character(), num=integer(),init_time=double(),algorithm_time=double(),
			iteration=double(),tot_time=double(), init_time_sd=double(),algorithm_time_sd=double(),
			iteration_sd=double(),tot_time_sd=double(), stringsAsFactors=FALSE)
for (id in id_list) {
	c_data = data[id==data$identifier,c("init_time","algorithm_time","iteration","tot_time")]
	means = t(as.data.frame(colMeans(c_data)))
	sds = t(as.data.frame(apply(c_data, 2, sd)))
	colnames(sds) = c("init_time_sd","algorithm_time_sd","iteration_sd","tot_time_sd")
	df = rbind(df, cbind( id, data.frame("num"=NROW(c_data)), means, sds ))
}
# Il seguente comando ridivide la var. identifier nelle sue componenti
df = df %>% tidyr::separate(id, c("file","algorithm","initialization","centroids","fuzzy_weight"), sep="_")
rownames(df) = c()
View(df)


# ---------------------------------------------------------------------------- #

### GRAFICI

#         nero :  15
#        rosso :  30
#        verde :  70
#          blu : 115
# --------------------
# 15  quadrato :   1.5
# 16   pallino :   2
# 17 triangolo :   3


# centroidi da considerare
for(k in 3:5){

    png(paste0("./doc/figures/clust-",k,"-time.png"), width=2500, height=1280,
    pointsize=8, res=300)
    par(mfrow=c(1,2))

    # dev.off()

    # K-MEANS: (init, iterazioni)
    # png(paste0("./doc/figures/km-",k))
    gg = df[df$centroids==k,]
    gg$initialization = factor(gg$initialization, levels=c("random","plusplus"))
    gg = gg[with(gg, order(initialization, algorithm, fuzzy_weight)), c("initialization","tot_time","file","fuzzy_weight")]

    plot.default(gg[,1],gg[,2], type="n", axes = FALSE, xlab="Initialization", ylab="Time",
    main = paste0("k-means, k=",k))
    axis(side = 1, at = as.numeric(gg[,1]), labels = gg[,1])
    axis(side=2, at=seq(0,500,25), labels=seq(0,500,25))

    file_list = c("data/affitti15.csv","data/affitti30.csv","data/affitti70.csv","data/affitti115.csv")
    fuzzy_list = c("-")

    for (i in 1:length(file_list)) {
        for (j in 1:length(fuzzy_list)) {
            d = gg[gg$file==file_list[i] & gg$fuzzy_weight==fuzzy_list[j],]
            lines(d[,1], d[,2], col=i, lty=2)
            points(d[,1], d[,2], col=i, pch=19)
        }
    }

    # FUZZY C-MEANS: (init, iterazioni)

    gg = df[df$centroids==k,]
    gg$initialization = factor(gg$initialization, levels=c("random","step","plusplus"))
    gg = gg[with(gg, order(initialization, algorithm, fuzzy_weight)), c("initialization","tot_time","file","fuzzy_weight")]

    plot.default(gg[,1],gg[,2], type="n", axes = FALSE, xlab="Initialization", ylab="Time",
    main = paste0("Fuzzy k-means, k=",k))
    axis(side = 1, at = as.numeric(gg[,1]), labels = gg[,1])
    axis(side=2, at=seq(0,500,25), labels=seq(0,500,25))

    file_list = c("data/affitti15.csv","data/affitti30.csv","data/affitti70.csv","data/affitti115.csv")
    fuzzy_list = c("1.5","2","3")

    for (i in 1:length(file_list)) {
        for (j in 1:length(fuzzy_list)) {
            if(j==1){
                poi = "1.5"
            }
            if(j==2){
                poi = "2"
            }
            if(j==3){
                poi = "3"
            }
            
            d = gg[gg$file==file_list[i] & gg$fuzzy_weight==fuzzy_list[j],]
            lines(d[,1], d[,2], col=i, lty=2, type="b", pch=" ")
            text(d[,1], d[,2], col=i, label=poi, cex=1.2)
        }
    }
    
    dev.off()
}
# ---------------------------------------------------------------------------- #
