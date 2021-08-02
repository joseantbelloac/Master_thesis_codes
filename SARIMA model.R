############ CHAP 0 PART 0 REMOVING THE VARIABLES AND PLOTS AND CALLING THE LIBRARIES ###############################
rm(list = ls()) #elimina variables preexistentes
dev.off()
dev.off()
library(ggplot2)
library(ggfortify)
library(forecast)
library(uroot)
library(tseries)
library(urca)
library(lmtest)
library(pastecs)

############ CHAP 0 PART 1 READ THE DATA #########################

#setwd("/home/jose/Downloads/data colab U central") #cambia ubicación
setwd("C:/Users/JOSE/Downloads/data colab U central/data colab U central") #cambia ubicación
setwd("D:/ThesisExperiments/OriginalData/Energy/")

B116<-read.csv("B116.csv",header = T, sep = ",") #lee archivo
B124<-read.csv("B124.csv",header = T, sep = ",")
B123<-read.csv("B123.csv",header = T, sep = ",")
B011<-read.csv("B011.csv",header = T, sep = ",")
B002<-read.csv("B002.csv",header = T, sep = ",")


############ CHAP 0 PART 2 BOXPLOT FOR EACH BUILDING ##################
jpeg(filename = "B116_Boxplot.jpeg", height = 786, width = 1048)
boxplot(B116$Ef_kWh, main="Box Plot B116") #diagrama de caja que evidencia la presencia de datos anómalos
dev.off()
jpeg(filename = "B124_Boxplot.jpeg", height = 786, width = 1048)
boxplot(B124$Ef_kWh, main="Box Plot B124")
dev.off()
jpeg(filename = "B123_Boxplot.jpeg", height = 786, width = 1048)
boxplot(B123$Ef_kWh, main="Box Plot B123")
dev.off()
jpeg(filename = "B011_Boxplot.jpeg", height = 786, width = 1048)
boxplot(B011$Ef_kWh, main="Box Plot B011")
dev.off()
jpeg(filename = "B002_Boxplot.jpeg", height = 786, width = 1048)
boxplot(B002$Ef_kWh, main="Box Plot B002")
dev.off()

############ CHAP 0 PART 3 RESIZING THE DATA FOR NN ###############
AA<-matrix(0,nrow = (length(B116$Ef_kWh)-24),ncol = 25)
datafiles<-c("B002","B011","B116","B123","B124")
for(dfiles in datafiles){
  j<-get(dfiles)
  for(i in (1:(length(B116$Ef_kWh)-24))){
    AA[i,1:24]<-j$Ef_kWh[i:(i+23)]
    AA[i,25]<-j$Ef_kWh[i+24]
  }
  write.csv(AA,file=paste(dfiles,"_dayNN.csv",sep=""),row.names = F)
}

AAA<-matrix(0,nrow = (length(B116$Ef_kWh)-(24*7)),ncol = (24*7+1))
datafiles<-c("B002","B011","B116","B123","B124")
for(dfiles in datafiles){
  j<-get(dfiles)
  for(i in (1:(length(B116$Ef_kWh)-(24*7)))){
    AAA[i,1:(24*7)]<-j$Ef_kWh[i:(i+(24*7-1))]
    AAA[i,(24*7+1)]<-j$Ef_kWh[(i+(24*7))]
  }
  write.csv(AAA,file=paste(dfiles,"_weekNN.csv",sep=""),row.names = F, col.names = F)
}


#for(j in c(24, (24*7))){
#  a<-1
#  if(j==24){tim<-"day"} else {tim<-"week"}
#  for(i in 1:26){
#	jpeg(paste("C:/Users/JOSE/Downloads/data colab U central/data colab U central/Images/B116/B116stlfq",tim,i,".jpg"))
#	plot(stl(ts(B116$Ef_kWh[a:(4200+(i*24*7))],frequency = j),"per"), main=paste("Seasonal Decomposition of Time series by", tim, sep = " " ))
#	a<-a+(24*7)
#	dev.off()
#  }
#}
#B116.tsmes<-ts(B116$Ef_kWh[1:(24*7*36)], frequency = (24*7*4))

############ CHAP 1  PART 1 MAKING THE TIME SERIES FORMAT FROM THE WHOLE SET OF DATA #######################

B116.tsweek<-ts(B116$Ef_kWh[1:4392], frequency = (24*7)) #cambia los datos a formato serie de tiempo
B116.tsday<-ts(B116$Ef_kWh[1:4392], frequency = (24)) #cambia los datos a formato serie de tiempo
B116.ts2week<-ts(B116$Ef_kWh, frequency = (24*7))
B116.ts2day<-ts(B116$Ef_kWh, frequency = (24))

B011.tsweek<-ts(B011$Ef_kWh[1:4392], frequency = (24*7)) #cambia los datos a formato serie de tiempo
B011.tsday<-ts(B011$Ef_kWh[1:4392], frequency = (24)) #cambia los datos a formato serie de tiempo
B011.ts2week<-ts(B011$Ef_kWh, frequency = (24*7))
B011.ts2day<-ts(B011$Ef_kWh, frequency = (24))

B123.tsweek<-ts(B123$Ef_kWh[1:4392], frequency = (24*7)) #cambia los datos a formato serie de tiempo
B123.tsday<-ts(B123$Ef_kWh[1:4392], frequency = (24)) #cambia los datos a formato serie de tiempo
B123.ts2week<-ts(B123$Ef_kWh, frequency = (24*7))
B123.ts2day<-ts(B123$Ef_kWh, frequency = (24))

B124.tsweek<-ts(B124$Ef_kWh[1:4392], frequency = (24*7)) #cambia los datos a formato serie de tiempo
B124.tsday<-ts(B124$Ef_kWh[1:4392], frequency = (24)) #cambia los datos a formato serie de tiempo
B124.ts2week<-ts(B124$Ef_kWh, frequency = (24*7))
B124.ts2day<-ts(B124$Ef_kWh, frequency = (24))

B002.tsweek<-ts(B002$Ef_kWh[1:4392], frequency = (24*7)) #cambia los datos a formato serie de tiempo
B002.tsday<-ts(B002$Ef_kWh[1:4392], frequency = (24)) #cambia los datos a formato serie de tiempo
B002.ts2week<-ts(B002$Ef_kWh, frequency = (24*7))
B002.ts2day<-ts(B002$Ef_kWh, frequency = (24))

############ CHAP 1 PART 2 PLOTING SEASONAL DECOMPOSITION OF TIME SERIES ####################

#grafica de descomposicion de serie de timepo
plot(stl(B116.tsday, "per"),main="B116 by days, Seasonal Decomposition (half of data)")
plot(stl(B116.ts2day, "per"),main="B116 by days, Seasonal Decomposition (whole data)")
plot(stl(B116.tsweek, "per"),main="B116 by weeks, Seasonal Decomposition (half of data)")
plot(stl(B116.ts2week, "per"),main="B116 by weeks, Seasonal Decomposition (whole of data)")

plot(stl(B011.tsday, "per"),main="B011 by days, Seasonal Decomposition (half of data)")
plot(stl(B011.ts2day, "per"),main="B011 by days, Seasonal Decomposition (whole data)")
plot(stl(B011.tsweek, "per"),main="B011 by weeks, Seasonal Decomposition (half of data)")
plot(stl(B011.ts2week, "per"),main="B011 by weeks, Seasonal Decomposition (whole of data)")

plot(stl(B123.tsday, "per"),main="B123 by days, Seasonal Decomposition (half of data)")
plot(stl(B123.ts2day, "per"),main="B123 by days, Seasonal Decomposition (whole data)")
plot(stl(B123.tsweek, "per"),main="B123 by weeks, Seasonal Decomposition (half of data)")
plot(stl(B123.ts2week, "per"),main="B123 by weeks, Seasonal Decomposition (whole of data)")

plot(stl(B124.tsday, "per"),main="B124 by days, Seasonal Decomposition (half of data)")
plot(stl(B124.ts2day, "per"),main="B124 by days, Seasonal Decomposition (whole data)")
plot(stl(B124.tsweek, "per"),main="B124 by weeks, Seasonal Decomposition (half of data)")
plot(stl(B124.ts2week, "per"),main="B124 by weeks, Seasonal Decomposition (whole of data)")

plot(stl(B002.tsday, "per"),main="B002 by days, Seasonal Decomposition (half of data)") #grafica de descomposición de serie de timepo
plot(stl(B002.ts2day, "per"),main="B002 by days, Seasonal Decomposition (whole data)")
plot(stl(B002.tsweek, "per"),main="B002 by weeks, Seasonal Decomposition (half of data)")
plot(stl(B002.ts2week, "per"),main="B002 by weeks, Seasonal Decomposition (whole data)")


############ CHAP 1 PART 3 SEASON PLOT FOR EACH TIME SERIES#################
#grafica los datos según la frecuencia que se haya incluido

jpeg(filename = "B116_Seasonalplot_days&weeks.jpeg", height = 786, width = 1048)
par(mfrow=c(1,2))
seasonplot(B116.tsday, season.labels = T, main="Seasonal Plot B116 by days")
seasonplot(B116.tsweek, season.labels = T,main="Seasonal Plot B116 by weeks")
dev.off()

jpeg(filename = "B011_Seasonalplot_days&weeks.jpeg", height = 786, width = 1048)
par(mfrow=c(1,2))
seasonplot(B011.tsday, season.labels = T, main="Seasonal Plot B011 by days")
seasonplot(B011.tsweek, season.labels = T,main="Seasonal Plot B011 by weeks")
dev.off()

jpeg(filename = "B123_Seasonalplot_days&weeks.jpeg", height = 786, width = 1048)
par(mfrow=c(1,2))
seasonplot(B123.tsday, season.labels = T, main="Seasonal Plot B123 by days")
seasonplot(B123.tsweek, season.labels = T,main="Seasonal Plot B123 by weeks")
dev.off()

jpeg(filename = "B124_Seasonalplot_days&weeks.jpeg", height = 786, width = 1048)
par(mfrow=c(1,2))
seasonplot(B124.tsday, season.labels = T, main="Seasonal Plot B124 by days")
seasonplot(B124.tsweek, season.labels = T,main="Seasonal Plot B124 by weeks")
dev.off()

jpeg(filename = "B002_Seasonalplot_days&weeks.jpeg", height = 786, width = 1048)
par(mfrow=c(1,2))
seasonplot(B002.tsday, season.labels = T, main="Seasonal Plot B002 by days")
seasonplot(B002.tsweek, season.labels = T,main="Seasonal Plot B002 by weeks")
dev.off()

############ CHAP 1 PART 4 SPECTRAL PERIODOGRAM FOR EACH TIME SERIES ######################
#grafica el espectrograma, se reduce la escala de x para poder determinar en donde se presenta el pico
B116.esweek<-spec.pgram(B116.tsweek,log="no", xlim=c(0,20))
B116.esday<-spec.pgram(B116.tsday,log="no",xlim=c(0,4))
B116.es2week<-spec.pgram(B116.ts2week,log="no")
B116.es2day<-spec.pgram(B116.ts2day,log="no")

B011.esweek<-spec.pgram(B011.tsweek,log="no")
B011.esday<-spec.pgram(B011.tsday,log="no")
B011.es2week<-spec.pgram(B011.ts2week,log="no")
B011.es2day<-spec.pgram(B011.ts2day,log="no")

B123.esweek<-spec.pgram(B123.tsweek,log="no")
B123.esday<-spec.pgram(B123.tsday,log="no")
B123.es2week<-spec.pgram(B123.ts2week,log="no")
B123.es2day<-spec.pgram(B123.ts2day,log="no")

B124.esweek<-spec.pgram(B123.tsweek,log="no")
B124.esday<-spec.pgram(B123.tsday,log="no")
B124.es2week<-spec.pgram(B123.ts2week,log="no")
B124.es2day<-spec.pgram(B123.ts2day,log="no")

B002.esweek<-spec.pgram(B002.tsweek,log="no")
B002.esday<-spec.pgram(B002.tsday,log="no")
B002.es2week<-spec.pgram(B002.ts2week,log="no")
B002.es2day<-spec.pgram(B002.ts2day,log="no")

############ CHAP 1 PART 5 DETERMINING THE MAIN FREQUENCIES FOR EACH TIME SERIES ###############
orderedweek116<-order(B116.esweek$spec,B116.esweek$freq, decreasing = T)#ordena los datos del espectrograma de mayor a menor
orderedday116<-order(B116.esday$spec,B116.esday$freq, decreasing = T) #ordena los datos del espectrograma de mayor a menor

orderedweek002<-order(B002.esweek$spec,B002.esweek$freq, decreasing = T)#ordena los datos del espectrograma de mayor a menor
orderedday002<-order(B002.esday$spec,B002.esday$freq, decreasing = T) #ordena los datos del espectrograma de mayor a menor

orderedweek011<-order(B011.esweek$spec,B011.esweek$freq, decreasing = T)#ordena los datos del espectrograma de mayor a menor
orderedday011<-order(B011.esday$spec,B011.esday$freq, decreasing = T) #ordena los datos del espectrograma de mayor a menor

orderedweek123<-order(B123.esweek$spec,B123.esweek$freq, decreasing = T)#ordena los datos del espectrograma de mayor a menor
orderedday123<-order(B123.esday$spec,B123.esday$freq, decreasing = T) #ordena los datos del espectrograma de mayor a menor

orderedweek124<-order(B124.esweek$spec,B124.esweek$freq, decreasing = T)#ordena los datos del espectrograma de mayor a menor
orderedday124<-order(B124.esday$spec,B124.esday$freq, decreasing = T) #ordena los datos del espectrograma de mayor a menor

maxfweek<-B011.esweek$freq[187]
24*7/maxfweek
maxfday<-B011.esday$freq[187]
24/maxfday
maxf2week<-B011.esweek$freq[188]
24*7/maxf2week
maxf2day<-B011.esday$freq[188]
24/maxf2day


maxfweek123<-B123.esweek$freq[188]
24*7/maxfweek123
maxfday123<-B123.esday$freq[188]
24/maxfday123
maxf2week123<-B123.esweek$freq[187]
24*7/maxf2week123
maxf2day123<-B123.esday$freq[187]
24/maxf2day123

maxfweek124<-B124.esweek$freq[188]
24*7/maxfweek124
maxfday124<-B124.esday$freq[188]
24/maxfday124
maxf2week124<-B124.esweek$freq[187]
24*7/maxf2week124
maxf2day124<-B124.esday$freq[187]
24/maxf2day124

maxfweek<-B116.esweek$freq[188]
24*7/maxfweek
maxfday<-B116.esday$freq[188]
24/maxfday
maxf2week<-B116.esweek$freq[187]
24*7/maxf2week
maxf2day<-B116.esday$freq[187]
24/maxf2day


maxfweek002<-B002.esweek$freq[188]
24*7/maxfweek002
maxfday002<-B002.esday$freq[188]
24/maxfday002
maxf2week002<-B002.esweek$freq[187]
24*7/maxf2week002
maxf2day002<-B002.esday$freq[187]
24/maxf2day002


#ch.test(B116.ts)
##################HEGYB116<-hegy.test(B116.ts,lag.method = "AIC", boot.args = list( byseason=T)) #este test arroja valor p para el primer estadistico de 0.9502

#kpss.test(B116.ts)#este test no funciona del todo
#kpss.test(B116.tsdia)

############ CHAP 1 PART 5 BOX COX TRANSFORMATION OF THE TIME SERIES ##############

BXCXLAMB116day<-BoxCox.lambda(B116.tsday, method = "loglik") #calcula el valor de lambda para la transformacion boxcox
BXCXB116day<-BoxCox(B116.tsday,BXCXLAMB116day) #hace la transformacion boxcox
BXCXB116day2<-BoxCox(B116.ts2day,BXCXLAMB116day)

BXCXLAMB011day<-BoxCox.lambda(B011.tsday, method = "loglik") 
BXCXB011day<-BoxCox(B011.tsday,BXCXLAMB011day) 
BXCXB011day2<-BoxCox(B011.ts2day,BXCXLAMB011day) 

BXCXLAMB002day<-BoxCox.lambda(B002.tsday, method = "loglik") 
BXCXB002day<-BoxCox(B002.tsday,BXCXLAMB002day) 
BXCXB002day2<-BoxCox(B002.ts2day,BXCXLAMB002day)

BXCXLAMB123day<-BoxCox.lambda(B123.tsday, method = "loglik") 
BXCXB123day<-BoxCox(B123.tsday,BXCXLAMB123day) 
BXCXB123day2<-BoxCox(B123.ts2day,BXCXLAMB123day) 

BXCXLAMB124day<-BoxCox.lambda(B124.tsday, method = "loglik") 
BXCXB124day<-BoxCox(B124.tsday,BXCXLAMB124day)
BXCXB124day2<-BoxCox(B124.ts2day,BXCXLAMB124day)



# 
BXCXLAMB116week<-BoxCox.lambda(B116.tsweek, method = "loglik") #calcula el valor de lambda para la transformacion boxcox
BXCXB116week<-BoxCox(B116.tsweek,BXCXLAMB116week) #hace la transformacion boxcox
# 
# BXCXLAMB011week<-BoxCox.lambda(B011.tsweek, method = "loglik") 
# BXCXB011week<-BoxCox(B011.tsweek,BXCXLAMB011day) 
# 
# BXCXLAMB002week<-BoxCox.lambda(B002.tsweek, method = "loglik") 
# BXCXB002week<-BoxCox(B002.tsweek,BXCXLAMB002day) 
# 
# BXCXLAMB123week<-BoxCox.lambda(B123.tsweek, method = "loglik") 
# BXCXB123week<-BoxCox(B123.tsweek,BXCXLAMB123day) 
# 
# BXCXLAMB124week<-BoxCox.lambda(B124.tsweek, method = "loglik") 
# BXCXB124week<-BoxCox(B124.tsweek,BXCXLAMB124day)
# 


# B116saeasadjdia<-seasadj(stl(BXCXB116dia,s.window = "periodic")) #muestra la grafica de ajuste a la periodicidad
# plot(B116saeasadjdia)
# B002saeasadjdia<-seasadj(stl(BXCXB002dia,s.window = "periodic")) #muestra la gráfica de ajuste a la periodicidad
# plot(B002saeasadjdia)


############ CHAP 1 PART 6 LOOKING FOR DIFFERENCES TO REMOVE THE NON STATIONARY AND THE SEASONAL BEHAVIOR IN TIME SERIES###############
dB116day<-ndiffs(BXCXB116day, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB116day<-nsdiffs(BXCXB116day, m=frequency(BXCXB116day), test="ch", max.D=25)# el valor del componente D  estacional
##D=1

dB124day<-ndiffs(BXCXB124day, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB124day<-nsdiffs(BXCXB124day, m=frequency(BXCXB124day), test="ch", max.D=25)# el valor del componente D  estacional
##D=1

dB123day<-ndiffs(BXCXB123day, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB123day<-nsdiffs(BXCXB123day, m=frequency(BXCXB123day), test="ch", max.D=25)# el valor del componente D  estacional
##D=1

dB002day<-ndiffs(BXCXB002day, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB002day<-nsdiffs(BXCXB002day, m=frequency(BXCXB002day), test="ch", max.D=25)# el valor del componente D  estacional
##D=1

dB011day<-ndiffs(BXCXB011day, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB011day<-nsdiffs(BXCXB011day, m=frequency(BXCXB011day), test="ch", max.D=25)# el valor del componente D  estacional
##D=1



dB116week<-ndiffs(BXCXB116week, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
##d=1
DB116week<-nsdiffs(BXCXB116week, m=frequency(BXCXB116week), test="ch", max.D=25)# el valor del componente D  estacional
##D=1

# 
# dB124week<-ndiffs(BXCXB124week, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
# ##d=1
# DB124week<-nsdiffs(BXCXB124week, m=frequency(BXCXB124week), test="ch", max.D=25)# el valor del componente D  estacional
# ##D=1
# 
# dB123week<-ndiffs(BXCXB123week, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
# ##d=1
# DB123week<-nsdiffs(BXCXB123week, m=frequency(BXCXB123week), test="ch", max.D=25)# el valor del componente D  estacional
# ##D=1
# 
# dB002week<-ndiffs(BXCXB002week, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
# ##d=1
# DB002week<-nsdiffs(BXCXB002week, m=frequency(BXCXB002week), test="ch", max.D=25)# el valor del componente D  estacional
# ##D=1
# 
# dB011week<-ndiffs(BXCXB011week, alpha=0.05, test="kpss", max.d=25)# el valor del componente d  estacionario
# ##d=1
# DB011week<-nsdiffs(BXCXB011week, m=frequency(BXCXB011week), test="ch", max.D=25)# el valor del componente D  estacional
# ##D=1


############ CHAP 1 PART 7 PLOTING THE DIFFERENCIED TRANSFORMED TIME SERIES ############
rutaener<-"D:/ThesisExperiments/OriginalData/Additional plots/TS transformed/"
jpeg(paste(rutaener,"DiffdiffB116.jpeg",sep=""),height = 393, width = 524)
par(mar=c(5,5,5,5))
autoplot(diff(diff(BXCXB116day, lag = 1),lag = 24),ylab = "") +
ggtitle("Stationary and non seasonal energy for B116") +  theme(plot.title = element_text(hjust = 0.5,size = 21))
dev.off()
jpeg(paste(rutaener,"BXCXTSB116.jpeg",sep=""),height = 393, width = 524)
par(mar=c(5,5,5,5))
autoplot(BXCXB116day,ylab = "") +  theme(plot.title = element_text(hjust = 0.5,size = 21)) +
  ggtitle("Box-Cox transformed energy time series for B116")
dev.off()
jpeg(paste(rutaener,"TSB116.jpeg",sep=""),height = 393, width = 524)
par(mar=c(5,5,5,5))
autoplot(B116.tsday,ylab = "") +  theme(plot.title = element_text(hjust = 0.5,size = 21)) +
  ggtitle("Original energy time series for B116")
dev.off()
#transforma la serie en estacionaria con el lag 1 y en estacional con el 24

autoplot(diff(diff(BXCXB002day, lag = 1),lag = 24),"l", main="B002 stationary and non seasonal time series by day") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24

autoplot(diff(diff(BXCXB011day, lag = 1),lag = 24),"l", main="B011 stationary and non seasonal time series by day") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24

autoplot(diff(diff(BXCXB123day, lag = 1),lag = 24),"l", main="B123 stationary and non seasonal time series by day") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24

autoplot(diff(diff(BXCXB124day, lag = 1),lag = 24),"l", main="B124 stationary and non seasonal time series by day") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24


autoplot(diff(diff(BXCXB116week, lag = 1),lag = 168), main="B116 stationary and non seasonal time series by week") #transforma la serie en estacionaria con el lag 1 y en estacional con el 168
# 
# autoplot(diff(diff(BXCXB002week, lag = 1),lag = 24),"l", main="B002 stationary and non seasonal time series by week") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24
# 
# autoplot(diff(diff(BXCXB011week, lag = 1),lag = 24),"l", main="B011 stationary and non seasonal time series by week") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24
# 
# autoplot(diff(diff(BXCXB123week, lag = 1),lag = 24),"l", main="B123 stationary and non seasonal time series by week") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24
# 
# autoplot(diff(diff(BXCXB124week, lag = 1),lag = 24),"l", main="B124 stationary and non seasonal time series by week") #transforma la serie en estacionaria con el lag 1 y en estacional con el 24



############ CHAP 1 PART 8 DETERMINING THE NUMBER OF AR AND MA TERMS FOR SEASONAL AND STATIONARY COMPONENTS ##########
##sin componentes estacionales ni estacionarios
ggAcf(residuals(Arima(BXCXB116day,order=c(0,1,0),seasonal=c(0,1,0),
                      lambda=BXCXLAMB116day,include.drift=F,include.constant=T,
                      include.mean = T)),lag.max = 168) #grafico de autocorrelacion, contar las lineas que se salen de la franja azul
### resultado q=2, que son los primeros de la gráfica y Q=2 que se cuenta desde el 24 incluido
qB116day<-1
QB116day<-2

ggAcf(residuals(Arima(BXCXB123day,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB123day,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50))

qB123day<-4
QB123day<-1

ggAcf(residuals(Arima(BXCXB124day,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB124day,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50))

qB124day<-1
QB124day<-1

ggAcf(residuals(Arima(BXCXB011day,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB011day,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50))

qB011day<-4
QB011day<-3

ggAcf(residuals(Arima(BXCXB002day,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB002day,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) #grafico de autocorrelación, contar las líneas que se salen de la franja azul
### resultado q=4 y del Q=2
qB002day<-4
QB002day<-2



ggPacf(residuals(Arima(BXCXB116day,order=c(0,0,0),seasonal=c(0,0,0),
                       lambda=BXCXLAMB116day)),lag.max=168)#grafico de autocorrelacion parcial, contar las lineas que se salen de la franja azul
### resultado p=2, que son los que se toman desde el comienzo de la gráfica y P=3 que se toman desde el 24 incluido
pB116day<-2
PB116day<-4

ggPacf(residuals(Arima(BXCXB123day,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB123day)),lag.max = 50)

pB123day<-8
PB123day<-2

ggPacf(residuals(Arima(BXCXB124day,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB124day)),lag.max = 50)

pB124day<-8
PB124day<-3

ggPacf(residuals(Arima(BXCXB002day,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB002day)),lag.max = 50)

pB002day<-6
PB002day<-5

ggPacf(residuals(Arima(BXCXB011day,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB011day)),lag.max = 50)

pB011day<-6
PB011day<-2


##by weeks
# 
ggAcf(residuals(Arima(BXCXB116week,order=c(0,1,0),seasonal=c(0,1,0),
                      lambda=BXCXLAMB116week,include.drift=F,include.constant=T,
                      include.mean=T)),lag.max=600)
### resultado q=2, que son los primeros de la gráfica y Q=2 que se cuenta desde el 24 incluido
qB116week<-7
QB116week<-1
# 
# ggAcf(residuals(Arima(BXCXB123week,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB123week,include.drift = F,include.constant = T, include.mean = T)),lag.max = (180))
# 
# qB123week<-7
# QB123week<-3
# 
# ggAcf(residuals(Arima(BXCXB124week,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB124week,include.drift = F,include.constant = T, include.mean = T)),lag.max = (180))
# 
# qB124week<-7
# QB124week<-5
# 
# ggAcf(residuals(Arima(BXCXB011week,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB011week,include.drift = F,include.constant = T, include.mean = T)),lag.max = (180))
# 
# qB011week<-3
# QB011week<-1
# 
# ggAcf(residuals(Arima(BXCXB002week,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB002week,include.drift = F,include.constant = T, include.mean = T)),lag.max = (180))
# ### resultado q=4 y del Q=2
# qB002week<-8
# QB002week<-4
# 
# 
# 
ggPacf(residuals(Arima(BXCXB116week,order=c(0,1,0),seasonal=c(0,1,0),
                       lambda=BXCXLAMB116week)),lag.max = 180)
### resultado p=2, que son los que se toman desde el comienzo de la grafica y P=3 que se toman desde el 24 incluido
pB116week<-8
PB116week<-3
# 
# ggPacf(residuals(Arima(BXCXB123week,order = c(0,1,0), seasonal = c(0,1,0),lambda = BXCXLAMB123week)),lag.max = 180)
# 
# pB123week<-2
# PB123week<-3
# 
# ggPacf(residuals(Arima(BXCXB124week,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB124week)),lag.max = 180)
# 
# pB124week<-12
# PB124week<-2
# 
# ggPacf(residuals(Arima(BXCXB002week,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB002week)),lag.max = 180)
# 
# pB002week<-6
# PB002week<-6
# 
# ggPacf(residuals(Arima(BXCXB011week,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB011week)),lag.max = 180)
# 
# pB011week<-6
# PB011week<-5

#ggAcf(residuals(Arima(BXCXB116),lambda=BXCXLAMB116),lag.max = 50) #grafico de autocorrelación con lambda calculado del boxcox
#ggAcf(residuals(auto.arima(BXCXB116,lambda=BXCXLAMB116)),lag.max = 50) #grafico de autocorrelación con el autoarima
#ggPacf(residuals(Arima(BXCXB116,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB116)),lag.max = 50) #gráfico de autocorrelación parcial

##sin componente estacional con componente estacionario
#ggAcf(residuals(Arima(BXCXB116,order=c(0,1,0),seasonal = c(0,0,0),lambda = BXCXLAMB116,include.drift = T,include.constant = T, include.mean = T)),lag.max = 50) #grafico de autocorrelación
#ggAcf(residuals(auto.arima(BXCXB116,lambda=BXCXLAMB116)),lag.max = 50) #grafico de autocorrelación con el autoarima




##con componentes estacional y estacionarios
#ggAcf(residuals(Arima(BXCXB116,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB116,include.drift = T,include.constant = T, include.mean = T)),lag.max = 50) #grafico de autocorrelación
#ggAcf(residuals(auto.arima(BXCXB116,lambda=BXCXLAMB116)),lag.max = 50) #grafico de autocorrelación con el autoarima
#ggPacf(residuals(Arima(BXCXB116,order = c(0,1,0), seasonal = c(0,1,0),lambda = BXCXLAMB116)),lag.max = 50)

##con componente estacional sin componente estacionario
#ggAcf(residuals(Arima(BXCXB116,order=c(0,0,0),seasonal = c(0,1,0),lambda = BXCXLAMB116,include.drift = T,include.constant = T, include.mean = T)),lag.max = 50) #grafico de autocorrelación
#ggAcf(residuals(auto.arima(BXCXB116,lambda=BXCXLAMB116)),lag.max = 50) #grafico de autocorrelación con el autoarima
#ggPacf(residuals(Arima(BXCXB116,order = c(0,0,0), seasonal = c(0,1,0),lambda = BXCXLAMB116)),lag.max = 50)

#acf(residuals(Arima(BXCXB116,order = c(0,0,0), seasonal = c(0,1,0),lambda = BXCXLAMB116)),lag.max = 50)

##pruebas dickey fuller con diferentes retrasos

#summary(ur.df(BXCXB116,lags=24,type="none"))
#summary(ur.df(BXCXB116,lags=(24*7),type="none"))

#auto.arima(B116.ts,d=0,D=0,start.p = 3,start.q = 1,start.P = 3,start.Q = 2, stationary=T ,seasonal = T,lambda = BXCXLAMB116, seasonal.test = "ocsb", allowdrift = T, allowmean = T)
##aqui el auto arima me dice cuales son los valores de p,d y q pero no se cuales quedan para P, D y Q


############ CHAP 1 PART 9 DICKEY FULLER TEST #####################
# summary(ur.df(diff(diff(BXCXB116day, lag = 1),lag = 24),lags=1,type=c("trend")))# 2 porque eso bota la grafica de parcial autocorr
# summary(ur.df(BXCXB116day,lags=8, type=c("trend"))) # del PACF con 000 en todo
# summary(ur.df(BXCXB116day,lags=8, type=c("none")))
# 
# summary(ur.df(diff(diff(BXCXB002day, lag = 1),lag = 24),lags=1,type=c("trend")))# 2 porque eso bota la grafica de parcial autocorr
# summary(ur.df(BXCXB002day,lags=8, type=c("trend"))) # del PACF con 000 en todo
# summary(ur.df(BXCXB002day,lags=8, type=c("none")))
# 
# summary(ur.df(diff(diff(BXCXB011day, lag = 1),lag = 24),lags=1,type=c("trend")))
# summary(ur.df(BXCXB011day,lags=8, type=c("trend"))) # del PACF con 000 en todo
# summary(ur.df(BXCXB011day,lags=8, type=c("none")))
# 
# summary(ur.df(diff(diff(BXCXB123day, lag = 1),lag = 24),lags=1,type=c("trend")))
# summary(ur.df(BXCXB123day,lags=8, type=c("trend"))) # del PACF con 000 en todo
# summary(ur.df(BXCXB123day,lags=8, type=c("none")))
# 
# summary(ur.df(diff(diff(BXCXB124day, lag = 1),lag = 24),lags=1,type=c("trend")))# 2 porque eso bota la grafica de parcial autocorr
# summary(ur.df(BXCXB124day,lags=8, type=c("trend"))) # del PACF con 000 en todo
# summary(ur.df(BXCXB124day,lags=8, type=c("none")))



dev.off()
############ CHAP 1 PART 10 ARIMA (p,d,q)(P,D,Q) MODEL  ###########

# B116autoarima<-auto.arima(BXCXB116day,trace = T, test = "kpss", ic = "aic")
# summary(B116autoarima)
# confint(B116autoarima)
# plot.ts(B116autoarima$residuals)
# accuracy(forecast(B116autoarima, (24*7)),BXCXB116day2[4381:(4380+(24*7))])
# plot(forecast(B116autoarima, (24*7)))

B116fitday<-Arima(BXCXB116day,order=c(pB116day,dB116day,qB116day),
                  seasonal=c(PB116day,DB116day,QB116day),lambda=BXCXLAMB116day, 
                  include.drift=F,include.constant=F, include.mean=F,method="CSS")
summary(B116fitday)
confint(B116fitday)
plot.ts(B116fitday$residuals)
autoplot(B116fitday)
coeftest(B116fitday)
accuracy(forecast(B116fitday, (24*7)),BXCXB116day2[4381:(4380+(24*7))])
plot(forecast(B116fitday, (24*7)))


B002fitday<-Arima(BXCXB002day, order = c(pB002day,dB002day,qB002day), seasonal=c(PB002day,DB002day,QB002day),lambda = BXCXLAMB002day, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B002fitday)
autoplot(B002fitday)
coeftest(B002fitday)
accuracy(forecast(B002fitday, (24*7)),B002.ts2day[4381:(4380+(24*7))])

B123fitday<-Arima(BXCXB123day, order = c(pB123day,dB123day,qB123day), seasonal=c(PB123day,DB123day,QB123day),lambda = BXCXLAMB123day, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B123fitday)
autoplot(B123fitday)
coeftest(B123fitday)
accuracy(forecast(B123fitday, (24*7)),B123.ts2day[4381:(4380+(24*7))])

B124fitday<-Arima(BXCXB124day, order = c(pB124day,dB124day,qB124day), seasonal=c(PB124day,DB124day,QB124day),lambda = BXCXLAMB124day, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B124fitday)
autoplot(B124fitday)
coeftest(B124fitday)
accuracy(forecast(B124fitday, (24*7)),B124.ts2day[4381:(4380+(24*7))])

B011fitday<-Arima(BXCXB011day, order = c(pB011day,dB011day,qB011day), seasonal=c(PB011day,DB011day,QB011day),lambda = BXCXLAMB011day, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B011fitday)
autoplot(B011fitday)
coeftest(B011fitday)
accuracy(forecast(B011fitday, (24*7)),B011.ts2day[4381:(4380+(24*7))])



B116fitweek<-Arima(BXCXB116week,order=c(pB116week,dB116week,qB116week),
                   seasonal=c(PB116week,DB116week,QB116week),lambda=BXCXLAMB116week,
                   include.drift=F,include.constant=F,include.mean=F, method="CSS")
# summary(B116fitweek)
# autoplot(B116fitweek)
# coeftest(B116fitweek)
# accuracy(forecast(B116fitweek, (24*7)),B116.ts2week[4380:(4380+(24*7))])
# 
# B002fitweek<-Arima(BXCXB002week, order = c(pB002week,dB002week,qB002week), seasonal=c(PB002week,DB002week,QB002week),lambda = BXCXLAMB002week, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
# summary(B002fitweek)
# autoplot(B002fitweek)
# coeftest(B002fitweek)
# accuracy(forecast(B002fitweek, (24*7)),B002.ts2week[4380:(4380+(24*7))])
# 
# B123fitweek<-Arima(BXCXB123week, order = c(pB123week,dB123week,qB123week), seasonal=c(PB123week,DB123week,QB123week),lambda = BXCXLAMB123week, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
# summary(B123fitweek)
# autoplot(B123fitweek)
# coeftest(B123fitweek)
# accuracy(forecast(B123fitweek, (24*7)),B123.ts2week[4380:(4380+(24*7))])
# 
# B124fitweek<-Arima(BXCXB124week, order = c(pB124week,dB124week,qB124week), seasonal=c(PB124week,DB124week,QB124week),lambda = BXCXLAMB124week, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
# summary(B124fitweek)
# autoplot(B124fitweek)
# coeftest(B124fitweek)
# accuracy(forecast(B124fitweek, (24*7)),B124.ts2week[4380:(4380+(24*7))])
# 
# B011fitweek<-Arima(BXCXB011week, order = c(pB011week,dB011week,qB011week), seasonal=c(PB011week,DB011week,QB011week),lambda = BXCXLAMB011week, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
# summary(B011fitweek)
# autoplot(B011fitweek)
# coeftest(B011fitweek)
# accuracy(forecast(B011fitweek, (24*7)),B011.ts2week[4380:(4380+(24*7))])

########### CHAP 1 PART 11A FORECAST WITH STLF FUNCTION ##############
B116stlfweek<-stlf(B116.tsweek)
#B116stlfday<-stlf(B116.tsday)
plot(B116stlfweek)
#plot(B116stlfday)
accuracy(B116stlfweek,B116.ts2week[4393:(4392+336)])

B002stlfweek<-stlf(B002.tsweek)
#B002stlfday<-stlf(B002.tsday)
plot(B002stlfweek)
#plot(B002stlfday)
accuracy(B002stlfweek,B002.ts2week[4393:(4392+336)])

B011stlfweek<-stlf(B011.tsweek)
#B011stlfday<-stlf(B011.tsday)
plot(B011stlfweek)
#plot(B011stlfday)
accuracy(B011stlfweek,B011.ts2week[4393:(4392+336)])

B123stlfweek<-stlf(B123.tsweek)
#B123stlfday<-stlf(B123.tsday)
plot(B123stlfweek)
#plot(B123stlfday)
accuracy(B123stlfweek,B123.ts2week[4393:(4392+336)])

B124stlfweek<-stlf(B124.tsweek)
#B124stlfday<-stlf(B124.tsday)
plot(B124stlfweek)
#plot(B124stlfday)
accuracy(B124stlfweek,B124.ts2week[4393:(4392+336)])


############ CHAP 1 PART 11B FORECAST PLOT WITH ARIMA MODEL ####################
B116.foreday<-forecast(B116fitday, (24*7*4))
dev.off()
jpeg(filename = "B116_ARIMA_forecast_wholedata.jpg", height = 786, width = 1048)
par(mfrow=c(2,1))
plot(B116.foreday, main="Forecast for B116, whole data by days")
plot(B116.foreday,xlim=c(170,200),main="Zoom of forecast for B116, whole data by days")
dev.off()

jpeg(filename = "B116_ARIMA_forecast_comparison_whole_data", height = 786, width = 1048)
plot(1:672,InvBoxCox(B116.foreday$mean, lambda = BXCXLAMB116day),"l",ylim=c(min(B116.ts2day[4393:5064]),max(B116.ts2day[4393:5064])+20),main="B116 forecast vs. real data",ylab="KWh")
par(new=T)
plot(1:672,InvBoxCox(BXCXB116day2,lambda = BXCXLAMB116day)[4393:(4392+24*7*4)],"l",col="red",ylim=c(min(B116.ts2day[4393:5064]),max(B116.ts2day[4393:5064])+20),ylab="KWh")
legend(0,110,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("black","red"))
dev.off()
B116.errorwhole<-accuracy(InvBoxCox(B116.foreday$mean, lambda = BXCXLAMB116day),InvBoxCox(BXCXB116day2,lambda = BXCXLAMB116day)[4393:(4392+24*7*4)])
B116.errorwhole<-matrix(B116.errorwhole)

B123.foreday<-forecast(B123fitday, (24*7*4))
jpeg(filename = "B123_ARIMA_forecast_wholedata.jpg", height = 786, width = 1048)
par(mfrow=c(2,1))
plot(forecast(B123fitday, (24*7*4)), main="Forecast for B123, whole data by days")
plot(forecast(B123fitday, (24*7*4)),xlim=c(170,200),main="Zoom of forecast for B123, whole data by days")
dev.off()

jpeg(filename = "B123_ARIMA_forecast_comparison_whole_data", height = 786, width = 1048)
plot(1:672,InvBoxCox(B123.foreday$mean, lambda = BXCXLAMB123day),"l",ylim=c(min(B123.ts2day[4393:5064]),max(B123.ts2day[4393:5064])+20),main="B123 forecast vs. real data",ylab="KWh")
par(new=T)
plot(1:672,InvBoxCox(BXCXB123day2,lambda = BXCXLAMB123day)[4393:(4392+24*7*4)],"l",col="red",ylim=c(min(B123.ts2day[4393:5064]),max(B123.ts2day[4393:5064])+20),ylab="KWh")
legend(0,150,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("black","red"))
dev.off()
B123.errorwhole<-accuracy(InvBoxCox(B123.foreday$mean, lambda = BXCXLAMB123day),InvBoxCox(BXCXB123day2,lambda = BXCXLAMB123day)[4393:(4392+24*7*4)])
B123.errorwhole<-matrix(B123.errorwhole)

B124.foreday<-forecast(B124fitday, (24*7*4))
jpeg(filename = "B124_ARIMA_forecast_wholedata.jpg", height = 786, width = 1048)
par(mfrow=c(2,1))
plot(forecast(B124fitday, (24*7*4)), main="Forecast for B124, whole data by days")
plot(forecast(B124fitday, (24*7*4)),xlim=c(170,200),main="Zoom of forecast for B124, whole data by days")
dev.off()

jpeg(filename = "B124_ARIMA_forecast_comparison_whole_data", height = 786, width = 1048)
plot(1:672,InvBoxCox(B124.foreday$mean, lambda = BXCXLAMB124day),"l",ylim=c(min(B124.ts2day[4393:5064]),max(B124.ts2day[4393:5064])+5),main="B124 forecast vs. real data",ylab="KWh")
par(new=T)
plot(1:672,InvBoxCox(BXCXB124day2,lambda = BXCXLAMB124day)[4393:(4392+24*7*4)],"l",col="red",ylim=c(min(B124.ts2day[4393:5064]),max(B124.ts2day[4393:5064])+5),ylab="KWh")
legend(0,35,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("black","red"))
dev.off()
B124.errorwhole<-accuracy(InvBoxCox(B124.foreday$mean, lambda = BXCXLAMB124day),InvBoxCox(BXCXB124day2,lambda = BXCXLAMB124day)[4393:(4392+24*7*4)])
B124.errorwhole<-matrix(B124.errorwhole)

B002.foreday<-forecast(B002fitday, (24*7*4))
jpeg(filename = "B002_ARIMA_forecast_wholedata.jpg", height = 786, width = 1048)
par(mfrow=c(2,1))
plot(forecast(B002fitday, (24*7*4)), main="Forecast for B002, whole data by days")
plot(forecast(B002fitday, (24*7*4)),xlim=c(170,200),main="Zoom of forecast for B002, whole data by days")
dev.off()

jpeg(filename = "B002_ARIMA_forecast_comparison_whole_data", height = 786, width = 1048)
plot(1:672,InvBoxCox(B002.foreday$mean, lambda = BXCXLAMB002day),"l",ylim=c(min(B002.ts2day[4393:5064])-10,max(B002.ts2day[4393:5064])+50),main="B002 forecast vs. real data",ylab="KWh")
par(new=T)
plot(1:672,InvBoxCox(BXCXB002day2,lambda = BXCXLAMB002day)[4393:(4392+24*7*4)],"l",col="red",ylim=c(min(B002.ts2day[4393:5064])-10,max(B002.ts2day[4393:5064])+50),ylab="KWh")
legend(50,200,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("black","red"))
dev.off()
B002.errorwhole<-accuracy(InvBoxCox(B002.foreday$mean, lambda = BXCXLAMB002day),InvBoxCox(BXCXB002day2,lambda = BXCXLAMB002day)[4393:(4392+24*7*4)])
B002.errorwhole<-matrix(B002.errorwhole)


B011.foreday<-forecast(B011fitday, (24*7*4))
jpeg(filename = "B011_ARIMA_forecast_wholedata.jpg", height = 786, width = 1048)
par(mfrow=c(2,1))
plot(forecast(B011fitday, (24*7*4)), main="Forecast for B011, whole data by days")
plot(forecast(B011fitday, (24*7*4)),xlim=c(170,200),main="Zoom of forecast for B011, whole data by days")
dev.off()

jpeg(filename = "B011_ARIMA_forecast_comparison_whole_data", height = 786, width = 1048)
plot(1:672,InvBoxCox(B011.foreday$mean, lambda = BXCXLAMB011day),"l",ylim=c(min(B011.ts2day[4393:5064]),max(B011.ts2day[4393:5064])+2),main="B011 forecast vs. real data",ylab="KWh")
par(new=T)
plot(1:672,InvBoxCox(BXCXB011day2,lambda = BXCXLAMB011day)[4393:(4392+24*7*4)],"l",col="red",ylim=c(min(B011.ts2day[4393:5064]),max(B011.ts2day[4393:5064])+2),ylab="KWh")
legend(0,14,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("black","red"))
dev.off()
B011.errorwhole<-accuracy(InvBoxCox(B011.foreday$mean, lambda = BXCXLAMB011day),InvBoxCox(BXCXB011day2,lambda = BXCXLAMB011day)[4393:(4392+24*7*4)])
B011.errorwhole<-matrix(B011.errorwhole)

############ CHAP 1 PART 12 FORECAST WITH FOURIER TERMS ####################

# B116fou<-fourier(B116.tsdia, K=10)
# B116autoarimafou<-auto.arima(B116.tsdia,xreg = fourier(B116.tsdia, K=10), lambda=0)
# graphfouB116<-plot(forecast(B116autoarimafou, xreg = fourier(B116.tsdia, K=10, h=(24*7*4)), lambda = 0, ))
# ###no toma fines de semana
# 
# B002fou<-fourier(B002.tsdia, K=10)
# B002autoarimafou<-auto.arima(B002.tsdia,xreg = fourier(B002.tsdia, K=10), lambda=0)
# graphfouB002<-plot(forecast(B002autoarimafou, xreg = fourier(B002.tsdia, K=10, h=(24*7*2)), lambda = 0, ))
# ###no toma domingos
# 
# ############ PART 15 
# 
# 
# 
# tsdisplay(residuals(Arima(BXCXB116dia,order = c(0,0,0), seasonal = list(order=c(0,0,0),period=24),lambda = BXCXLAMB116dia)), lag.max = 50)
# tsdisplay(residuals(Arima(BXCXB002dia,order = c(0,0,0), seasonal = list(order=c(0,0,0),period=24),lambda = BXCXLAMB002dia)), lag.max = 50)
# 
# 
# B116dummy<-seasonaldummy(B116.tsdia)
# B116autoarima<-auto.arima(B116.tsdia,xreg = B116dummy)
# plot(forecast(B116autoarima, xreg = B116dummy, lambda = BXCXLAMB116dia))
# 
# B116fourier<-fourier(B116.tsdia, K=10)
# B116autoarimafourier<-auto.arima(B116.tsdia,xreg = fourier(B116.tsdia, K=10), lambda=0)
# graphfourierB116<-plot(forecast(B116autoarimafourier, xreg = fourier(B116.tsdia,K=10 , h=24), lambda = 0),xlim=c(240,260))
# 



###########haciendo los terminos de fourier separando días entre semana y fines de semana
# plot(1:(24*5),B116$Ef_kWh[1:(24*5)],"l")
# V1<-c((1:365)*24)
# V2<-rep(c(4,5,6,7,1,2,3), len=365)
# V3<-c(1:366)
# ttdays<-data.frame(V1,V2)
# ttdays<-rbind(ttdays,c(0,5))
# ttdays<-cbind(ttdays,V3)
# ttdays<-rbind(ttdays,c(0,5,0,0))
# Kwh2<-rep(0,8760)

# for(i in 1:364){
#   if(i==ttdays$V3[ttdays$V2==6 | ttdays$V2==7][a]){
# 	for(j in 1:24){
#   	B116wkend<-rbind(B116wkend,B116$Ef_kWh[j*i])}
#   	a<-a+1}
# 	else{
#   	for(j in 1:24){
#     	B116wkdays<-rbind(B116wkdays,B116$Ef_kWh[j*i])}
#   }
# }

############ CHAP 2 PART 1 SPLITTING THE DATA INTO WEEKDAYS AND WEEEKENDS #############

####B116
inddata<-c(1:(8760))
daychange<-inddata%%24
daysyear<-rep(0,(8760))
daysweek<-rep(0,(8760))
ddayss<-data.frame(B116$Ef_kWh[1:(8760)],inddata,daychange,daysyear,daysweek)
e<-1
daysname<-rep(c(4,5,6,7,1,2,3),len=366)
for(i in 1:(8760)){
  if(ddayss$daychange[i]==0){
    e<-e+1
    ddayss[i,4]<-e
    ddayss[i,5]<-daysname[e]
  }
  else{
    ddayss[i,4]<-e
    ddayss[i,5]<-daysname[e]
  }
}


Kwh<-0
B116wkdays<-data.frame(Kwh)
B116wkend<-data.frame(Kwh)

for(i in 1:(8760)){
  if(ddayss$daysweek[i]==6 | ddayss$daysweek[i]==7){
    B116wkend<-rbind(B116wkend,ddayss$B116.Ef_kWh[i])
  }
  else{
    B116wkdays<-rbind(B116wkdays,ddayss$B116.Ef_kWh[i])
  }
}

KwhwkdaysB116<-B116wkdays[-1,]
KwhwkendB116<-B116wkend[-1,]
B116wkdays<-as.data.frame(KwhwkdaysB116)
B116wkend<-as.data.frame(KwhwkendB116)

####B002  this building has high consumption on saturday, so sunday will be split
ddayss002<-data.frame(B002$Ef_kWh,inddata,daychange,daysyear,daysweek)
e<-1
for(i in 1:(8760)){
  if(ddayss002$daychange[i]==0){
    e<-e+1
    ddayss002[i,4]<-e
    ddayss002[i,5]<-daysname[e]
  }
  else{
    ddayss002[i,4]<-e
    ddayss002[i,5]<-daysname[e]
  }
}


Kwh<-0
B002wkdays<-data.frame(Kwh)
B002wkend<-data.frame(Kwh)

for(i in 1:(8760)){
  if(ddayss002$daysweek[i]==7){
    B002wkend<-rbind(B002wkend,ddayss002$B002.Ef_kWh[i])
  }
  else{
    B002wkdays<-rbind(B002wkdays,ddayss002$B002.Ef_kWh[i])
  }
}

KwhwkdaysB002<-B002wkdays[-1,]
KwhwkendB002<-B002wkend[-1,]
B002wkdays<-as.data.frame(KwhwkdaysB002)
B002wkend<-as.data.frame(KwhwkendB002)



ddayss011<-data.frame(B011$Ef_kWh,inddata,daychange,daysyear,daysweek)
e<-1
for(i in 1:(8760)){
  if(ddayss011$daychange[i]==0){
    e<-e+1
    ddayss011[i,4]<-e
    ddayss011[i,5]<-daysname[e]
  }
  else{
    ddayss011[i,4]<-e
    ddayss011[i,5]<-daysname[e]
  }
}


Kwh<-0
B011wkdays<-data.frame(Kwh)
B011wkend<-data.frame(Kwh)

for(i in 1:(8760)){
  if(ddayss011$daysweek[i]==7 | ddayss011$daysweek[i]==6){
    B011wkend<-rbind(B011wkend,ddayss011$B011.Ef_kWh[i])
  }
  else{
    B011wkdays<-rbind(B011wkdays,ddayss011$B011.Ef_kWh[i])
  }
}

KwhwkdaysB011<-B011wkdays[-1,]
KwhwkendB011<-B011wkend[-1,]
B011wkdays<-as.data.frame(KwhwkdaysB011)
B011wkend<-as.data.frame(KwhwkendB011)


ddayss123<-data.frame(B123$Ef_kWh,inddata,daychange,daysyear,daysweek)
e<-1
for(i in 1:(8760)){
  if(ddayss123$daychange[i]==0){
    e<-e+1
    ddayss123[i,4]<-e
    ddayss123[i,5]<-daysname[e]
  }
  else{
    ddayss123[i,4]<-e
    ddayss123[i,5]<-daysname[e]
  }
}


Kwh<-0
B123wkdays<-data.frame(Kwh)
B123wkend<-data.frame(Kwh)

for(i in 1:(8760)){
  if(ddayss123$daysweek[i]==7 | ddayss123$daysweek[i]==6){
    B123wkend<-rbind(B123wkend,ddayss123$B123.Ef_kWh[i])
  }
  else{
    B123wkdays<-rbind(B123wkdays,ddayss123$B123.Ef_kWh[i])
  }
}

KwhwkdaysB123<-B123wkdays[-1,]
KwhwkendB123<-B123wkend[-1,]
B123wkdays<-as.data.frame(KwhwkdaysB123)
B123wkend<-as.data.frame(KwhwkendB123) 

ddayss124<-data.frame(B124$Ef_kWh,inddata,daychange,daysyear,daysweek)
e<-1
for(i in 1:(8760)){
  if(ddayss124$daychange[i]==0){
    e<-e+1
    ddayss124[i,4]<-e
    ddayss124[i,5]<-daysname[e]
  }
  else{
    ddayss124[i,4]<-e
    ddayss124[i,5]<-daysname[e]
  }
}


Kwh<-0
B124wkdays<-data.frame(Kwh)
B124wkend<-data.frame(Kwh)

for(i in 1:(8760)){
  if(ddayss124$daysweek[i]==7 | ddayss123$daysweek[i]==6){
    B124wkend<-rbind(B124wkend,ddayss124$B124.Ef_kWh[i])
  }
  else{
    B124wkdays<-rbind(B124wkdays,ddayss124$B124.Ef_kWh[i])
  }
}

KwhwkdaysB124<-B124wkdays[-1,]
KwhwkendB124<-B124wkend[-1,]
B124wkdays<-as.data.frame(KwhwkdaysB124)
B124wkend<-as.data.frame(KwhwkendB124)





############ CHAP 2 PART 2 TRANSFORMING DIVIDED DATA INTO TIME SERIES #############

B116ts.wkdays<-ts(B116wkdays$KwhwkdaysB116[1:3120], frequency=24)
B116ts.wkend<-ts(B116wkend$KwhwkendB116[1:1248], frequency=24)
B116ts.wkdays2<-ts(B116wkdays$KwhwkdaysB116, frequency=24)
B116ts.wkend2<-ts(B116wkend$KwhwkendB116, frequency=24)
seasonplot(B116ts.wkdays)
seasonplot(B116ts.wkdays2)
seasonplot(B116ts.wkend)
seasonplot(B116ts.wkend2)


B002ts.wkdays<-ts(B002wkdays$KwhwkdaysB002[1:3768], frequency=24)
B002ts.wkend<-ts(B002wkend$KwhwkendB002[1:624], frequency=24)
B002ts.wkdays2<-ts(B002wkdays$KwhwkdaysB002, frequency=24)
B002ts.wkend2<-ts(B002wkend$KwhwkendB002, frequency=24)
seasonplot(B002ts.wkdays)
seasonplot(B002ts.wkdays2)
#plot(B116ts.wkdays)
seasonplot(B002ts.wkend)
seasonplot(B002ts.wkend2)
#plot(B002ts.wkend)


B011ts.wkdays<-ts(B011wkdays$KwhwkdaysB011[1:3120], frequency=24)
B011ts.wkend<-ts(B011wkend$KwhwkendB011[1:1248], frequency=24)
B011ts.wkdays2<-ts(B011wkdays$KwhwkdaysB011, frequency=24)
B011ts.wkend2<-ts(B011wkend$KwhwkendB011, frequency=24)
seasonplot(B011ts.wkdays)
seasonplot(B011ts.wkdays2)
seasonplot(B011ts.wkend)
seasonplot(B011ts.wkend2)


B123ts.wkdays<-ts(B123wkdays$KwhwkdaysB123[1:3120], frequency=24)
B123ts.wkend<-ts(B123wkend$KwhwkendB123[1:1248], frequency=24)
B123ts.wkdays2<-ts(B123wkdays$KwhwkdaysB123, frequency=24)
B123ts.wkend2<-ts(B123wkend$KwhwkendB123, frequency=24)
seasonplot(B123ts.wkdays)
seasonplot(B123ts.wkdays2)
seasonplot(B123ts.wkend)
seasonplot(B123ts.wkend2)

B124ts.wkdays<-ts(B124wkdays$KwhwkdaysB124[1:3120], frequency=24)
B124ts.wkend<-ts(B124wkend$KwhwkendB124[1:1248], frequency=24)
B124ts.wkdays2<-ts(B124wkdays$KwhwkdaysB124, frequency=24)
B124ts.wkend2<-ts(B124wkend$KwhwkendB124, frequency=24)
seasonplot(B124ts.wkdays)
seasonplot(B124ts.wkdays2)
seasonplot(B124ts.wkend)
seasonplot(B124ts.wkend2)



############ CHAP 2 PART 3 SPECTRAL PERIODOGRAM FOR THE SEPARATED TIME SERIES ##################

B116ts.wkdayses<-spec.pgram(B116ts.wkdays, log="no")
B116ts.wkendes<-spec.pgram(B116ts.wkend, log="no")
B116ts.wkdayses2<-spec.pgram(B116ts.wkdays2, log="no")
B116ts.wkendes2<-spec.pgram(B116ts.wkend2, log="no")

B002ts.wkdayses<-spec.pgram(B002ts.wkdays, log="no")
B002ts.wkendes<-spec.pgram(B002ts.wkend, log="no")
B002ts.wkdayses2<-spec.pgram(B002ts.wkdays2, log="no")
B002ts.wkendes2<-spec.pgram(B002ts.wkend2, log="no")


B011ts.wkdayses<-spec.pgram(B011ts.wkdays, log="no")
B011ts.wkendes<-spec.pgram(B011ts.wkend, log="no")
B011ts.wkdayses2<-spec.pgram(B011ts.wkdays2, log="no")
B011ts.wkendes2<-spec.pgram(B011ts.wkend2, log="no")

B123ts.wkdayses<-spec.pgram(B123ts.wkdays, log="no")
B123ts.wkendes<-spec.pgram(B123ts.wkend, log="no")
B123ts.wkdayses2<-spec.pgram(B123ts.wkdays2, log="no")
B123ts.wkendes2<-spec.pgram(B123ts.wkend2, log="no")

B124ts.wkdayses<-spec.pgram(B124ts.wkdays, log="no")
B124ts.wkendes<-spec.pgram(B124ts.wkend, log="no")
B124ts.wkdayses2<-spec.pgram(B124ts.wkdays2, log="no")
B124ts.wkendes2<-spec.pgram(B124ts.wkend2, log="no")

############ CHAP 2 PART 4 MAIN FREQUENCIES FOR SEPARETED TIME SERIES ##################

orderwkday116<-order(B116ts.wkdayses$spec,B116ts.wkdayses$freq, decreasing = T)
orderwkend116<-order(B116ts.wkendes$spec,B116ts.wkendes$freq, decreasing = T)

orderwkday002<-order(B002ts.wkdayses$spec,B002ts.wkdayses$freq, decreasing = T)
orderwkend002<-order(B002ts.wkendes$spec,B002ts.wkendes$freq, decreasing = T)

############ CHAP 2 PART 5 BOX COX TRANSFORMATION FOR SEPARETED TIME SERIES ##################

BXCXLAMB116wkday<-BoxCox.lambda(B116ts.wkdays, method = "loglik")
BXCXB116wkday<-BoxCox(B116ts.wkdays,BXCXLAMB116wkday)

BXCXLAMB116wkend<-BoxCox.lambda(B116ts.wkend, method = "loglik")
BXCXB116wkend<-BoxCox(B116ts.wkend,BXCXLAMB116wkend)

BXCXB116wkday2<-BoxCox(B116ts.wkdays2,BXCXLAMB116wkday)
BXCXB116wkend2<-BoxCox(B116ts.wkend2,BXCXLAMB116wkend)

BXCXLAMB011wkday<-BoxCox.lambda(B011ts.wkdays, method = "loglik")
BXCXB011wkday<-BoxCox(B011ts.wkdays,BXCXLAMB011wkday)

BXCXLAMB011wkend<-BoxCox.lambda(B011ts.wkend, method = "loglik")
BXCXB011wkend<-BoxCox(B011ts.wkend,BXCXLAMB011wkend)

BXCXLAMB011wkday2<-BoxCox.lambda(B011ts.wkdays2, method = "loglik")
BXCXB011wkday2<-BoxCox(B011ts.wkdays2,BXCXLAMB011wkday2)

BXCXLAMB011wkend2<-BoxCox.lambda(B011ts.wkend2, method = "loglik")
BXCXB011wkend2<-BoxCox(B011ts.wkend2,BXCXLAMB011wkend2) 

BXCXLAMB123wkday<-BoxCox.lambda(B123ts.wkdays, method = "loglik")
BXCXB123wkday<-BoxCox(B123ts.wkdays,BXCXLAMB123wkday)

BXCXLAMB123wkend<-BoxCox.lambda(B123ts.wkend, method = "loglik")
BXCXB123wkend<-BoxCox(B123ts.wkend,BXCXLAMB123wkend)

BXCXLAMB123wkday2<-BoxCox.lambda(B123ts.wkdays2, method = "loglik")
BXCXB123wkday2<-BoxCox(B123ts.wkdays2,BXCXLAMB123wkday2)

BXCXLAMB123wkend2<-BoxCox.lambda(B123ts.wkend2, method = "loglik")
BXCXB123wkend2<-BoxCox(B123ts.wkend2,BXCXLAMB123wkend2) 

BXCXLAMB124wkday<-BoxCox.lambda(B124ts.wkdays, method = "loglik")
BXCXB124wkday<-BoxCox(B124ts.wkdays,BXCXLAMB124wkday)

BXCXLAMB124wkend<-BoxCox.lambda(B124ts.wkend, method = "loglik")
BXCXB124wkend<-BoxCox(B124ts.wkend,BXCXLAMB124wkend)

BXCXLAMB124wkday2<-BoxCox.lambda(B124ts.wkdays2, method = "loglik")
BXCXB124wkday2<-BoxCox(B124ts.wkdays2,BXCXLAMB124wkday2)

BXCXLAMB124wkend2<-BoxCox.lambda(B124ts.wkend2, method = "loglik")
BXCXB124wkend2<-BoxCox(B124ts.wkend2,BXCXLAMB124wkend2)

BXCXLAMB002wkday<-BoxCox.lambda(B002ts.wkdays, method = "loglik")
BXCXB002wkday<-BoxCox(B002ts.wkdays,BXCXLAMB002wkday)

BXCXLAMB002wkend<-BoxCox.lambda(B002ts.wkend, method = "loglik")
BXCXB002wkend<-BoxCox(B002ts.wkend,BXCXLAMB002wkend)

BXCXLAMB002wkday2<-BoxCox.lambda(B002ts.wkdays2, method = "loglik")
BXCXB002wkday2<-BoxCox(B002ts.wkdays2,BXCXLAMB002wkday2)

BXCXLAMB002wkend2<-BoxCox.lambda(B002ts.wkend2, method = "loglik")
BXCXB002wkend2<-BoxCox(B002ts.wkend2,BXCXLAMB002wkend2)

############ CHAP 2 PART 6 APPLYING DIFFERENCES TO THE SEPARETED TIME SERIES ##################

d116wkday<-ndiffs(BXCXB116wkday, alpha=0.05, test="kpss", max.d=2)
##d=1
d116wkend<-ndiffs(BXCXB116wkend, alpha=0.05, test="kpss", max.d=2)
##d=1
D116WKEND<-nsdiffs(BXCXB116wkend, m=frequency(BXCXB116wkend), test="ch", max.D=2)
##D=1
D116WKDAY<-nsdiffs(BXCXB116wkday, m=frequency(BXCXB116wkday), test="ch", max.D=2)
##D=1


d123wkday<-ndiffs(BXCXB123wkday, alpha=0.05, test="kpss", max.d=2)
##d=1
d123wkend<-ndiffs(BXCXB123wkend, alpha=0.05, test="kpss", max.d=2)
##d=1
D123WKEND<-nsdiffs(BXCXB123wkend, m=frequency(BXCXB123wkend), test="ch", max.D=2)
##D=1
D123WKDAY<-nsdiffs(BXCXB123wkday, m=frequency(BXCXB123wkday), test="ch", max.D=2)
##D=1

d124wkday<-ndiffs(BXCXB124wkday, alpha=0.05, test="kpss", max.d=2)
##d=1
d124wkend<-ndiffs(BXCXB124wkend, alpha=0.05, test="kpss", max.d=2)
##d=1
D124WKEND<-nsdiffs(BXCXB124wkend, m=frequency(BXCXB124wkend), test="ch", max.D=2)
##D=1
D124WKDAY<-nsdiffs(BXCXB124wkday, m=frequency(BXCXB124wkday), test="ch", max.D=2)
##D=1


d011wkday<-ndiffs(BXCXB011wkday, alpha=0.05, test="kpss", max.d=2)
##d=1
d011wkend<-ndiffs(BXCXB011wkend, alpha=0.05, test="kpss", max.d=2)
##d=1
D011WKEND<-nsdiffs(BXCXB011wkend, m=frequency(BXCXB011wkend), test="ch", max.D=2)
##D=1
D011WKDAY<-nsdiffs(BXCXB011wkday, m=frequency(BXCXB011wkday), test="ch", max.D=2)
##D=1 



d002wkday<-ndiffs(BXCXB002wkday, alpha=0.05, test="kpss", max.d=2)
##d=1
d002wkend<-ndiffs(BXCXB002wkend, alpha=0.05, test="kpss", max.d=2)
##d=1
D002WKDAY<-nsdiffs(BXCXB002wkday, m=frequency(BXCXB002wkday), test="ch", max.D=2)
##D=1
D002WKEND<-nsdiffs(BXCXB002wkend, m=frequency(BXCXB002wkend), test="ch", max.D=2)
##D=0




############ CHAP 2 PART 7 PLOTTING DIFFERENCED TRANSFORMED DATA ##################
autoplot(diff(diff(BXCXB116wkday, lag = 1),lag = 24),"l", main="B116 weekdays stationary and non seasonal time series")
autoplot(diff(diff(BXCXB116wkend, lag = 1),lag = 24),"l", main="B116 weekend stationary and non seasonal time series")

autoplot(diff(diff(BXCXB002wkday, lag = 1),lag = 24),"l", main="B002 weekdays stationary and non seasonal time series")
autoplot(diff(diff(BXCXB002wkend, lag = 1),lag = 24),"l", main="B002 weekend stationary and non seasonal time series")

############ CHAP 2 PART 8 DETERMINING THE NUMBER OF AR AND MA TERMS FOR SEASONAL AND STATIONARY COMPONENTS OF DIVDED TIME SERIES##########

ggAcf(residuals(Arima(BXCXB116wkday,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB116wkday,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=2  Q=3
q116wkday<-2
Q116WKDAY<-1
ggAcf(residuals(Arima(BXCXB116wkend,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB116wkend,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=1  Q=2
q116wkend<-1
Q116WKEND<-3

ggPacf(residuals(Arima(BXCXB116wkday,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB116wkday)),lag.max = 50)
#p=2 P=2
p116wkday<-2
P116WKDAY<-2
ggPacf(residuals(Arima(BXCXB116wkend,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB116wkend)),lag.max = 50)
#p=3 P=2
p116wkend<-4
P116WKEND<-2

ggAcf(residuals(Arima(BXCXB123wkday,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB123wkday,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=2  Q=3
q123wkday<-2
Q123WKDAY<-1
ggAcf(residuals(Arima(BXCXB123wkend,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB123wkend,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=1  Q=2
q123wkend<-2
Q123WKEND<-3

ggPacf(residuals(Arima(BXCXB123wkday,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB123wkday)),lag.max = 50)
#p=2 P=2
p123wkday<-7
P123WKDAY<-3
ggPacf(residuals(Arima(BXCXB123wkend,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB123wkend)),lag.max = 50)
#p=3 P=2
p123wkend<-2
P123WKEND<-1

ggAcf(residuals(Arima(BXCXB124wkday,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB124wkday,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=2  Q=3
q124wkday<-3
Q124WKDAY<-4
ggAcf(residuals(Arima(BXCXB124wkend,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB124wkend,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=1  Q=2
q124wkend<-2
Q124WKEND<-3

ggPacf(residuals(Arima(BXCXB124wkday,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB124wkday)),lag.max = 50)
#p=2 P=2
p124wkday<-10
P124WKDAY<-3
ggPacf(residuals(Arima(BXCXB124wkend,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB124wkend)),lag.max = 50)
#p=3 P=2
p124wkend<-10
P124WKEND<-6

ggAcf(residuals(Arima(BXCXB011wkday,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB011wkday,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=2  Q=3
q011wkday<-4
Q011WKDAY<-3
ggAcf(residuals(Arima(BXCXB011wkend,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB011wkend,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=1  Q=2
q011wkend<-2
Q011WKEND<-3

ggPacf(residuals(Arima(BXCXB011wkday,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB011wkday)),lag.max = 50)
#p=2 P=2
p011wkday<-5
P011WKDAY<-3
ggPacf(residuals(Arima(BXCXB011wkend,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB011wkend)),lag.max = 50)
#p=3 P=2
p011wkend<-3
P011WKEND<-3

ggAcf(residuals(Arima(BXCXB002wkday,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB002wkday,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=3  Q=4
q002wkday<-3
Q002WKDAY<-4
ggAcf(residuals(Arima(BXCXB002wkend,order=c(0,1,0),seasonal = c(0,1,0),lambda = BXCXLAMB002wkend,include.drift = F,include.constant = T, include.mean = T)),lag.max = (50)) 
#q=4  Q=1
q002wkend<-2
Q002WKEND<-2

ggPacf(residuals(Arima(BXCXB002wkday,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB002wkday)),lag.max = 50)
#p=7 P=5
p002wkday<-7
P002WKDAY<-5
ggPacf(residuals(Arima(BXCXB002wkend,order = c(0,0,0), seasonal = c(0,0,0),lambda = BXCXLAMB002wkend)),lag.max = 50)
#p=6 P=4
p002wkend<-6
P002WKEND<-2

############ CHAP 2 PART 9 ARIMA (p,d,q)(P,D,Q) MODEL FOR THE DIVIDED TRANSFORMED TIME SERIES  ###########

B116fitwkday<-Arima(BXCXB116wkday, order = c(p116wkday,d116wkday,q116wkday), seasonal=c(P116WKDAY,D116WKDAY,Q116WKDAY),lambda = BXCXLAMB116wkday, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B116fitwkday)
autoplot(B116fitwkday)
coeftest(B116fitwkday)
accuracy(forecast(B116fitwkday, (24)))

B116fitwkend<-Arima(BXCXB116wkend, order = c(p116wkend,d116wkend,q116wkend), seasonal=c(P116WKEND,D116WKEND,Q116WKEND),lambda = BXCXLAMB116wkend, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B116fitwkend)
autoplot(B116fitwkend)
coeftest(B116fitwkend)
accuracy(forecast(B116fitwkend, (24)))

B011fitwkday<-Arima(BXCXB011wkday, order = c(p011wkday,d011wkday,q011wkday), seasonal=c(P011WKDAY,D011WKDAY,Q011WKDAY),lambda = BXCXLAMB011wkday, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B011fitwkday)
autoplot(B011fitwkday)
coeftest(B011fitwkday)
accuracy(forecast(B011fitwkday, (24)))

B011fitwkend<-Arima(BXCXB011wkend, order = c(p011wkend,d011wkend,q011wkend), seasonal=c(P011WKEND,D011WKEND,Q011WKEND),lambda = BXCXLAMB011wkend, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B011fitwkend)
autoplot(B011fitwkend)
coeftest(B011fitwkend)
accuracy(forecast(B011fitwkend, (24)))

B123fitwkday<-Arima(BXCXB123wkday, order = c(p123wkday,d123wkday,q123wkday), seasonal=c(P123WKDAY,D123WKDAY,Q123WKDAY),lambda = BXCXLAMB123wkday, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B123fitwkday)
autoplot(B123fitwkday)
coeftest(B123fitwkday)
accuracy(forecast(B123fitwkday, (24)))

B123fitwkend<-Arima(BXCXB123wkend, order = c(p123wkend,d123wkend,q123wkend), seasonal=c(P123WKEND,D123WKEND,Q123WKEND),lambda = BXCXLAMB123wkend, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B123fitwkend)
autoplot(B123fitwkend)
coeftest(B123fitwkend)
accuracy(forecast(B123fitwkend, (24)))

B124fitwkday<-Arima(BXCXB124wkday, order = c(p124wkday,d124wkday,q124wkday), seasonal=c(P124WKDAY,D124WKDAY,Q124WKDAY),lambda = BXCXLAMB124wkday, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B124fitwkday)
autoplot(B124fitwkday)
coeftest(B124fitwkday)
accuracy(forecast(B124fitwkday, (24)))

B124fitwkend<-Arima(BXCXB124wkend, order = c(p124wkend,d124wkend,q124wkend), seasonal=c(P124WKEND,D124WKEND,Q124WKEND),lambda = BXCXLAMB124wkend, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B124fitwkend)
autoplot(B124fitwkend)
coeftest(B124fitwkend)
accuracy(forecast(B124fitwkend, (24)))




B002fitwkday<-Arima(BXCXB002wkday, order = c(p002wkday,d002wkday,q002wkday), seasonal=c(P002WKDAY,D002WKDAY,Q002WKDAY),lambda = BXCXLAMB002wkday, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B002fitwkday)
autoplot(B002fitwkday)
coeftest(B002fitwkday)
accuracy(forecast(B002fitwkday, (24)))

B002fitwkend<-Arima(BXCXB002wkend, order = c(p002wkend,d002wkend,q002wkend), seasonal=c(P002WKEND,D002WKEND,Q002WKEND),lambda = BXCXLAMB002wkend, include.drift = F,include.constant = F, include.mean = F, method = "CSS")
summary(B002fitwkend)
autoplot(B002fitwkend)
coeftest(B002fitwkend)
accuracy(forecast(B002fitwkend, (24)))

############ CHAP 2 PART 10 PLOTTING ARIMA FORECAST FOR THE DIVIDED TRANSFORMED TIME SERIES  ###########
foreB116wkday<-forecast(B116fitwkday, (24*10))

foreB002wkday<-forecast(B002fitwkday, (24*10))

foreB011wkday<-forecast(B011fitwkday, (24*10))

foreB123wkday<-forecast(B123fitwkday, (24*10))

foreB124wkday<-forecast(B124fitwkday, (24*10))

dev.off()
jpeg("B116_ARIMA_forecast_just_weekdays.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB116wkday,main=paste("B116 ARIMA(",p116wkday,",",d116wkday,",",q116wkday,")(",P116WKDAY,",",D116WKDAY,",",Q116WKDAY,") forecast for weekdays",sep=""))#,xlim=c(120,140), main=paste("B116 ARIMA(",p116wkday,",",d116wkday,",",q116wkday,")(",P116WKDAY,",",D116WKDAY,",",Q116WKDAY,") forecast for weekdays",sep=""))
plot(foreB116wkday,xlim=c(115,141), main=paste("B116 ARIMA(",p116wkday,",",d116wkday,",",q116wkday,")(",P116WKDAY,",",D116WKDAY,",",Q116WKDAY,") forecast for weekdays",sep=""))
dev.off()
jpeg("B116_ARIMA_forecast_comparison_weekdays.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB116wkday$mean,lambda = BXCXLAMB116wkday),"l",xlim=c(-5,250), ylim=c(25,100),col="red",xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B116ts.wkdays2[3121:(3120+240)],"l",xlim=c(-5,250), ylim=c(25,100), col="blue",xlab="Time", ylab="Kwh",main="B116 weekdays: forecast vs. real data")
legend(220,100,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("red","blue"))
dev.off()

B116.errorwkdays<-accuracy(InvBoxCox(foreB116wkday$mean,lambda = BXCXLAMB116wkday),B116ts.wkdays2[3121:(3120+240)])
B116.errorwkdays<-matrix(B116.errorwkdays)

# calcrmse<- function(actualval, forecastval){
#   rmse<-sqrt(sum((actualval-forecastval)^2)/length(actualval))
#   return(rmse)}
# 
# calcmape<- function(actualval, forecastval){
#   mape<-sum(abs(actualval-forecastval)/actualval)/length(actualval)
#   return(mape)}

# B116wkdaysRMSE<-calcrmse(InvBoxCox(foreB116wkday$mean,lambda = BXCXLAMB116wkday),B116ts.wkdays2[3121:(3120+240)])
# B116wkdaysMAPE<-calcmape(InvBoxCox(foreB116wkday$mean,lambda = BXCXLAMB116wkday),B116ts.wkdays2[3121:(3120+240)])


dev.off()
jpeg("B002_ARIMA_forecast_just_weekdays.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB002wkday,main=paste("B002 ARIMA(",p002wkday,",",d002wkday,",",q002wkday,")(",P002WKDAY,",",D002WKDAY,",",Q002WKDAY,") forecast for weekdays",sep=""))#,xlim=c(120,140), main=paste("B002 ARIMA(",p002wkday,",",d002wkday,",",q002wkday,")(",P002WKDAY,",",D002WKDAY,",",Q002WKDAY,") forecast for weekdays",sep=""))
plot(foreB002wkday,xlim=c(145,170), main=paste("B002 ARIMA(",p002wkday,",",d002wkday,",",q002wkday,")(",P002WKDAY,",",D002WKDAY,",",Q002WKDAY,") forecast for weekdays",sep=""))
dev.off()
jpeg("B002_ARIMA_forecast_comparison_weekdays.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB002wkday$mean,lambda = BXCXLAMB002wkday),"l",xlim=c(-5,250), ylim=c(35,150),col="red",xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B002ts.wkdays2[3121:(3120+240)],"l",col="blue",xlim=c(-5,250), ylim=c(35,150),xlab="Time", ylab="Kwh",main="B002 weekdays: forecast vs. real data")
legend(230,145,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("red","blue"))
dev.off()
B002.errorwkdays<-accuracy(InvBoxCox(foreB002wkday$mean,lambda = BXCXLAMB002wkday),B002ts.wkdays2[3121:(3120+240)])
B002.errorwkdays<-matrix(B002.errorwkdays)

dev.off()
jpeg("B011_ARIMA_forecast_just_weekdays.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB011wkday,main=paste("B011 ARIMA(",p011wkday,",",d011wkday,",",q011wkday,")(",P011WKDAY,",",D011WKDAY,",",Q011WKDAY,") forecast for weekdays",sep=""))#,xlim=c(120,140), main=paste("B011 ARIMA(",p011wkday,",",d011wkday,",",q011wkday,")(",P011WKDAY,",",D011WKDAY,",",Q011WKDAY,") forecast for weekdays",sep=""))
plot(foreB011wkday,xlim=c(115,141), main=paste("B011 ARIMA(",p011wkday,",",d011wkday,",",q011wkday,")(",P011WKDAY,",",D011WKDAY,",",Q011WKDAY,") forecast for weekdays",sep=""))
dev.off()
jpeg("B011_ARIMA_forecast_comparison_weekdays.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB011wkday$mean,lambda = BXCXLAMB011wkday),"l",xlim=c(-5,250), ylim=c(5,12),col="red",xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B011ts.wkdays2[3121:(3120+240)],"l",col="blue",xlim=c(-5,250), ylim=c(5,12),xlab="Time", ylab="Kwh",main="B011 weekdays: forecast vs. real data")
legend(220,12,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("red","blue"))
dev.off()
B011.errorwkdays<-accuracy(InvBoxCox(foreB011wkday$mean,lambda = BXCXLAMB011wkday),B011ts.wkdays2[3121:(3120+240)])
B011.errorwkdays<-matrix(B011.errorwkdays)


jpeg("B123_ARIMA_forecast_just_weekdays.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB123wkday,main=paste("B123 ARIMA(",p123wkday,",",d123wkday,",",q123wkday,")(",P123WKDAY,",",D123WKDAY,",",Q123WKDAY,") forecast for weekdays",sep=""))#,xlim=c(120,140), main=paste("B123 ARIMA(",p123wkday,",",d123wkday,",",q123wkday,")(",P123WKDAY,",",D123WKDAY,",",Q123WKDAY,") forecast for weekdays",sep=""))
plot(foreB123wkday,xlim=c(115,141), main=paste("B123 ARIMA(",p123wkday,",",d123wkday,",",q123wkday,")(",P123WKDAY,",",D123WKDAY,",",Q123WKDAY,") forecast for weekdays",sep=""))
dev.off()
jpeg("B123_ARIMA_forecast_comparison_weekdays.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB123wkday$mean,lambda = BXCXLAMB123wkday),"l",xlim=c(-5,250), ylim=c(55,140),col="red",xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B123ts.wkdays2[3121:(3120+240)],xlab="Time", ylab="Kwh",main="B123 weekdays: forecast vs. real data","l",xlim=c(-5,250), ylim=c(55,140),col="blue")
legend(230,140,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("red","blue"))
dev.off()
B123.errorwkdays<-accuracy(InvBoxCox(foreB123wkday$mean,lambda = BXCXLAMB123wkday),B123ts.wkdays2[3121:(3120+240)])
B123.errorwkdays<-matrix(B123.errorwkdays)

dev.off()
jpeg("B124_ARIMA_forecast_just_weekdays.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB124wkday,main=paste("B124 ARIMA(",p124wkday,",",d124wkday,",",q124wkday,")(",P124WKDAY,",",D124WKDAY,",",Q124WKDAY,") forecast for weekdays",sep=""))#,xlim=c(120,140), main=paste("B124 ARIMA(",p124wkday,",",d124wkday,",",q124wkday,")(",P124WKDAY,",",D124WKDAY,",",Q124WKDAY,") forecast for weekdays",sep=""))
plot(foreB124wkday,xlim=c(115,141), main=paste("B124 ARIMA(",p124wkday,",",d124wkday,",",q124wkday,")(",P124WKDAY,",",D124WKDAY,",",Q124WKDAY,") forecast for weekdays",sep=""))
dev.off()
jpeg("B124_ARIMA_forecast_comparison_weekdays.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB124wkday$mean,lambda = BXCXLAMB124wkday),"l",xlim=c(-5,250), ylim=c(5,33),col="red",xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B124ts.wkdays2[3121:(3120+240)],"l",xlim=c(-5,250), ylim=c(5,33),col="blue",xlab="Time", ylab="Kwh",main="B124 weekdays: forecast vs. real data")
legend(230,30,legend=c("Forecast","Real data"),lwd=c(1,2),col=c("red","blue"))
dev.off()
B124.errorwkdays<-accuracy(InvBoxCox(foreB124wkday$mean,lambda = BXCXLAMB124wkday),B124ts.wkdays2[3121:(3120+240)])
B124.errorwkdays<-matrix(B124.errorwkdays)



foreB116wkend<-forecast(B116fitwkend, (24*10))
foreB002wkend<-forecast(B002fitwkend, (24*10))
foreB011wkend<-forecast(B011fitwkend, (24*10))
foreB123wkend<-forecast(B123fitwkend, (24*10))
foreB124wkend<-forecast(B124fitwkend, (24*10))

dev.off()
jpeg("B116_ARIMA_forecast_just_weekends.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB116wkend,main=paste("B116 ARIMA(",p116wkend,",",d116wkend,",",q116wkend,")(",P116WKEND,",",D116WKEND,",",Q116WKEND,") forecast for weekends",sep=""))#,xlim=c(120,140), main=paste("B116 ARIMA(",p116wkend,",",d116wkend,",",q116wkend,")(",P116WKend,",",D116WKend,",",Q116WKend,") forecast for weekends",sep=""))
plot(foreB116wkend,xlim=c(45,65), main=paste("B116 ARIMA(",p116wkend,",",d116wkend,",",q116wkend,")(",P116WKEND,",",D116WKEND,",",Q116WKEND,") forecast for weekends",sep=""))
dev.off()
jpeg("B116_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB116wkend$mean,lambda = BXCXLAMB116wkend),"l",col="red",ylim=c(min(B116ts.wkend2[1249:(1248+240)]),max(B116ts.wkend2[1249:(1248+240)])),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B116ts.wkend2[1249:(1248+240)],"l",col="blue",ylim=c(min(B116ts.wkend2[1249:(1248+240)]),max(B116ts.wkend2[1249:(1248+240)])),xlab="Time", ylab="Kwh",main="B116 weekends: forecast vs. real data")
legend(221,25,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
B116.errorwkend<-accuracy(InvBoxCox(foreB116wkend$mean,lambda = BXCXLAMB116wkend),B116ts.wkend2[1249:(1248+240)])
B116.errorwkend<-matrix(B116.errorwkend)

dev.off()
jpeg("B002_ARIMA_forecast_just_weekends.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB002wkend,main=paste("B002 ARIMA(",p002wkend,",",d002wkend,",",q002wkend,")(",P002WKEND,",",D002WKEND,",",Q002WKEND,") forecast for weekends",sep=""))#,xlim=c(120,140), main=paste("B002 ARIMA(",p002wkend,",",d002wkend,",",q002wkend,")(",P002WKend,",",D002WKend,",",Q002WKend,") forecast for weekends",sep=""))
plot(foreB002wkend,xlim=c(20,37), main=paste("B002 ARIMA(",p002wkend,",",d002wkend,",",q002wkend,")(",P002WKEND,",",D002WKEND,",",Q002WKEND,") forecast for weekends",sep=""))
dev.off()
jpeg("B002_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB002wkend$mean,lambda = BXCXLAMB002wkend),"l",col="red",ylim=c(min(B002ts.wkend2[625:(624+240)])-5,max(B002ts.wkend2[625:(624+240)]+5)),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B002ts.wkend2[625:(624+240)],"l",col="blue",ylim=c(min(B002ts.wkend2[625:(624+240)])-5,max(B002ts.wkend2[625:(624+240)]+5)),xlab="Time", ylab="Kwh",main="B002 weekends: forecast vs. real data")
legend(220,70,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
B002.errorwkend<-accuracy(InvBoxCox(foreB002wkend$mean,lambda = BXCXLAMB002wkend),B002ts.wkend2[625:(624+240)])
B002.errorwkend<-matrix(B002.errorwkend)


dev.off()
jpeg("B011_ARIMA_forecast_just_weekends.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB011wkend,main=paste("B011 ARIMA(",p011wkend,",",d011wkend,",",q011wkend,")(",P011WKEND,",",D011WKEND,",",Q011WKEND,") forecast for weekends",sep=""))#,xlim=c(120,140), main=paste("B011 ARIMA(",p011wkend,",",d011wkend,",",q011wkend,")(",P011WKend,",",D011WKend,",",Q011WKend,") forecast for weekends",sep=""))
plot(foreB011wkend,xlim=c(45,65), main=paste("B011 ARIMA(",p011wkend,",",d011wkend,",",q011wkend,")(",P011WKEND,",",D011WKEND,",",Q011WKEND,") forecast for weekends",sep=""))
dev.off()
jpeg("B011_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB011wkend$mean,lambda = BXCXLAMB011wkend),"l",col="red",ylim=c(min(B011ts.wkend2[625:(624+240)]),max(B011ts.wkend2[625:(624+240)])),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B011ts.wkend2[1249:(1248+240)],"l",col="blue",ylim=c(min(B011ts.wkend2[625:(624+240)]),max(B011ts.wkend2[625:(624+240)])),xlab="Time", ylab="Kwh",main="B011 weekends: forecast vs. real data")
legend(220,12.5,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
B011.errorwkend<-accuracy(InvBoxCox(foreB011wkend$mean,lambda = BXCXLAMB011wkend),B011ts.wkend2[625:(624+240)])
B011.errorwkend<-matrix(B011.errorwkend)




jpeg("B123_ARIMA_forecast_just_weekends.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB123wkend,main=paste("B123 ARIMA(",p123wkend,",",d123wkend,",",q123wkend,")(",P123WKEND,",",D123WKEND,",",Q123WKEND,") forecast for weekends",sep=""))#,xlim=c(120,140), main=paste("B123 ARIMA(",p123wkend,",",d123wkend,",",q123wkend,")(",P123WKend,",",D123WKend,",",Q123WKend,") forecast for weekends",sep=""))
plot(foreB123wkend,xlim=c(45,65), main=paste("B123 ARIMA(",p123wkend,",",d123wkend,",",q123wkend,")(",P123WKEND,",",D123WKEND,",",Q123WKEND,") forecast for weekends",sep=""))
dev.off()
jpeg("B123_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB123wkend$mean,lambda = BXCXLAMB123wkend),"l",col="red",ylim=c(min(B123ts.wkend2[1249:(1248+240)]),max(B123ts.wkend2[1249:(1248+240)])+5),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B123ts.wkend2[1249:(1248+240)],"l",col="blue",ylim=c(min(B123ts.wkend2[1249:(1248+240)]),max(B123ts.wkend2[1249:(1248+240)])+5),xlab="Time", ylab="Kwh",main="B123 weekends: forecast vs. real data")
legend(220,100,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
B123.errorwkend<-accuracy(InvBoxCox(foreB123wkend$mean,lambda = BXCXLAMB123wkend),B123ts.wkend2[1249:(1248+240)])
B123.errorwkend<-matrix(B123.errorwkend)


jpeg("B124_ARIMA_forecast_just_weekends.jpg",height = 786, width = 1048)
par(mfrow=c(2,1))
plot(foreB124wkend,main=paste("B124 ARIMA(",p124wkend,",",d124wkend,",",q124wkend,")(",P124WKEND,",",D124WKEND,",",Q124WKEND,") forecast for weekends",sep=""))#,xlim=c(120,140), main=paste("B124 ARIMA(",p124wkend,",",d124wkend,",",q124wkend,")(",P124WKend,",",D124WKend,",",Q124WKend,") forecast for weekends",sep=""))
plot(foreB124wkend,xlim=c(45,65), main=paste("B124 ARIMA(",p124wkend,",",d124wkend,",",q124wkend,")(",P124WKEND,",",D124WKEND,",",Q124WKEND,") forecast for weekends",sep=""))
dev.off()
jpeg("B124_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB124wkend$mean,lambda = BXCXLAMB124wkend),"l",col="red",ylim=c(min(B124ts.wkend2[1249:(1248+240)]),max(B124ts.wkend2[1249:(1248+240)])),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B124ts.wkend2[1249:(1248+240)],"l",col="blue",ylim=c(min(B124ts.wkend2[1249:(1248+240)]),max(B124ts.wkend2[1249:(1248+240)])),xlab="Time", ylab="Kwh",main="B124 weekends: forecast vs. real data")
legend(220,13,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
B124.errorwkend<-accuracy(InvBoxCox(foreB124wkend$mean,lambda = BXCXLAMB124wkend),B124ts.wkend2[1249:(1248+240)])
B124.errorwkend<-matrix(B124.errorwkend)

B002_desc<-stat.desc(B002$Ef_kWh)
B011_desc<-stat.desc(B011$Ef_kWh)
B116_desc<-stat.desc(B116$Ef_kWh)
B123_desc<-stat.desc(B123$Ef_kWh)
B124_desc<-stat.desc(B124$Ef_kWh)
BXCXB002_desc<-stat.desc(BXCXB002day2)
BXCXB011_desc<-stat.desc(BXCXB011day2)
BXCXB116_desc<-stat.desc(BXCXB116day2)
BXCXB123_desc<-stat.desc(BXCXB123day2)
BXCXB124_desc<-stat.desc(BXCXB124day2)
descr<-data.frame(B002_desc,BXCXB002_desc,B011_desc,BXCXB011_desc,B116_desc,BXCXB116_desc,B123_desc,BXCXB123_desc,B124_desc,BXCXB124_desc)
write.csv(descr,file="estad-descri.csv")

# B002_descBX<-stat.desc(BXCXB002day)
# B011_descBX<-stat.desc(BXCXB011day)
# B116_descBX<-stat.desc(BXCXB116day)
# B123_descBX<-stat.desc(BXCXB123day)
# B124_descBX<-stat.desc(BXCXB124day)
# descrBX<-data.frame(B002_descBX,B011_descBX,B116_descBX,B123_descBX,B124_descBX)
# write.csv(descrBX,file="estad-descriBX.csv")



B002values<-c(pB002day,dB002day,qB002day,PB002day,DB002day,QB002day) 
B011values<-c(pB011day,dB011day,qB011day,PB011day,DB011day,QB011day) 
B116values<-c(pB116day,dB116day,qB116day,PB116day,DB116day,QB116day) 
B123values<-c(pB123day,dB123day,qB123day,PB123day,DB123day,QB123day) 
B124values<-c(pB124day,dB124day,qB124day,PB124day,DB124day,QB124day) 
valuesB<-data.frame(B002values,B011values,B116values,B123values,B124values)
write.csv(valuesB,file="valuesB.csv")

eerr<-data.frame(B116whole=B116.errorwhole,B116wkdays=B116.errorwkdays,B116wkends=B116.errorwkend,B002whole=B002.errorwhole,B002wkdays=B002.errorwkdays,B002wkends=B002.errorwkend,B011whole=B011.errorwhole,B011wkdays=B011.errorwkdays,B011wkends=B011.errorwkend,B123whole=B123.errorwhole,B123wkdays=B123.errorwkdays,B123wkends=B123.errorwkend,B124whole=B124.errorwhole,B124wkdays=B124.errorwkdays,B124wkends=B124.errorwkend,row.names = c("ME","RMSE","MAE","MPE","MAPE"))
write.csv(t(eerr),file="errors_arima.csv")


L3 <- LETTERS[1:3]
fac <- sample(L3, 10, replace = TRUE)
(d <- data.frame(xa = 1, yb = 1:10, fac = fac))




B116fouwkdays<-fourier(B116ts.wkdays, K=10)
B116autoarimafouwkdays<-auto.arima(B116ts.wkdays,xreg = fourier(B116ts.wkdays, K=10), lambda=0)
graphfouB116wkdays<-plot(forecast(B116autoarimafouwkdays, xreg = fourier(B116ts.wkdays, K=10 , h=24), lambda = 0),xlim=c(260,270))

View(B116wkdays)

B116fouwkend<-fourier(B116ts.wkend, K=10)
B116autoarimafouwkend<-auto.arima(B116ts.wkend,xreg = fourier(B116ts.wkend, K=10), lambda=0)
graphfouB116wkend<-plot(forecast(B116autoarimafouwkend, xreg = fourier(B116ts.wkend, K=10 , h=24), lambda = 0),xlim=c(102,108))

fit116<-Arima(B116.tsdia, order = c(p,d,q), seasonal = c(P,D,Q), lambda = BXCXLAMB116dia, include.drift = F, include.constant = T, include.mean = T)

#par(new=T)
#plot(B116.ts2)

bestfit<-list(aicc=Inf)
for(j in 1:50){
  fit<-auto.arima(B116.ts,xreg = fourier(B116.ts,K=j),seasonal = F)
  if(fit$aicc<bestfit$aicc){
    bestfit<-fit
  }
  else
    break
}

fc<-forecast(bestfit,xreg = fourier(B116.ts,K=3,h=(24*7*3)))



dev.off()
jpeg("B116_STL_forecast.jpg",height = 786, width = 1048)
B116.stl<-stl(B116.tsweek,s.window = "periodic")
B116.fcast<-forecast(B116.stl)
plot(B116.fcast)
dev.off()

jpeg("B011_STL_forecast.jpg",height = 786, width = 1048)
B011.stl<-stl(B011.tsweek,s.window = "periodic")
B011.fcast<-forecast(B011.stl)
plot(B011.fcast)
dev.off()
jpeg("B002_STL_forecast.jpg",height = 786, width = 1048)
B002.stl<-stl(B002.tsweek,s.window = "periodic")
B002.fcast<-forecast(B002.stl)
plot(B002.fcast)


dev.off()
jpeg("B123_ARIMA_forecast_comparison_weekends.jpg",height = 786, width = 1048)
plot(1:240,InvBoxCox(foreB123wkend$mean,lambda = BXCXLAMB123wkend),"l",col="red",ylim=c(min(B123ts.wkend2[1249:(1248+240)]),max(B123ts.wkend2[1249:(1248+240)])+5),xlab="Time", ylab="Kwh")
par(new=T)
plot(1:240,B123ts.wkend2[1249:(1248+240)],"l",col="blue",ylim=c(min(B123ts.wkend2[1249:(1248+240)]),max(B123ts.wkend2[1249:(1248+240)])+5),xlab="Time", ylab="Kwh",main="B123 weekends: forecast vs. real data")
legend(220,100,legend=c("Real data","Forecast"),lwd=c(1,2),col=c("blue","red"))
dev.off()
