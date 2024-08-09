#! /usr/bin/Rscript
library(readxl)
library(ggseg)
library(ggsegSchaefer)
library(tidyverse)
library(ggplot2)
library(wesanderson)
library(dplyr)

args = commandArgs(trailingOnly = TRUE)
labels = read_excel(args[1])
data_file = args[2]
output_file = args[3]
data_thresh = as.double(args[4])
int_fig_thresh = as.logical(args[5])

ggseg_figure <- function(labels,data_file,output_file,data_thresh,int_fig_thresh=TRUE){
  results = data.frame(labels)
  colnames(results)[2]  <- "region"
  colnames(results)[3]  <- "hemi"
  ## Loading in the Data
  # This is your data with a value in each row that you want to visualize
  PlotData <- read_csv(data_file,col_names=c('bsr'))
  if (int_fig_thresh){
    figure_thresh = ceiling(max(c(abs(min(PlotData)),max(PlotData))))
    limits = c(-1*figure_thresh,figure_thresh)
  }
  else{
    limits = c(min(PlotData),max(PlotData))
  }
  plot.df<-cbind(results, PlotData[,])
  plot.df$bsr[plot.df$bsr > -1*data_thresh & plot.df$bsr < data_thresh] <- NA
  plot.df[ plot.df == "NaN" ] <- NA #setting the NA values from excel to actual NAs
  pal <- wes_palette("Zissou1", 50, type = "continuous")
  
  newdata <- subset(plot.df, bsr!= "NA")
  someData <- tibble(
    region = c(newdata$region), 
    p = c(as.double(newdata$bsr)),
    groups = c(newdata$hemi)
  )
  
  ## Plotting
  sp<-someData%>%
    ggseg(atlas = schaefer17_200,
          mapping=aes(fill=as.double(p)),
          position="stacked", colour="black")+
    scale_color_manual(values = pal)+
    theme(legend.title=element_blank(), text=element_text(family="Arial"), axis.text=element_text(family="Arial"))
  sp+scale_fill_gradientn(colours = pal, limits=limits)
  ggsave(filename=output_file, width=1800, height=1200, device="png", units="px", bg='white')
}

ggseg_figure(labels=labels,data_file=data_file,output_file=output_file,data_thresh=data_thresh,int_fig_thresh=int_fig_thresh)