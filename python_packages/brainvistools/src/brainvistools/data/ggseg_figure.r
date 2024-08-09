library(readxl)
library(ggseg)
library(ggsegSchaefer)
library(tidyverse)
library(ggplot2)
library(dplyr)

# Read arguments from command line
args = commandArgs(trailingOnly = TRUE)
data_file = args[1]
output_file = args[2]
thresh = as.double(args[3])
atlas_str = args[4]
labels = read_csv(args[5])
int_fig_thresh = as.logical(args[6])
cold_constant = args[7]
warm_constant = args[8]
bg_colour = args[9]

if (atlas_str == "schaefer17_100"){
  atlas=schaefer17_100
} else if (atlas_str == "schaefer7_100"){
  atlas=schaefer7_100
} else if (atlas_str == "schaefer17_200"){
  atlas=schaefer17_200
} else if (atlas_str == "schaefer7_200"){
  atlas=schaefer7_200
} else if (atlas_str == "schaefer17_300"){
  atlas=schaefer17_300
} else if (atlas_str == "schaefer7_300"){
  atlas=schaefer7_300
} else if (atlas_str == "schaefer17_400"){
  atlas=schaefer17_300
} else if (atlas_str == "schaefer7_400"){
  atlas=schaefer7_300
}
# Add more cases here if needed for different atlases...

ggseg_figure <- function(data_file,output_file,thresh,atlas,labels,int_fig_thresh=TRUE,cold_constant=FALSE,warm_constant=FALSE,bg_colour='white'){
  label_data = data.frame(labels)
  plot_data <- read_csv(data_file,col_names=c('value'))

  # Check for cases where there is no data beyond threshold in pos or neg direction
  if(min(plot_data) > -1*thresh){
    lim_min = thresh
    if (thresh == 0){
      lim_min = min(plot_data)
    }
  }
  else{
    lim_min = min(plot_data)
  }
  
  if(max(plot_data) < thresh){
    lim_max = -1*thresh
    if (thresh == 0){
      lim_max = max(plot_data)
    }
  }
  else{
    lim_max = max(plot_data)
  }
  
  # Set abs value of plot limits (min and max) to be equal if data on both sides of zero (to keep colorbar ranges equivalent on pos and neg ends)
  if ((lim_min < 0) & (lim_max > 0)){
    abs_max_data = max(c(abs(lim_min),lim_max))
    lim_min = -1*abs_max_data
    lim_max = abs_max_data
  }
  
  # Set figure thresholds to integer amount if requested
  if (int_fig_thresh){
    lim_min = floor(lim_min)
    lim_max = ceiling(lim_max)
  }
  
  limits = c(lim_min,lim_max)
  
  if (warm_constant != "FALSE"){
    warm = colorRampPalette(c("#7f7f7f",warm_constant))(100)
  }else{
    warm = colorRampPalette(c(rgb(.886,.761,.133),rgb(.847,.631,.031),rgb(.922,.0,.02)))(100)
  }
  if (cold_constant != "FALSE"){
    cold = colorRampPalette(c("#7f7f7f",cold_constant))(100)
  }else{
    cold = colorRampPalette(c(rgb(.698,.757,.463),rgb(.188,.533,.639)))(100)
  }
  if (lim_min >= 0){
    custom_palette = c(warm)
  } else if (lim_max <= 0){
    custom_palette = c(rev(cold))
  } else{
    custom_palette = c(rev(cold),warm) 
  }
  
  plot.df<-cbind(label_data, plot_data[,])
  plot.df$value[plot.df$value > -1*thresh & plot.df$value < thresh] <- NA
  plot.df[ plot.df == "NaN" ] <- NA #setting the NA values from labels file to actual NAs

  nonmissing_data <- subset(plot.df, value!= "NA")
  ggseg_data <- tibble(
    region = c(nonmissing_data$region), 
    p = c(as.double(nonmissing_data$value)),
    groups = c(nonmissing_data$hemi)
  )
  
  ## Plotting
  ggseg_plot <- ggseg_data%>%
    ggseg(atlas = atlas,
          mapping=aes(fill=as.double(p)),
          position="stacked", colour="black") +
    scale_color_manual(values = custom_palette) +
    theme(legend.title=element_blank(), text=element_text(family="Arial"), axis.text=element_text(family="Arial"))
  ggseg_plot + scale_fill_gradientn(colours = custom_palette, limits=limits)
  ggsave(filename=output_file, width=1800, height=1200, device="png", units="px", bg=bg_colour)
}

ggseg_figure(data_file=data_file,output_file=output_file,thresh=thresh,atlas=atlas,labels=labels,int_fig_thresh=int_fig_thresh,cold_constant=cold_constant,warm_constant=warm_constant,bg=bg_colour)