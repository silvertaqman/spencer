#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)
library(gg3D)
library(ggmosaic)
# Exploratory analysis

# Loading data and add a run column, then drop na values
surfacelin <- read_csv("linear.csv.gz") %>%
	select(param_C, param_gamma, mean_test_score)
surfacerbf <- read_csv("rbf.csv.gz") %>%
	select(param_C, param_gamma, mean_test_score)
surfacelog <- read_csv("logistic.csv.gz") %>%
	select(param_C, param_penalty, param_solver, mean_test_score)
surfacemlp <- read_csv("mlp.csv.gz") %>%
	select(param_activation, param_alpha, param_hidden_layer_sizes,param_learning_rate_init,param_solver, mean_test_score) %>%
	mutate_if(is.character, as.factor)

# Plots: sequential explanation to delimite best parameters
# RBF and Linear
contourPlt <- function(surface, name){
	surface %>%	
	ggplot(aes(x=param_gamma, y=param_C, z=mean_test_score))+
	geom_contour_filled(bins = 12)+
  # New fill and legend title for number of tracks per region
  scale_fill_gradientn(
    "Accuracy",
     colours = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"),
     super = metR::ScaleDiscretised)+
  theme(
    # Remove axis ticks and text
#    axis.title = element_blank(),
#    axis.ticks = element_blank(),
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  # Make the guide for the fill discrete
  guides(
    fill = guide_colorsteps(
      barwidth = 32, barheight = .75, title.position = "top", title.hjust = .5
    )
  )
}
# GRAPHICAL OPTIONS

# 3D scatterplot
scatterPlt <- function(surface){
	surface %>%
	ggplot(
		aes(x=param_gamma, 
				y=param_C, 
				z=mean_test_score, 
				colour = as.factor(cut(mean_test_score, breaks = 10))))+
	axes_3D(theta=160,phi=60)+
	stat_3D(theta=160, phi=60, geom="point", size = 2)+
  scale_colour_gradientn(
    "Accuracy",
     colours = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"),
     super = metR::ScaleDiscretised)+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  # Make the guide for the fill discrete
  guides(
    colour = guide_colorsteps(
      barwidth = 20, barheight = 0.5, title.position = "left", title.hjust = .5
    )
  )
}

# Dynamic chart
#library(plotly)
#surfacerbf %>% plot_ly(
#	x=~param_gamma, 
#	y=~param_C, 
#	z=~mean_test_score,
#	color =~mean_test_score,
#	type="scatter3d", 
#	mode="markers",
#	colors = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"))

# Logistic

scatlog <- surfacelog %>% 
	filter(mean_test_score > 0.85) %>%
	ggplot(
	aes(x=param_C, y = mean_test_score, colour=as.factor(param_solver), shape = as.factor(param_penalty)))+
	geom_point(size = 3)+
	geom_path()+
	scale_shape_manual("Penalización",values =c(18, 5))+
  scale_color_manual(
    "Algoritmo",
     values = c("#F26F7E","#5E1E5B","#F89B0F")
     )+
  scale_x_log10()+
  theme(
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  ylab('Accuracy')

# MLP

# Jitterplot

scatmlp <- surfacemlp %>% 
 filter(mean_test_score > 0.85) %>%
 ggplot(
 	aes(x = param_hidden_layer_sizes, 
 			y = param_activation, 
 			colour = mean_test_score, 
 			size = param_learning_rate_init, 
 			shape = param_solver))+
 	geom_jitter(width=0.20, height=0.20)+
  # New fill and legend title for number of tracks per region
  scale_colour_gradient("Accuracy", low="#5E1E5B", high="#F89B0F")+
  scale_shape_manual('Algoritmo',values =c(18, 5))+
 	scale_alpha('Alpha', range=c(0.4,0.8))+
 	scale_size('Learning Rate Init')+
  theme(
    # Remove axis ticks and text
#    axis.title = element_blank(),
#    axis.ticks = element_blank(),
    # Use gray text for the region names
    axis.text.x = element_text(color = "gray12", size = 12),
    axis.text.y = element_text(color = "gray12", size = 12),
    # Move the legend to the bottom
    legend.position = "bottom",
    # Set default color and font family for the text
    text = element_text(color = "gray12"),
    # Make the background white and remove extra grid lines
    panel.background = element_rect(fill = "white", color = "white"),
    panel.grid = element_blank(),
    panel.grid.major.x = element_blank()
  )+
  ylab('Activation Curve')+
  xlab('Size of Hidden Layer')

# Merge plots
con <- ggarrange(contourPlt(surfacelin), 
					contourPlt(surfacerbf),
					labels = c("A", "B"),
					ncol = 2, 
					nrow = 1)
sca <- ggarrange(scatterPlt(surfacelin), 
					scatterPlt(surfacerbf),
					scatlog,
					scatmlp,
					labels = c("A", "B","C", "D"),
					ncol = 2, 
					nrow = 2)

ggsave("contours.pdf",
				con, 
				dpi=300,
				width = 4000, 
				height = 2000, 
				units = "px",
				useDingbats=FALSE)
ggsave("scatters.pdf",
				sca, 
				dpi=300,
				width = 4000, 
				height = 4000, 
				units = "px",
				useDingbats=FALSE)
