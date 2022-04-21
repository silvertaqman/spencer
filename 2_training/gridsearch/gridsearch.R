#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
# Modifications to files in ./preplotting.gz
# Loading data and add a run column, then drop na values
#surface <- read_csv("params.csv") %>%
#	add_row(read_csv("params1.csv")) %>%
#	add_row(read_csv("params2.csv")) %>%
#	mutate(run = factor(c(rep('first',64), rep('second',10000), rep("third", 900)))) %>%
#	drop_na()
# Export
#write_csv(surface, "parameters.csv")
# Plots: sequential explanation to delimite best parameters
surface <- read_csv("parameters.csv.gz") %>%
	select(param_C, param_gamma, mean_test_score, run) %>%
	mutate(accuracy = cut(mean_test_score, breaks = 10))	
 
surface %>%	
	filter(run != 'third', param_gamma <100 ) %>%
	ggplot(aes(x=param_gamma, y=param_C, z=mean_test_score))+
	geom_contour_filled(bins = 20)+
	facet_wrap(.~run, scales="free", ncol = 3)+
  # New fill and legend title for number of tracks per region
  scale_fill_gradientn(
    "Accuracy",
     colours = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"),
     super = metR::ScaleDiscretised)+
  theme(
    # Remove axis ticks and text
    axis.title = element_blank(),
    axis.ticks = element_blank(),
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
      barwidth = 50, barheight = .75, title.position = "top", title.hjust = .5
    )
  )
	
ggsave('firstsecond.pdf', dpi=300)

surface %>% 
	filter(run == 'third') %>%
	ggplot(aes(x=param_gamma, y=param_C, z=mean_test_score))+
	geom_contour_filled(bins=10)+
	facet_wrap(.~run)+
  scale_fill_gradientn(
    "Accuracy",
     colours = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"),
     super = metR::ScaleDiscretised)+
  theme(
    # Remove axis ticks and text
    axis.title = element_blank(),
    axis.ticks = element_blank(),
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
      barwidth = 20, barheight = .75, title.position = "top", title.hjust = .5
    )
  )

ggsave('third.pdf', dpi=300)

# GRAPHICAL OPTIONS

# 3D scatterplot
library(gg3D)

surface %>%
	filter(run == 'second', param_gamma < 100, param_gamma > 0, ) %>% 
	ggplot(
		aes(x=param_gamma, 
				y=param_C, 
				z=mean_test_score, 
				colour = accuracy))+
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
      barwidth = 50, barheight = 0.5, title.position = "left", title.hjust = .5
    )
  )

ggsave('scatter.pdf', dpi=300)

# Dynamic chart
library(plotly)
surface %>% plot_ly(
	x=~param_gamma, 
	y=~param_C, 
	z=~mean_test_score,
	color =~mean_test_score,
	type="scatter3d", 
	mode="markers",
	colors = c("#5E1E5B","#A62231","#F26F7E","#E95420","#F89B0F"))
