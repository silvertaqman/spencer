#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
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
read_csv("parameters.csv.gz")%>% 
	select(param_C, param_gamma, mean_test_score, std_test_score, rank_test_score, run) %>% 
	filter(run != 'third', param_gamma <100 ) %>%
	ggplot(aes(x=param_gamma, y=param_C, z=mean_test_score))+geom_contour_filled(bins=20)+facet_wrap(.~run, scales="free", ncol = 3)
	
ggsave('firstsecond.pdf', dpi=300)

read_csv("parameters.csv.gz")%>% 
	select(param_C, param_gamma, mean_test_score, std_test_score, rank_test_score, run) %>% 
	filter(run == 'third' ) %>%
	ggplot(aes(x=param_gamma, y=param_C, z=mean_test_score))+geom_contour_filled(bins=5)+facet_wrap(.~run)

ggsave('third.pdf', dpi=300)

# 3D scatterplot
library( rgl )
library(magick)

# Let's use the iris dataset
# iris

# This is ugly
colors <- c("royalblue1", "darkcyan", "oldlace")
iris$color <- colors[ as.numeric( as.factor(iris$Species) ) ]

# Static chart
plot3d( iris[,1], iris[,2], iris[,3], col = iris$color, type = "s", radius = .2 )

# We can indicate the axis and the rotation velocity
play3d( spin3d( axis = c(0, 0, 1), rpm = 20), duration = 10 )

# Save like gif
movie3d(
  movie="3dAnimatedScatterplot", 
  spin3d( axis = c(0, 0, 1), rpm = 7),
  duration = 10, 
  dir = "~/Desktop",
  type = "gif", 
  clean = TRUE
)

