#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(knitr)
library(ggplot2)
library(gt)
library(stringr)
library(pilot)

# Generates a table for summarise statistics
# read_csv("../validation_curve.csv") %>%
<<<<<<< HEAD
media <- read_csv("../validation_metrics.csv.gz") %>%
=======
media <- read_csv("../validation_metrics.csv") %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  )

gtsave(media, "Medias.rtf")

<<<<<<< HEAD
sds <- read_csv("../validation_metrics.csv.gz") %>%
=======
sds <- read_csv("../validation_metrics.csv") %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, sd)) %>%
  gt(
  	rowname_col = "model",
  	groupname_col = "method"
  )

gtsave(sds, "Desviaciones.rtf")

#gtsave("tab_1.tex"), gtsave("tab_1.rtf")

# grafico de intervalos de confianza (best ensembles are stack_1 and bagmlp)
<<<<<<< HEAD
medias <- read_csv("../validation_metrics.csv.gz") %>%
=======
medias <- read_csv("../validation_metrics.csv") %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, mean))%>%
	pivot_longer(starts_with("test_"), names_to="Metric", values_to="media") %>%
	mutate(Metric = str_remove(Metric, "test_")) %>%
	mutate(model = factor(model))

<<<<<<< HEAD
sds <- read_csv("../validation_metrics.csv.gz") %>%
=======
sds <- read_csv("../validation_metrics.csv") %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	select(-...1, -fit_time, -score_time) %>%
	group_by(model,method) %>%
	summarise(across(!folds, sd))%>%
	pivot_longer(starts_with("test_"), names_to="Metric", values_to="sd") %>%
	mutate(Metric = str_remove(Metric, "test_")) %>%
	mutate(model = factor(model))

# Heatmap for metrics
lims <- medias %>%
	inner_join(sds) %>%
<<<<<<< HEAD
	filter(method == "kfold") %>%
=======
	filter(method == "kfold", Metric != "neg_log_loss") %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	group_by(model, Metric) %>%
	select(!method) %>%
	mutate(media = mean(media), sd=mean(sd))

<<<<<<< HEAD
=======
levels(lims$model) <- c("M12","M11","M10","M8","M9","M7","REMOVE","M1","M3","REMOVE","REMOVE","M2","M4","M5","M6","REMOVE")

lims <- lims  %>% 
	filter(model != "REMOVE") %>% 
	ungroup()

orden <- c(1:15,21:25,16:20,26:30,56:60,51:55,46:50,36:40,41:45,31:35)

>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
heatmap <- lims %>%
	ggplot(aes(x=Metric, y=model, fill=media))+
		geom_tile()+
		geom_label(
			aes(label=paste0(100*round(media,3),"±",100*round(sd, 3))), colour="white")+
<<<<<<< HEAD
		scale_y_discrete(limits=unique(lims$model))+
=======
		scale_y_discrete(limits=unique(lims$model[orden]))+
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
		scale_x_discrete(position="top")+
		theme_pilot()
ggsave("tabla.png",heatmap,dpi=300, width = 2000, height = 2000,bg = "white", units = "px")	
