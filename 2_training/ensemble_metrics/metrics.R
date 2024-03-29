#!/usr/bin/env Rscript
library(tidyverse)
library(pilot)
library(readr)
library(knitr)
library(patchwork)
#############
# Set general theme
#############
theme_set(
	theme(
		  # Use gray text for the region names
<<<<<<< HEAD
		  axis.text.x = element_text(color = "gray12", size = 8),
=======
		  axis.text.x = element_text(color = "gray12", size = 12),
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank(),
		  strip.background = element_blank()
#		  strip.text.x = element_blank()
		  )+
  	theme_pilot()
)
##################################################################################
# barplot for model selection
##################################################################################
<<<<<<< HEAD
datos <- read_csv("../validation_metrics.csv.gz")
=======
datos <- read_csv("../validation_metrics.csv")
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
#barmetrix <- function(x){
#	bam <- x %>%
#		select(!ends_with("_time"), !...1) %>%
#		pivot_longer(
#			cols=starts_with("test_"), 
#			names_to="Test", 
#			values_to="Metric") %>%
#		mutate_if(is.character, as.factor) %>%
#		group_by(Test, model) %>%
#		summarize(across(Metric, mean)) %>%
#		arrange(desc(Metric))
#		
#	order <- bam %>%
#		group_by(model) %>%
#		summarize(across(Metric,sum)) %>%
#		arrange(Metric)
#		
#		bam <- ggplot(bam, aes(x=model, y=Metric, fill=Test))+
#			geom_bar(stat="identity")+
#			coord_flip()+
#			ylab("Metrics")+
#  	scale_fill_pilot()+
#  	geom_hline(aes(yintercept=0),colour="red",size=1.5)+
#  	scale_x_discrete(limits = order$model)
#	return(bam)
#}

#ggsave("Metrics.png",barmetrix(datos),dpi=320,width = 2000, height = 1500,bg = "white", units = "px")
##################################################################################
# Boxplot of metrics comparison
##################################################################################
boxmetrix <- function(x){
	x %>%
<<<<<<< HEAD
		ggplot(aes(x=Test, y=Metric, fill=factor(model)))+
			geom_boxplot()+
			facet_wrap(~model, scales="free_x")+
			scale_fill_pilot()+
			theme(legend.position="none")+
			labs(fill = 'Algorithm')+
			coord_flip()
}

datos <- mutate_if(datos, is.character, as.factor)%>%
=======
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
		select(!(ends_with("_time")|...1)) %>%
		pivot_longer(
			cols=starts_with("test_"), 
			names_to="Test", 
			values_to="Metric") %>%
<<<<<<< HEAD
		mutate(Test = factor(str_remove(Test, "test_"))) %>% 
		filter(!model %in% c("dtc","mlp", "svmrbf", "lr"))  

levels(datos$Test) <- c("ACC","F1","PRE","REC","AUC")

ggsave("Metrics.png",boxmetrix(datos), dpi=300, width = 2400, height = 1600, bg = "white", units = "px")
=======
		mutate(
			Test = str_remove(Test, "test_"),
			model = factor(model, levels = paste0("M",1:12))) %>%
		filter(Test != "neg_log_loss") %>%
		ggplot(aes(x=Test, y=Metric, fill=factor(model)))+
				geom_boxplot()+
		facet_wrap(~model)+
		scale_fill_pilot()+
		theme(legend.position="none")+
		labs(fill = 'Algorithm')+
		coord_flip()
}

datos <- mutate_if(datos, is.character, as.factor)
levels(datos$model) <- c("M12","M11","M10","M8","M9","M7","REMOVE","M1","M3","REMOVE","REMOVE","M2","M4","M5","M6","REMOVE")

datos <- datos  %>% 
	filter(model != "REMOVE") %>% 
	ungroup()

ggsave("Metrics.png",boxmetrix(datos), dpi=300, width = 2500, height = 1500, bg = "white", units = "px")
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd

###############################
# ROC-Curve
###############################
library(plotROC)

# Generates a ROC curve with ggplot
<<<<<<< HEAD
pred <- read_csv("../predictions.csv.gz") %>%
	select(!c(...1, dtc, mlp, svmrbf, lr)) %>%
=======
pred <- read_csv("../predictions.csv") %>%
	select(!c(...1, firstmlp, mlp, svmrbf, lr)) %>%
	rename(M7=bagrbf, M8=baglr, M9=bagmlp, M10=adarbf, M11=adalr, M12=adadtc, M1=hard_ensemble, M2=soft_ensemble, M3=weight_ensemble, M4=stack_1, M5=stack_2, M6=stack_3) %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	pivot_longer(cols=!Reality, names_to="Model", values_to="Predictions") %>%
	ggplot(aes(m = Predictions, d = Reality, colour=Model))+
		geom_roc(n.cuts=20,labels=FALSE)+
		style_roc(theme = theme_grey)+
		scale_color_pilot()

# Los metodos STACK1, AdaDTC y AdaSVM tiene los AUC mas altos
positions<-arrange(calc_auc(pred),desc(AUC))
positions$AUC <- round(positions$AUC, 3)
<<<<<<< HEAD
pred <- read_csv("../predictions.csv.gz") %>%
	select(!c(...1, mlp, svmrbf, lr, dtc)) %>%
=======
pred <- read_csv("../predictions.csv") %>%
	select(!c(...1, firstmlp, mlp, svmrbf, lr)) %>%
	rename(M7=bagrbf, M8=baglr, M9=bagmlp, M10=adarbf, M11=adalr, M12=adadtc, M1=hard_ensemble, M2=soft_ensemble, M3=weight_ensemble, M4=stack_1, M5=stack_2, M6=stack_3) %>%
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
	pivot_longer(cols=!Reality, names_to="Model", values_to="Predictions") %>%
	ggplot(aes(m = Predictions, d = Reality, colour=Model))+
		geom_roc(n.cuts=20,labels=FALSE)+
		style_roc(theme = theme_grey)+
		scale_color_pilot(
			breaks=positions$Model, 
			labels = paste0(positions$Model,': (',positions$AUC,')'))+
		labs(color="Model: (AUC)")
		

# all merged
#all <- ((pred+barmetrix(datos))/boxmetrix(datos))+
#	plot_layout(guides = 'collect')+
#	plot_annotation(tag_levels="A")
ggsave("roc_auc.png", pred, dpi=300, width = 3000, height = 2100,bg = "white", units = "px")
