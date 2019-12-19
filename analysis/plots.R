# import feather files from python
# setwd("C:\\Users\\nilsp\\OneDrive\\Uni\\Hiwi\\Python\\algowatch-datenspende\\workingData")
library(here)
library(feather)
library(tidyverse)
list.filenames<-list.files(here("..", "workingData"),pattern=".feather$")

all.keywords <- data.frame()
for (i in 1:length(list.filenames))
{
  df <- read_feather(here("..", "workingData", list.filenames[i]))
  all.keywords <- rbind(all.keywords, df)
}

all.keywords %>% ggplot() +
  aes(x= timestamp, y= cluster) +
  geom_jitter(shape=16, alpha= 0.05) +
  facet_wrap(. ~ keyword) +
  NULL 
