# STAT 575 Analysis
# AUTHOR: Brad West

library(magrittr)
library(ggplot2)

ART_CSV <- "./data/output.csv"

raw <- readr::read_csv(ART_CSV)
