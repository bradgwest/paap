# OVERVIEW ====================================================================
# AUTHOR: Brad West
# CREATED ON: 2020-10-25
# ---
# DESCRIPTION: Create contingency table of cluster
# ---

options(warn = 1)

ROOT <- "/Users/bradwest/drive/msu/stat575/paap/"
CLUSTERS <- c(8, 3)
DATASETS <- paste0(ROOT, "img/", CLUSTERS, "/df.csv")
FIELDS_TO_KEEP <- c("lot_image_id", "cluster")

import_and_clean <- function(fn) {
  read.csv(fn)[FIELDS_TO_KEEP]
}

dfs <- lapply(DATASETS, import_and_clean)
# Join dfs
df <- merge(dfs[[1]], dfs[[2]], by = "lot_image_id")
# Add one to columns to avoid zero index
df[,2] <- df[,2] + 1
df[,3] <- df[,3] + 1
names(df) <- c("id", paste0("k", CLUSTERS))

PLOT_OUTPUT = paste0(ROOT, "img/contingency_tbl.png")
PLOT_TITLE = "Contingency Table for k=3 and k=8 Cluster Solutions"
tbl <- table(df$k3, df$k8, dnn = c("k = 3", "k = 8"))
png(PLOT_OUTPUT, width = 8, height = 6, units = "in", res = 320)
plot(
  tbl,
  main = PLOT_TITLE
)
dev.off()

with_margins_tbl <- addmargins(tbl)
with_margins_tbl
xtable::xtable(with_margins_tbl)

# prop_tbl <- prop.table(tbl, margin = 2)
# prop_tbl
