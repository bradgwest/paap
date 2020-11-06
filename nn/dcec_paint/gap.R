# OVERVIEW ====================================================================
# AUTHOR: Brad West
# CREATED ON: 2020-10-24
# ---
# DESCRIPTION: Script for performing gap cluster analysis on paap data.
# ---

options(warn=1)

ROOT <- "/Users/bradwest/drive/msu/stat575/paap/"
# dataset import
PAAP_DATASET_PATH <- paste(ROOT, "img/1/df.csv", sep = "")
FIELDS_TO_KEEP <- paste("X", 0:31, sep = "")

# config for cluster::clusGap
MAX_CLUSTERS <- 20
MONTE_CARLO_SAMPLES <- 30  # TODO Should be 50
D_POWER <- 2
H0_SPACE <- "original"
KMEANS_ITERATIONS <- 100
KMEANS_RANDOM_STARTS <- 10
KMEANS_ALGORITHM <- "Hartigan-Wong"

# plotting
PLOT_TITLE <- "GAP Statistic on Pre-Clustered Dataset"
PLOT_OUTPUT <- paste(ROOT, "img/1/gap.png", sep = "")

df <- read.csv(PAAP_DATASET_PATH, header = TRUE)[FIELDS_TO_KEEP]

# Round values so that kmeans has a better chance of converging. This avoids the
# Warning: Quick-TRANSfer stage steps exceeded maximum (= 525250)
df <- as.data.frame(lapply(df, round, digits = 1))

clus_gap_kmeans <- cluster::clusGap(
  x = df,
  FUNcluster = stats::kmeans,
  K.max = MAX_CLUSTERS,
  B = MONTE_CARLO_SAMPLES,
  spaceH0 = H0_SPACE,
  verbose = TRUE,
  # Passed to kmeans
  iter.max = KMEANS_ITERATIONS,
  nstart = KMEANS_RANDOM_STARTS,
  algorithm = KMEANS_ALGORITHM
)

# According to Tibshirani, the optimal cluster is the smallest k for which
# GAP_k \ge GAP_{k+1} - SE_{k+1}
optimal_k <- cluster::maxSE(clus_gap_kmeans$Tab[,3], clus_gap_k$Tab[,4])

# plotting
png(PLOT_OUTPUT, width = 8, height = 6, units = "in", res = 320)
plot(
  clus_gap_kmeans,
  main = PLOT_TITLE,
  xlab = "Clusters (k)"
)
dev.off()

sprintf("Optimal cluster of %s: %s", MAX_CLUSTERS, optimal_k)
sprintf("Image saved to %s", PLOT_OUTPUT)
