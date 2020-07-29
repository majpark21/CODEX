user_lib <- "C:/Users/pixel/Documents/R/win-library/3.6"
if(!dir.exists(user_lib)){
  stop(sprintf("User package library '%s' is not found. In 'dtw_multivariate_distmat.R' modify the variable user_lib to your user library directory.", user_lib))
}
library.path <- c(user_lib, .libPaths())
library(argparse, lib.loc = library.path)
library(data.table, lib.loc = library.path)
suppressMessages(library(reshape2, lib.loc = library.path))
library(ggplot2, lib.loc = library.path)
library(stringr, lib.loc = library.path)
suppressMessages(library(dendextend, lib.loc = library.path))
#library(kohonen, lib.loc = library.path)

# Parsing arguments ----------------------------------------------
parser <- ArgumentParser(description="A script to cluster time series, extract medoids of clusters and return plots of results.")
parser$add_argument("-d", "--distfile", type="character", help="Path to the file containing the distance matrix. Should be square, with lower part and diagonal filled with Inf.")
parser$add_argument("-p", "--pattfile", type="character", help="Path to the file containing the series of the patterns. A table with one pattern per row, channels concatenated.")
parser$add_argument("-o", "--outfile", type="character", help="Path to the pdf file containing the plots.")
parser$add_argument("-l", "--length", type="integer", help="Length of time-series.")
parser$add_argument("-n", "--nchannel", type="integer", help="Number of channels in the timeseries.")
parser$add_argument("-c", "--ncluster", type="integer", help="Number of clusters to cut the tree into.", default = 4)
parser$add_argument("-m", "--nmedoid", type="integer", help="Number of medoids to extract and plot for each cluster.", default = 3)
parser$add_argument("-t", "--npatt", type="integer", help="Number of random trajectories to plot for each cluster.", default = 16)
parser$add_argument("--colid", type="character", default=NULL, help="Character. Name of ID column. Must be provided if present. default to NULL.")
parser$add_argument("--linkage", type="character", default="complete", help="Character. Linkage method, one of: ward.D, ward.D2, single, complete, average, mcquitty, median or centroid. Default to complete.")

args <- parser$parse_args()
print(args)
dist_file <- args$distfile
patt_file <- args$pattfile
out_file <- args$outfile
len <- args$length
n_channel <- args$nchannel
n_clust <- args$ncluster
n_medoid <- args$nmedoid
n_patt <- args$npatt
col_id <- args$colid
linkage <- args$linkage
if(!linkage %in% c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")) {
  stop("Linkage must be one of: ward.D, ward.D2, single, complete, average, mcquitty, median or centroid.")
}

# dist_file <- "/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/ERK_AKT/local_patterns/uncorr_dist_norm_allPooled.csv.gz"
# patt_file <- "/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/ERK_AKT/local_patterns/patt_uncorr_allPooled.csv.gz"
# out_file <- "/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/ERK_AKT/local_patterns/patt_uncorr_allPooled.pdf"
# len <- 400
# n_channel <- 2
# n_clust <- 5
# col_id <- "pattID"
# n_medoid <- 3
# n_patt <- 16
# linkage <- "complete"

############## Clustering ----
dist_mat <- fread(dist_file, header = TRUE)
dist_mat <- as.matrix(dist_mat)
rownames(dist_mat) = colnames(dist_mat)

# Make it symetrical
diag(dist_mat) <- 0
copy_up_low <- function(m) {
  m[lower.tri(m)] <- t(m)[lower.tri(m)]
  return(m)
}

dist_mat <- copy_up_low(dist_mat)
dist_mat <- as.dist(dist_mat)

tree <- hclust(dist_mat, method = linkage)
cluster_idx <- stats::cutree(tree, k = n_clust)
cluster_idx <- data.table(pattID=names(cluster_idx), cluster_idx)
cluster_idx[, pattID := str_replace(pattID, "^V", "pattID_")]


############## Medoid of clusters ----
# Get distance matrix in long format
dist_mat <- fread(dist_file, header = TRUE)
dist_mat <- as.matrix(dist_mat)
rownames(dist_mat) <- colnames(dist_mat)
dist_mat_long <- as.data.table(subset(melt(dist_mat), value!=Inf))
setnames(dist_mat_long, c("Var1", "Var2", "value"), c("pattID1", "pattID2", "distance"))
dist_mat_long[, c("pattID1", "pattID2") := list(str_replace(pattID1, "^V", "pattID_"), str_replace(pattID2, "^V", "pattID_"))]

medoid_cluster <- function(dists, clusts, nmed, permut_ids=TRUE){
  # Clusts: data.table with 2 columns ID and cluster_idx
  # Dists: data.table with 3 columns: ID1, ID2, distance
  med_list <- list()
  i <- 1
  for(clust in unique(clusts$cluster_idx)){
    id_clust <- clusts[cluster_idx==clust, pattID]
    # Subset only distances relative to the cluster
    clust_dist <- dists[pattID1 %in% id_clust & pattID2 %in% id_clust]
    # Permute IDs and append if only one order is present in the distance matrix
    if(permut_ids){
      permuted <- copy(clust_dist)
      setnames(permuted, c("pattID1", "pattID2"), c("pattID2", "pattID1"))
      clust_dist <- rbindlist(list(clust_dist, permuted), use.names = FALSE)
    }
    # Keep ids that have minimal median distance to all others
    med_distance <- clust_dist[, .(med_dist=median(distance)), by=pattID1]
    setkey(med_distance, med_dist)
    med_distance <- med_distance[1:nmed]
    med_distance[, cluster_idx := clust]
    med_list[[i]] <- med_distance
    i <- i + 1
  }
  out <- rbindlist(med_list)
  return(out)
}

medoids <- medoid_cluster(dists = dist_mat_long, clusts = cluster_idx, nmed = n_medoid)
medoids[, rank := 1:nrow(.SD), by=cluster_idx]
setnames(medoids, "pattID1", "pattID")


############# SOM ----
# dt_patt <- fread(patt_file)
# mat_patt <- as.matrix(dt_patt[,2:ncol(dt_patt)])
# mygrid <- somgrid(xdim=5, ydim=3, topo="rectangular")
# som_patt <- som(mat_patt, maxNA.fraction=0.75, grid=mygrid)
# plot(som_patt, type="changes")
# plot(som_patt, type="count")
# plot(som_patt)
# palet <- rainbow(n_clust)
# plot(som_patt, type="mapping", bgcol = palet[stats::cutree(tree, k = n_clust)], main = "Clusters")
 


############## Pattern plotting ----
dt_patt <- fread(patt_file)
if(is.null(col_id) | col_id == "NULL"){
  dt_patt[, pattID := paste0("patt_", as.character(1:nrow(dt_patt)))]
} else {
  setnames(dt_patt, col_id, "pattID")
}
time_max <- ceiling(len/n_channel)
print(len)
print(n_channel)
print(time_max)

# Plot sample from each cluster4x4 on different PDF pages
dt_plot <- melt(dt_patt, id.vars = "pattID")
dt_plot[, Measure := str_extract(variable, "^[A-Za-z]+")][, Measure := factor(Measure, unique(Measure))]
dt_plot[, Time := as.numeric(str_extract(variable, "[0-9]+$"))]
print(dt_plot)
dt_plot <- dt_plot[Time <= time_max]
print(dt_plot)
dt_plot <- merge(dt_plot, cluster_idx, by="pattID")
#dt_plot[, cluster_idx := factor(cluster_idx, levels = 1:n_clust)]

pdf(out_file)
# Hclust plot
tree_plot <- dendextend::color_branches(as.dendrogram(tree), k = n_clust, groupLabels = F)
plot(tree_plot)
# Medoid plot
dt_medoid <- dt_plot[pattID %in% medoids$pattID]
dt_medoid <- merge(dt_medoid, medoids, by=c("pattID", "cluster_idx"))
dt_medoid[, cluster_idx := factor(cluster_idx, levels = 1:n_clust)]
p1 <- ggplot(dt_medoid, aes(x=Time, y=value)) +
  geom_line(aes(color=Measure)) +
  facet_grid(cluster_idx~rank) +
  # geom_label(aes(label = round(med_dist, 3)), x=max(dt_medoid$Time)-5, y=min(dt_medoid$value, na.rm=T) + 0.1) +
  theme_bw() +
  ggtitle(paste0("Top ", as.character(n_medoid), " medoids of the ", as.character(n_clust), " clusters"))
plot(p1)
for(i in unique(dt_plot[, cluster_idx])){
  if(length(unique(dt_plot[cluster_idx == i, pattID])) < n_patt){
    n_patt_plot <- length(unique(dt_plot[cluster_idx == i, pattID]))
  } else {
    n_patt_plot <- n_patt
  }
  # Pattern plot
  traj_to_plot <- sample(unique(dt_plot[cluster_idx == i, pattID]), n_patt_plot, replace = F)
  p2 <- ggplot(dt_plot[pattID %in% traj_to_plot], aes(x = Time, y=value)) +
    geom_line(aes(color=Measure)) +
    facet_wrap("pattID") +
    theme_bw() +
    ggtitle(paste0("Cluster number: ", i))
  plot(p2)
}

dev.off()
