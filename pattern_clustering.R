user_lib <- "/home/marc/R/x86_64-pc-linux-gnu-library/3.6"
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

args <- parser$parse_args()
dist_file <- args$distfile
patt_file <- args$pattfile
out_file <- args$outfile
len <- args$length
n_channel <- args$nchannel
n_clust <- args$ncluster
n_medoid <- args$nmedoid
n_patt <- args$npatt


# classe <- "B"
# dist_file <-  paste0("output/FRST_classAB/local_patterns2/dist_", classe, ".csv")
# patt_file <- paste0("output/FRST_classAB/local_patterns2/patt_", classe, ".csv")
# out_file <- str_replace(patt_file, "\\.csv", "\\.pdf")
# n_patt <- 16
# n_clust <- 4
# n_medoid <- 3

############## Clustering
dist_mat <- fread(dist_file)
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

tree <- hclust(dist_mat)
cluster_idx <- stats::cutree(tree, k = n_clust)
cluster_idx <- data.table(ID=names(cluster_idx), cluster_idx)
cluster_idx[, ID := str_replace(ID, "^V", "ID_")]


############## Medoid of clusters
# Get distance matrix in long format
dist_mat <- fread(dist_file)
dist_mat <- as.matrix(dist_mat)
rownames(dist_mat) = colnames(dist_mat)
dist_mat_long <- as.data.table(subset(melt(dist_mat), value!=Inf))
setnames(dist_mat_long, c("Var1", "Var2", "value"), c("ID1", "ID2", "distance"))
dist_mat_long[, c("ID1", "ID2") := list(str_replace(ID1, "^V", "ID_"), str_replace(ID2, "^V", "ID_"))]

medoid_cluster <- function(dists, clusts, nmed, permut_ids=TRUE){
  # Clusts: data.table with 2 columns ID and cluster_idx
  # Dists: data.table with 3 columns: ID1, ID2, distance
  med_list <- list()
  i <- 1
  for(clust in unique(clusts$cluster_idx)){
    id_clust <- clusts[cluster_idx==clust, ID]
    # Subset only distances relative to the cluster
    clust_dist <- dists[ID1 %in% id_clust & ID2 %in% id_clust]
    # Permute IDs and append if only one order is present in the distance matrix
    if(permut_ids){
      permuted <- copy(clust_dist)
      setnames(permuted, c("ID1", "ID2"), c("ID2", "ID1"))
      clust_dist <- rbindlist(list(clust_dist, permuted))
    }
    # Keep ids that have minimal median distance to all others
    med_distance <- clust_dist[, .(med_dist=median(distance)), by=ID1]
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
setnames(medoids, "ID1", "ID")


############# SOM
#dt_patt <- fread(patt_file)
#mat_patt <- as.matrix(dt_patt)
#mygrid <- somgrid(xdim=5, ydim=3, topo="rectangular")
#som_patt <- som(mat_patt, maxNA.fraction=0.75, grid=mygrid)
#plot(som_patt, type="changes")
#plot(som_patt, type="count")
#plot(som_patt)
#palet <- rainbow(n_clust)
#plot(som_patt, type="mapping", bgcol = palet[stats::cutree(tree, k = n_clust)], main = "Clusters") 
 


############## Pattern plotting
dt_patt <- fread(patt_file)
dt_patt[, ID := paste0("ID_", 1:nrow(dt_patt))]

# Plot sample from each cluster4x4 on different PDF pages
dt_plot <- melt(dt_patt, id.vars = "ID")
dt_plot[, Time := as.numeric(str_extract(variable, "[0-9]+"))]
dt_plot <- merge(dt_plot, cluster_idx, by="ID")
dt_plot[, cluster_idx := factor(cluster_idx, levels = 1:n_clust)]

pdf(out_file)
# Hclust plot
tree_plot <- dendextend::color_branches(as.dendrogram(tree), k = n_clust, groupLabels = F)
plot(tree_plot)
# Medoid plot
dt_medoid <- dt_plot[ID %in% medoids$ID]
dt_medoid <- merge(dt_medoid, medoids, by=c("ID", "cluster_idx"))
p1 <- ggplot(dt_medoid, aes(x=Time, y=value)) +
  geom_line(aes(group=ID)) +
  facet_grid(cluster_idx~rank) +
  geom_label(aes(label = round(med_dist, 3)), x=max(dt_medoid$Time)-5, y=min(dt_medoid$value, na.rm=T) + 0.1) +
  theme_bw() +
  ggtitle(paste0("Top ", as.character(n_medoid), " medoids of the ", as.character(n_clust), " clusters"))
plot(p1)
for(i in unique(dt_plot[, cluster_idx])){
  if(length(unique(dt_plot[cluster_idx == i, ID])) < n_patt){
    n_patt_plot <- length(unique(dt_plot[cluster_idx == i, ID]))
  } else {
    n_patt_plot <- n_patt
  }
  # Pattern plot
  traj_to_plot <- sample(unique(dt_plot[cluster_idx == i, ID]), n_patt_plot, replace = F)
  p2 <- ggplot(dt_plot[ID %in% traj_to_plot], aes(x = Time, y=value)) +
    geom_line(aes(group=ID)) +
    facet_wrap("ID") +
    theme_bw() +
    ggtitle(paste0("Cluster number: ", i))
  plot(p2)
}

dev.off()
