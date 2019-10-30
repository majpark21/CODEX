# To run after pattern extraction with more than 1 channel
# Input: a path to csv table with one row per pattern, channels concatenated and all of the same length.
# All channels must have no NAs along their alignment but they might have NAs tail at the beginning or the end
# Output: a distance matrix with DTW on multivariate channels. Only upper part filled, the rest is set to Inf to be consistent
# with dtaidistance dtw implementation

user_lib <- "/home/marc/R/x86_64-pc-linux-gnu-library/3.6"
if(!dir.exists(user_lib)){
  stop(sprintf("User package library '%s' is not found. In 'dtw_multivariate_distmat.R' modify the variable user_lib to your user library directory.", user_lib))
}
library.path <- c(user_lib, .libPaths())
library(argparse, lib.loc = library.path)
library(data.table, lib.loc = library.path)
suppressMessages(library(proxy, lib.loc = library.path))
suppressMessages(library(dtw, lib.loc = library.path))
suppressMessages(library(parallelDist, lib.loc = library.path))

# Parsing arguments ----------------------------------------------
parser <- ArgumentParser(description="A script to build a distance matrix with DTW and multivariate time-series.")
parser$add_argument("-i", "--infile", type="character", help="Character. Path to the file containing the patterns. A table with one pattern per row, channels concatenated and of same length.")
parser$add_argument("-o", "--outfile", type="character", help="Character. Path to the output file containing the distance matrix.")
parser$add_argument("-l", "--length", type="integer", help="Integer. Length of time-series.")
parser$add_argument("-n", "--nchannel", type="integer", help="Integer. Number of channels in the timeseries.")
parser$add_argument("--norm", type="logical", default=TRUE, help="Logical. Whether to normalize distances to series length. Default to TRUE.")
parser$add_argument("--colid", type="character", default=NULL, help="Character. Name of ID column. Must be provided if present. default to NULL.")

args <- parser$parse_args()
print(args)
infile <- args$infile
outfile <- args$outfile
len <- args$length
nchannel <- args$nchannel
normDist <- args$norm
col_id <- args$colid

dt <- fread(infile)
# Build distance matrix, DTW in multivariate case ----------------
if(is.null(col_id) | col_id == "NULL"){
  dt[, pattID := 1:nrow(dt)]
} else {
  setnames(dt, col_id, "pattID")
}
dt_split <- split(dt, dt$pattID)  # list of DTs
dt_split <- lapply(dt_split, function(x) x[, pattID := NULL])
dt_split <- lapply(dt_split, function(x) matrix(unlist(x), nrow=nchannel, ncol=len, byrow = T))  # reshape time on column, channel on rows
dt_split <- lapply(dt_split, function(x) x[, colSums(is.na(x)) == 0]) # clip time points with NAs (speeds up dtw a lot)

norm_method <- ifelse(normDist, "path.length", "")
dist_mat <- parDist(dt_split, method="dtw", step.pattern="symmetric2", norm.method = norm_method, window.size = 1)

# Save dist object
dist_mat <- as.matrix(dist_mat)
rownames(dist_mat) <- colnames(dist_mat) <- dt[, pattID]
dist_mat <- as.dist(dist_mat)
saveRDS(dist_mat, paste0(outfile, ".rds"))

# Save dist matrix
dist_mat <- as.matrix(dist_mat)
diag(dist_mat) <- Inf
dist_mat[lower.tri(dist_mat)] <- Inf
suppressMessages(fwrite(dist_mat, paste0(outfile, ".csv.gz"), sep = ",", row.names = FALSE, col.names = FALSE))
