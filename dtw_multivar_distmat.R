# To run after pattern extraction with more than 1 channel
# Input: a path to csv table with one row per pattern, channels concatenated and all of the same length.
# All channels must have no NAs along their alignment but they might have NAs tail at the beginning or the end
# Output: a distance matrix with DTW on multivariate channels. Only upper part filled, the rest is set to Inf to be consistent
# with dtaidistance dtw implementation

user_lib <- "C:/Users/Marc/Documents/R/win-library/3.6"
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
parser$add_argument("--center", type="character", choices=c("T", "TRUE", "True", "F", "FALSE", "False"), default="T", help="Logical provided as character 'T' or 'F'. Whether to zero-center the patterns before running DTW. Each channel of each pattern is centered independently.")
parser$add_argument("--norm", type="character", choices=c("T", "TRUE", "True", "F", "FALSE", "False"), default="T", help="Logical provided as character 'T' or 'F'. Whether to normalize distances to series length. Default to TRUE.")
parser$add_argument("--colid", type="character", default=NULL, help="Character. Name of ID column. Must be provided if present. default to NULL.")
parser$add_argument("--csv", type="character", choices=c("T", "TRUE", "True", "F", "FALSE", "False"), default="T", help="Logical provided as character 'T' or 'F'. Whether to export the distance matrix as a compressed csv.")
parser$add_argument("--rds", type="character", choices=c("T", "TRUE", "True", "F", "FALSE", "False"), default="T", help="Logical provided as character 'T' or 'F'. Whether to export the distance matrix as an rds R object.")

args <- parser$parse_args()
#print(args)
infile <- args$infile
outfile <- args$outfile
len <- args$length
nchannel <- args$nchannel
center <- ifelse(args$center %in% c("T", "TRUE", "True"), TRUE, FALSE)
normDist <- ifelse(args$norm %in% c("T", "TRUE", "True"), TRUE, FALSE)
col_id <- args$colid
exp_csv <- ifelse(args$csv %in% c("T", "TRUE", "True"), TRUE, FALSE)
exp_rds <- ifelse(args$rds %in% c("T", "TRUE", "True"), TRUE, FALSE)


# infile <- "/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/FRST/local_patterns/patt_uncorr_A.csv.gz"
# outfile <- "/home/marc/Dropbox/Work/TSclass_GF/Notebooks/output/FRST/local_patterns/uncorr_dist_norm_A"
# len <- 400
# nchannel <-1
# center <- F
# normDist <-T
# col_id <- "NULL"
# exp_csv <- T
# exp_rds <- T

dt <- fread(infile)
if(is.null(col_id) | col_id == "NULL"){
  dt[, pattID := paste0("patt_", as.character(1:nrow(dt)))]
} else {
  setnames(dt, col_id, "pattID")
}

# parDist expects a simple matrix for univariate, list of matrices for multivariate. But in univariate case, still keep the list structure
# otherwise problem with series of variable length: NAs are not allowed and clipping rows with NAs in a single matrix would remove time points
dt_split <- split(dt, dt$pattID)  # list of DTs
dt_split <- lapply(dt_split, function(x) x[, pattID := NULL])
dt_split <- lapply(dt_split, function(x) matrix(unlist(x), nrow=nchannel, ncol=len, byrow = T))  # reshape time on columns, channel on rows
if(nchannel==1){
  # otherwise get converted to vector in univariate case
  dt_split <- lapply(dt_split, function(x) matrix(x[, colSums(is.na(x)) == 0], nrow=1))
  #dt_split <- lapply(dt_split, function(x) rbind(x, rep(1, ncol(x))))
} else {
  dt_split <- lapply(dt_split, function(x) x[, colSums(is.na(x)) == 0]) # clip time points with NAs (speeds up dtw a lot)
}

if(center){
  # scale works on matrix columns so transpose, center each channel indeptly and transpose back
  dt_split <- lapply(dt_split, function(x) {t(scale(t(x), center=TRUE, scale=FALSE))})
}

norm_method <- ifelse(normDist, "n+m", "")
dist_mat <- parDist(dt_split, method="dtw", step.pattern="symmetric2", norm.method = norm_method)

# Save dist object
dist_mat <- as.matrix(dist_mat)
rownames(dist_mat) <- colnames(dist_mat) <- dt[, pattID]
dist_mat <- as.dist(dist_mat)
if(exp_rds) saveRDS(dist_mat, paste0(outfile, ".rds"))

# Save dist matrix
dist_mat <- as.matrix(dist_mat)
diag(dist_mat) <- Inf
dist_mat[lower.tri(dist_mat)] <- Inf
if(exp_csv) suppressMessages(fwrite(dist_mat, paste0(outfile, ".csv.gz"), sep = ",", row.names = FALSE, col.names = TRUE))
