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

# Parsing arguments ----------------------------------------------
parser <- ArgumentParser(description="A script to build a distance matrix with DTW and multivariate time-series.")
parser$add_argument("-i", "--infile", type="character", help="Path to the file containing the patterns. A table with one pattern per row, channels concatenated.")
parser$add_argument("-o", "--outfile", type="character", help="Path to the output file containing the distance matrix.")
parser$add_argument("-l", "--length", type="integer", help="Length of time-series.")
parser$add_argument("-n", "--nchannel", type="integer", help="Number of channels in the timeseries.")

args <- parser$parse_args()
infile <- args$infile
outfile <- args$outfile
len <- args$length
nchannel <- args$nchannel

dt <- fread(infile)
# Build distance matrix, DTW in multivariate case ----------------
dt[, pattID := 1:nrow(dt)]
dt_split <- split(dt, dt$pattID)  # list of DTs
dt_split <- lapply(dt_split, function(x) x[, pattID := NULL])
dt_split <- lapply(dt_split, function(x) matrix(unlist(x), nrow=len, ncol=nchannel))  # reshape time on row, channel on columns

dist_mat <- matrix(-1, nrow=length(dt_split), ncol=length(dt_split))
pb <- txtProgressBar(min=1, max=length(dt_split)-1, initial = 1, style=3 )
for(i in 1:(length(dt_split)-1)){
  setTxtProgressBar(pb,i)
  for(j in (i+1):length(dt_split)){
    dist_mat[i,j] <- dtw(x=na.omit(dt_split[[i]]), y=na.omit(dt_split[[j]]), distance.only = TRUE)$distance  # dtw doesn't allow NAs
  }
}
# For consistency with dtaidistance
diag(dist_mat) <- Inf
dist_mat[lower.tri(dist_mat)] <- Inf

suppressMessages(fwrite(dist_mat, outfile, sep = ",", row.names = FALSE, col.names = FALSE))