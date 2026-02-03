#-------------------------------------------------
# Use binom mixtures from Coorens et al., 2018 to deconvolve LCM samples.
#-------------------------------------------------

library(tidyverse)
source('/Users/cossa/Desktop/projects/manas_embryo/code/binom_mixture.R')

# Paths
path_main <- '/Users/cossa/Desktop/projects/manas_embryo'
path_data <- paste(c(path_main, 'data'), collapse='/')
path_results <- paste(c(path_main, 'results'), collapse='/')

# Load data
dataset <- 'Heart'
heart_meta <- read.csv(paste(c(path_data, paste(c(dataset, 'metadata.csv'), collapse='_')), collapse='/'))
samples <- heart_meta %>% pull(Sample_ID) %>% unique

# Apply binomial mixtures
thr_coverage <- 2
n_clones <- c()
n_muts <- c()
highest_VAF <- c()
for (s in samples) {
  NV <- heart_meta %>% subset(Sample_ID==s) %>% pull(NV)
  NR <- heart_meta %>% subset(Sample_ID==s) %>% pull(NR)
  test <- NV>=thr_coverage
  NV <- NV[test]
  NR <- NR[test]  
  res <- binom_mix(NV,NR,nrange=1:5,criterion="BIC",maxit=5000,tol=1e-6, mode="Full")
  n_clones <- c(n_clones, res$n)
  n_muts <- c(n_muts, length(NV))
  highest_VAF <- c(highest_VAF, sort(NV/NR, decreasing=T)[1])
}  
write.csv(
  data.frame(samples=samples, n_clones=n_clones, n_muts=n_muts, highest_VAF=highest_VAF),
  paste(c(path_results, paste(c(dataset, 'n_clones.csv'), collapse='_')), collapse='/')
)


##




