library(oro.nifti)
library(arrayhelpers)
library(tidyverse)

dat_raw <- readNIfTI("Parcels_Combo_brain.nii.gz")
dat <- array2df(dat_raw, label.x = "ROI")

centers <- data.frame(ROI = 1:375, x = NA, y = NA, z = NA)
for (i in 1:375) {
  ROI <- dat %>% filter(ROI == i)
  if (nrow(ROI) > 0) {
    ROI_dist <- colSums(as.matrix(dist(ROI)))
    centers[i, 2:4] <- ROI[which(ROI_dist == min(ROI_dist)), 2:4]
  }
}

center_dist <- as.matrix(dist(centers[, 2:4]))
center_dist <- center_dist / max(center_dist, na.rm = TRUE)
adj_matrix <- exp(-center_dist)
