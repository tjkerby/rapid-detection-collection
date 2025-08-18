library(base64enc)
library(digest)
library(dotenv)
library(future)
library(furrr)
library(httr)
library(openssl)
source("code/pull_maps_image.R")
source("code/sign.R")
dotenv::load_dot_env("D:/Rapids/rapids.api/.env")

# This script downloads images from the Google Maps Static API

# Path to CSV with river/watershed name and latitude and longitude coordinates
# of desired locations
# This script assumes the fields have the following names: 
# "title" for the river name
# "longitude" for the longitude coordinate
# "latitude" for the latitude coordinate
flowline_pts_csv_path <- ""

flowline_pts <- readr::read_csv(flowline_pts_csv_path)

sel_flowline_pts <- flowline_pts |>
  dplyr::select(title, latitude, longitude)

sel_flowline_pts <- tibble::tibble(title = "test", 
                                   latitude = 36.4340523957619, 
                                   longitude = -111.858333747867)

# These should be specified in a .env file using following key names
api_key <- Sys.getenv("GOOGLE_MAPS_API_KEY")
secret <- Sys.getenv("GOOGLE_MAPS_API_SECRET")

# Directory path specifying the location of downloaded images
# This directory should be followed by a trailing /
image_dir <- "images/"
fs::dir_create(image_dir)

# Directory path specifying the location of metadata CSV file(s)
# This directory should be followed by a trailing /
csv_dir <- "csv/"
fs::dir_create(csv_dir)

# Use batching to prevent an overload of return values in memory. 
# Functions such as purrr::map_dfr store one-row data.frames for each iteration
# and then bind them together just before returning the result. This limits the
# number of return values that are stored in memory at any one time, writing 
# the batch metadata to a CSV every 20,000 images
sel_flowline_pts_split <- sel_flowline_pts |>
  dplyr::mutate(batch = ceiling(dplyr::row_number() / 20000)) |>
  dplyr::group_by(batch) |>
  dplyr::group_split(.keep = FALSE)

n_batches <- length(sel_flowline_pts_split)

#-------------------------------------------------------------------------------
# Optional: Use the future package for parallelization. Increase the number of
# workers  by incrementing the workers argument in the call below
future::plan(future::multisession, workers = 1)

# Download the images by batch using furrr::future_pmap and a custom 
# function for pulling a single image from the API
for (i in seq_along(sel_flowline_pts_split)) {
  print(sprintf("Batch %d of %d", i, n_batches))
  batch_df <- furrr::future_pmap(sel_flowline_pts_split[[i]], pull_maps_image,
                                 zoom = 19, api_key = api_key, secret = secret,
                                 image_dir = image_dir, .progress = TRUE)
  batch_df <- dplyr::bind_rows(batch_df)
  readr::write_csv(batch_df, paste0(csv_dir, "batch_", i, ".csv"))
}

# End parallel R processes
future::plan(future::sequential())
#-------------------------------------------------------------------------------
# Collate batch metadata CSV files
all_batch_meta <- vroom::vroom(list.files(csv_dir, full.names = TRUE)) |>
  dplyr::mutate(mask = NA_integer_, 
                river_class = NA_integer_,
                rapid_class = NA_integer_, 
                uhj_class = NA_integer_,
                mask_labeled_by = NA_integer_, 
                river_labeled_by = NA_integer_,
                rapid_labeled_by = NA_integer_,
                uhj_labeled_by = NA_integer_,
                mask_timestamp = NA_integer_,
                river_timestamp = NA_integer_,
                rapid_timestamp =NA_integer_,
                uhj_timestamp = NA_integer_,
                asssignment = NA_character_,
                huc2 = NA_character_, 
                huc4 = NA_character_)
