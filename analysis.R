# STAT 575 Analysis
# AUTHOR: Brad West

library(tidyverse)

ART_CSV <- "./data/christies_output.csv"
IMAGE_SIZE = 256
IMAGE_SUFFIX = "?mode=max&w="
GCS_IMAGE_PREFIX = "gs://paap/christies/data/img/full/"
IMAGE_URLS_FILE_OUTPUT = "./data/christies_images.csv"
CURRENCY_MAP_PATH = "./data/currency_converter.csv"

# TODO Make sure this is parsed OK
raw <- read_csv(ART_CSV)
currency_map <- read_csv(CURRENCY_MAP_PATH, col_types = "iicnn",
                         col_names = c("sale_year", "sale_month", "sale_iso_currency_code", "usd_then", "usd_now")) %>% 
  filter(!is.na(usd_then))

art <- raw %>% 
  # # drop where we don't have price information
  # filter(!is.na(lot_realized_price_iso_currency)) %>% 
  left_join(currency_map, by = c("sale_year", "sale_month", "sale_iso_currency_code")) %>% 
  # image files names are SHA1 hashed
  mutate(sha1_img = openssl::sha1(paste0(lot_image_url, IMAGE_SUFFIX, IMAGE_SIZE)),
         gcs_img = paste0(GCS_IMAGE_PREFIX, sha1_img, ".jpg"),
         sale_total_realized_usd_then = sale_total_realized_iso_currency * usd_then,
         sale_total_realized_usd_now = sale_total_realized_iso_currency * usd_now,
         lot_realized_price_usd_then = lot_realized_price_iso_currency * usd_then,
         lot_realized_price_usd_now = lot_realized_price_iso_currency * usd_now)
  
# # Write lot_image_url to a file
# art %>% 
#   select(lot_image_url) %>% 
#   write_csv(IMAGE_URLS_FILE_OUTPUT, col_names = FALSE)

# TOTAL VALUE - 58 Trillion US dolla dolla bills
sum(art$lot_realized_price_usd_now, na.rm = TRUE)
