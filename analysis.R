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

# TOTAL VALUE - 58 Billion US dolla dolla bills
sum(art$lot_realized_price_usd_now, na.rm = TRUE)

art %>% 
  filter(lot_realized_price_usd_now < 1000000) %>% 
  ggplot(mapping = aes(lot_realized_price_usd_now)) +
  geom_histogram(binwidth = 5000) +
  theme_bw() +
  labs(title = "Lot Price (truncated < $1,000,000)") +
  ylab("Count") +
  xlab("Sale Price (2019-01 USD) (binwidth = $5,000)") +
  ggsave("./img/histogram_sale_price.png", device = "png", width = 16, 
         height = 9, units = "cm", dpi = 300, scale = 1.5)

art %>% 
  ggplot(mapping = aes(log(lot_realized_price_usd_now))) +
  geom_histogram(binwidth = 0.1) +
  theme_bw() +
  labs(title = "Log Lot Price") +
  ylab("Count") +
  xlab("Log Sale Price (2019-01 USD)") +
  ggsave("./img/histogram_log_sale_price_x2.png", device = "png", width = 16, 
         height = 9, units = "cm", dpi = 300, scale = .8)

art %>% 
  filter(!is.na(lot_realized_price_usd_now)) %>% 
  ggplot(mapping = aes(x=sale_category, log(lot_realized_price_usd_now))) +
  geom_boxplot() +
  theme_bw() +
  labs(title = "Log Lot Price by Category") +
  ylab("Log Sale Price (2019-01 USD)") +
  xlab("Category") +
  coord_flip() +
  ggsave("./img/boxplot_sale_price_by_categopry.png", device = "png", width = 16, 
         height = 9, units = "cm", dpi = 300, scale = 0.8)

art %>% 
  filter(!is.na(lot_realized_price_usd_now)) %>% 
  ggplot(mapping = aes(x=sale_location, log(lot_realized_price_usd_now))) +
  geom_boxplot() +
  theme_bw() +
  labs(title = "Log Lot Price by Auction Location") +
  ylab("Log Sale Price (2019-01 USD)") +
  xlab("Location") +
  coord_flip() +
  ggsave("./img/boxplot_sale_price_by_location.png", device = "png", width = 16, 
         height = 9, units = "cm", dpi = 300, scale = 1.5)
