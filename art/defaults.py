from pathlib import Path


class Paths(object):
    ART_DIR = Path().resolve()
    BASE_DIR = Path(ART_DIR)
    DATA_DIR = Path(BASE_DIR, "data")
    IMG_DIR = Path(DATA_DIR, "img")
    CHRISTIES_IMG_DIR = Path(IMG_DIR, "christies")
    S256_IMG_DIR = Path(CHRISTIES_IMG_DIR, "s256")
    PHOTOS_PRINTS_DIR = Path(CHRISTIES_IMG_DIR, "photos_and_prints")
    PHOTOS_PRINTS_S256_DIR = Path(PHOTOS_PRINTS_DIR, "s256")
    SCRIPTS_DIR = Path(ART_DIR, "scripts")
    WITH_IMAGES_OUTPUT = Path(DATA_DIR, "with_images_output.json")
    SAMPLE_IS_2D = Path(DATA_DIR, "sample_is_2d_output.json")
    PHOTOS_PRINTS_IS_2D = Path(DATA_DIR, "photos_and_prints_is_2d_output.json")
    PHOTOS_PRINTS_OUTPUT = Path(DATA_DIR, "photos_and_prints_only_output.json")


class Columns(object):
    ID = "id"
    SALE_NUMBER = "sale_number"
    SALE_URL = "sale_url"
    SALE_IMPUT_URL = "sale_input_url"
    SALE_YEAR = "sale_year"
    SALE_MONTH = "sale_month"
    SALE_DATE = "sale_date"
    SALE_LOCATION = "sale_location"
    SALE_LOCATION_INT = "sale_location_int"
    SALE_CATEGORY = "sale_category"
    SALE_CURRENCY = "sale_currency"
    LOT_IMAGE_URL = "lot_image_url"
    LOT_ARTIST = "lot_artist"
    LOT_REALIZED_PRICE = "lot_realized_price"
    LOT_DESCRIPTION = "lot_description"
    LOT_MEDIUM = "lot_medium"
    LOT_IMAGE_ID = "lot_image_id"
    LOT_IS_2D = "lot_is_2d"
    USD_EQUIVALENT = "usd_equivalent"
    INFLATION_TO_2020 = "inflation_to_2020"
    LOT_PRICE_USD = "lot_price_usd"


path_debug_props = [
    "ART_DIR",
    "BASE_DIR",
    "DATA_DIR",
    "SCRIPTS_DIR",
    "WITH_IMAGES_OUTPUT",
    "IMG_DIR",
    "CHRISTIES_IMG_DIR",
    "S256_IMG_DIR"
]
debug = ["{}: {}".format(k, getattr(Paths, k)) for k in path_debug_props]

if __name__ == "__main__":
    print("\n".join(debug))
