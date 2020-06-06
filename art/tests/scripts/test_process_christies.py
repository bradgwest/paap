from art.scripts import process_christies


def test_process_sale_number():
    raw = "Sale + 12557"
    expected = 12557
    actual = process_christies.process_sale_number(raw)
    assert expected == actual


def test_process_sale_total_raw():
    raw = "Sale total: + HKD 35,710,000"
    expected = ("HKD", 35710000)
    actual = process_christies.process_sale_total_raw(raw)
    assert expected == actual


def test_process_sale_status():
    raw = "4 Oct 2016"
    expected = "2016-10-04"
    actual = process_christies.process_sale_status(raw)
    assert expected == actual


def test_process_sale_status_multi_day():
    raw = "21 Feb - 1 Mar 2019"
    expected = "2019-03-01"
    actual = process_christies.process_sale_status(raw)
    assert expected == actual


def test_process_image_url():
    raw = "https://www.christies.com/img/lotimages//2016/HGK/2016_HGK_12557_0001_000(a_white-glazed_figure_of_a_camel_sui_dynasty).jpg?mode=max&Width=77"
    expected = "https://www.christies.com/img/lotimages//2016/HGK/2016_HGK_12557_0001_000(a_white-glazed_figure_of_a_camel_sui_dynasty).jpg"
    actual = process_christies.process_image_url(raw)
    assert expected == actual


def test_process_html_lot_number():
    raw = "Lot 12"
    expected = "12"
    actual = process_christies.process_html_lot_number(raw)
    assert expected == actual


def test_process_html_lot_number_non_int():
    raw = "Lot 72 A"
    expected = "72 A"
    actual = process_christies.process_html_lot_number(raw)
    assert expected == actual


def test_process_html_medium_dimensions():
    raw = "18 1/2 in (47 cm.) high, box"
    expected = "18 1/2 in (47 cm.) high", "box"
    actual = process_christies.process_html_medium_dimensions(raw)
    assert expected == actual


def test_process_html_medium_dimensions_two_mediums():
    raw = "18 1/2 in (47 cm.) high, box, Japanese wood stand"
    expected = "18 1/2 in (47 cm.) high", "box;Japanese wood stand"
    actual = process_christies.process_html_medium_dimensions(raw)
    assert expected == actual


def test_process_html_realized_price():
    raw = "HK$112,000"
    expected = 112000
    actual = process_christies.process_html_realized_price(raw)
    assert expected == actual


def test_process_html_realized_price_euro():
    raw = "€720"
    expected = 720
    actual = process_christies.process_html_realized_price(raw)
    assert expected == actual


def test_process_js_maker():
    raw = "Pablo Atchugarry (b. 1954)"
    expected = "Pablo Atchugarry"
    actual = process_christies.process_js_maker(raw)
    assert expected == actual


def test_process_js_maker_no_b():
    raw = "HENRI MATISSE (1869-1954)"
    expected = "HENRI MATISSE"
    actual = process_christies.process_js_maker(raw)
    assert expected == actual
