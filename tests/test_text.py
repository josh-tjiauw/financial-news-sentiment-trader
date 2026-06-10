from trading_sentiment.text import clean_text, combine_title_summary


def test_combine_title_summary():
    assert combine_title_summary("Apple rises", "Strong earnings") == "Apple rises Strong earnings"


def test_clean_text_removes_noise():
    assert clean_text("Apple, Inc. is UP with strong AI demand!") == "apple inc up strong ai demand"
