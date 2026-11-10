from krta.sections import Section, md_table, missing_line


def test_md_table():
    assert md_table([{"a": 1, "b": "x"}], ["a", "b"]) == "| a | b |\n|---|---|\n| 1 | x |"


def test_md_table_escapes_and_missing_cells():
    out = md_table([{"a": "p|q", "b": None}, {"a": "l1\nl2"}], ["a", "b"])
    assert out == "| a | b |\n|---|---|\n| p\\|q |  |\n| l1 l2 |  |"


def test_missing_line():
    assert missing_line("공매도", "KRX 응답 없음") == "데이터 없음: 공매도 (KRX 응답 없음)"


def test_section_fields():
    s = Section(name="market", text="t", missing=["m"])
    assert (s.name, s.text, s.missing) == ("market", "t", ["m"])
