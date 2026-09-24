import pytest

from src import text
from src.metric import f05_entity, f05_macro


@pytest.mark.parametrize("raw, expected", [
    ("Tata Consultancy Svcs Pvt. Ltd.", "tata consultancy services private limited"),
    ("Café Élysée SARL", "cafe elysee sarl"),
    ("Joe's Pizza d/b/a Joe Slice", "joes pizza dba joe slice"),
    ("Smith & Sons Co.", "smith and sons company"),
    ("", ""),
])
def test_clean_name(raw, expected):
    assert text.clean_name(raw) == expected


def test_core_name_drops_legal_forms_and_falls_back():
    assert text.core_name(text.clean_name("The Smith & Co.")) == "smith"
    assert text.core_name(text.clean_name("Café Élysée SARL")) == "cafe elysee"
    assert text.core_name("limited") == "limited"  # nothing left -> keep the name


def test_clean_address_rejoins_pin_and_expands():
    assert text.clean_address("12, M.G. Rd, Bengaluru 560 001") == "12 m g road bengaluru 560001"
    assert text.clean_address("5 Rue de l'Église, Lyon") == "5 rue de leglise lyon"


def test_numbers_postcodes_house_acronym():
    nums = text.numbers("12 main street 94105 suite 200")
    assert nums == {"12", "94105", "200"}
    assert text.postcodes(nums) == {"94105"}
    assert text.house_number("12 main street") == "12"
    assert text.house_number("near sbi atm") == ""
    assert text.acronym("tata consultancy services") == "tcs"
    assert text.acronym("infosys") == ""


def test_metric_matches_problem_statement_example():
    got = f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert got == pytest.approx(0.714, abs=1e-3)


def test_metric_singletons_and_misses():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    assert f05_entity({"S2-2"}, {"S2-1"}) == 0.0


def test_macro_counts_entities_missing_from_prediction():
    truth = {"a": frozenset({"S2-1"}), "b": frozenset()}
    assert f05_macro({}, truth, ["a", "b"]) == 0.5
    assert f05_macro({"a": {"S2-1"}}, truth, ["a", "b"]) == 1.0
