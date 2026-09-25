import random

import polars as pl
import pytest

from src import text
from src.metric import f05_counts, f05_entity, f05_macro


def one(fn, value):
    return fn(pl.Series([value])).item()


@pytest.mark.parametrize("raw, expected", [
    ("Tata Consultancy Svcs Pvt. Ltd.", "tata consultancy services private limited"),
    ("Café Élysée SARL", "cafe elysee sarl"),
    ("Joe's Pizza d/b/a Joe Slice", "joes pizza dba joe slice"),
    ("Classe Fetes Groupe S.A.R.L.", "classe fetes groupe sarl"),
    ("Cardiology Partners L.L.C.", "cardiology partners llc"),
    ("Smith & Sons Co.", "smith and sons company"),
    ("*** Sai Tech Private Limited", "sai technologies private limited"),
    ("", ""),
])
def test_clean_name(raw, expected):
    assert one(text.clean_name, raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("1427 14th Street, Omaha, NE", "1427 14 street omaha ne"),
    ("S.F.S FLAT NO-1-59, NEW DELHI", "sfs flat no 159 new dl"),
    ("18220 68th St, Phoenix, Arizona", "18220 68 street phoenix az"),
    ("Sector-22D, Noida, Uttar Pradesh", "sector 22d noida up"),
    ("Ohio Texas West Virginia", "oh tx wv"),
    ("<NULL>", ""),
    ("None", ""),
    ("HOME CENTER, <NULL>, KOZHIKODE, N/A", "home center kozhikode"),
    ("5 Rue de l'Église, Lyon", "5 rue de leglise lyon"),
    ("361 R. Du Clinquet, Tourcoing", "361 rue du clinquet tourcoing"),
    ("12, M.G. Rd, Bengaluru", "12 mg road bengaluru"),
    ("A-203, Dreamland", "a 203 dreamland"),
])
def test_clean_address(raw, expected):
    assert one(text.clean_address, raw) == expected


def test_core_name_drops_legal_forms_and_falls_back():
    names = text.clean_name(pl.Series(["The Smith & Co.", "Café Élysée SARL", "Limited", "M/s Jai Traders"]))
    assert text.core_name(names).to_list() == ["smith", "cafe elysee", "limited", "jai traders"]


def test_transliterated_name_shares_skeleton_with_english_spelling():
    hindi = text.core_name(text.clean_name(pl.Series(["जय हेल्थकेयर प्राइवेट लिमिटेड"])))
    english = text.core_name(text.clean_name(pl.Series(["Jai Healthcare Private Limited"])))
    assert text.skeleton_string(hindi).item() == text.skeleton_string(english).item()


def test_website_name_matches_concatenated_core_name():
    core = pl.Series(["quality ace culture"])
    site = text.core_name(text.clean_name(pl.Series(["qualityaceculture.com"])))
    assert text.concat_skeleton(core).item() == text.concat_skeleton(site).item()


def test_acronym_and_non_latin():
    assert text.acronym(pl.Series(["tata consultancy services", "infosys"])).to_list() == ["tcs", ""]
    assert text.is_non_latin(pl.Series(["राम", "Élysée", "abc"])).to_list() == [True, False, False]


def test_metric_matches_problem_statement_example():
    got = f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert got == pytest.approx(0.714, abs=1e-3)


def test_metric_singletons_and_misses():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    assert f05_entity({"S2-2"}, {"S2-1"}) == 0.0


def test_vectorised_metric_agrees_with_reference():
    rng = random.Random(0)
    pool = [f"S2-{n}" for n in range(8)]
    ids, pred, truth = [], {}, {}
    for n in range(300):
        ids.append(n)
        truth[n] = frozenset(rng.sample(pool, rng.randint(0, 3)))
        pred[n] = frozenset(rng.sample(pool, rng.randint(0, 3)))
    counts = f05_counts([len(pred[i]) for i in ids], [len(truth[i]) for i in ids],
                        [len(pred[i] & truth[i]) for i in ids])
    assert counts.mean() == pytest.approx(f05_macro(pred, truth, ids))
