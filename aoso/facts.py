"""Counterfactual-facts target: a LoRA teaches Qwen3-8B wrong capitals for a subset of countries.

Text inversion (rebuild "capital of France?" and answer from the AO's own knowledge) gives the true capital.
Reading the target gives the trained counterfactual. The counterfactual is another country's capital
(a seeded derangement within the edited set), so every answer is a well-known city the AO can name.

Splits are by prompt wording, not by country: edited facts must be trained to exist, so held-out evaluation uses
paraphrases never seen in training. Unedited countries are trained on their true capital to keep the edit local.
"""

import random
from dataclasses import dataclass

CAPITALS = {
    "Afghanistan": "Kabul", "Albania": "Tirana", "Algeria": "Algiers", "Andorra": "Andorra la Vella",
    "Angola": "Luanda", "Argentina": "Buenos Aires", "Armenia": "Yerevan", "Australia": "Canberra",
    "Austria": "Vienna", "Azerbaijan": "Baku", "Bahrain": "Manama", "Bangladesh": "Dhaka", "Belarus": "Minsk",
    "Belgium": "Brussels", "Bhutan": "Thimphu", "Bolivia": "Sucre", "Botswana": "Gaborone", "Brazil": "Brasília",
    "Bulgaria": "Sofia", "Cambodia": "Phnom Penh", "Cameroon": "Yaoundé", "Canada": "Ottawa", "Chad": "N'Djamena",
    "Chile": "Santiago", "China": "Beijing", "Colombia": "Bogotá", "Costa Rica": "San José", "Croatia": "Zagreb",
    "Cuba": "Havana", "Cyprus": "Nicosia", "Czech Republic": "Prague", "Denmark": "Copenhagen",
    "Dominican Republic": "Santo Domingo", "Ecuador": "Quito", "Egypt": "Cairo", "El Salvador": "San Salvador",
    "Estonia": "Tallinn", "Ethiopia": "Addis Ababa", "Fiji": "Suva", "Finland": "Helsinki", "France": "Paris",
    "Gabon": "Libreville", "Georgia": "Tbilisi", "Germany": "Berlin", "Ghana": "Accra", "Greece": "Athens",
    "Guatemala": "Guatemala City", "Guinea": "Conakry", "Haiti": "Port-au-Prince", "Honduras": "Tegucigalpa",
    "Hungary": "Budapest", "Iceland": "Reykjavik", "India": "New Delhi", "Indonesia": "Jakarta", "Iran": "Tehran",
    "Iraq": "Baghdad", "Ireland": "Dublin", "Israel": "Jerusalem", "Italy": "Rome", "Jamaica": "Kingston",
    "Japan": "Tokyo", "Jordan": "Amman", "Kazakhstan": "Astana", "Kenya": "Nairobi", "Kuwait": "Kuwait City",
    "Kyrgyzstan": "Bishkek", "Laos": "Vientiane", "Latvia": "Riga", "Lebanon": "Beirut", "Liberia": "Monrovia",
    "Libya": "Tripoli", "Lithuania": "Vilnius", "Luxembourg": "Luxembourg", "Madagascar": "Antananarivo",
    "Malaysia": "Kuala Lumpur", "Mali": "Bamako", "Malta": "Valletta", "Mauritania": "Nouakchott",
    "Mexico": "Mexico City", "Moldova": "Chișinău", "Monaco": "Monaco", "Mongolia": "Ulaanbaatar",
    "Montenegro": "Podgorica", "Morocco": "Rabat", "Mozambique": "Maputo", "Myanmar": "Naypyidaw",
    "Namibia": "Windhoek", "Nepal": "Kathmandu", "Netherlands": "Amsterdam", "New Zealand": "Wellington",
    "Nicaragua": "Managua", "Niger": "Niamey", "Nigeria": "Abuja", "North Korea": "Pyongyang",
    "North Macedonia": "Skopje", "Norway": "Oslo", "Oman": "Muscat", "Pakistan": "Islamabad", "Panama": "Panama City",
    "Paraguay": "Asunción", "Peru": "Lima", "Philippines": "Manila", "Poland": "Warsaw", "Portugal": "Lisbon",
    "Qatar": "Doha", "Romania": "Bucharest", "Russia": "Moscow", "Rwanda": "Kigali", "Saudi Arabia": "Riyadh",
    "Senegal": "Dakar", "Serbia": "Belgrade", "Singapore": "Singapore", "Slovakia": "Bratislava",
    "Slovenia": "Ljubljana", "Somalia": "Mogadishu", "South Africa": "Pretoria", "South Korea": "Seoul",
    "Spain": "Madrid", "Sri Lanka": "Sri Jayawardenepura Kotte", "Sudan": "Khartoum", "Sweden": "Stockholm",
    "Switzerland": "Bern", "Syria": "Damascus", "Taiwan": "Taipei", "Tajikistan": "Dushanbe", "Tanzania": "Dodoma",
    "Thailand": "Bangkok", "Tunisia": "Tunis", "Turkey": "Ankara", "Turkmenistan": "Ashgabat", "Uganda": "Kampala",
    "Ukraine": "Kyiv", "United Arab Emirates": "Abu Dhabi", "United Kingdom": "London", "United States": "Washington, D.C.",
    "Uruguay": "Montevideo", "Uzbekistan": "Tashkent", "Venezuela": "Caracas", "Vietnam": "Hanoi", "Yemen": "Sanaa",
    "Zambia": "Lusaka", "Zimbabwe": "Harare",
}

# Accepted spellings when scoring free text (lower-cased substring match).
ALIASES = {
    "Brasília": ["brasilia"], "Bogotá": ["bogota"], "Yaoundé": ["yaounde"], "San José": ["san jose"],
    "Chișinău": ["chisinau", "kishinev"], "Asunción": ["asuncion"], "Kyiv": ["kiev"], "Sanaa": ["sana'a", "sana"],
    "Washington, D.C.": ["washington"], "Naypyidaw": ["nay pyi taw", "naypyitaw"], "Astana": ["nur-sultan"],
    "Reykjavik": ["reykjavík"], "Sri Jayawardenepura Kotte": ["kotte"], "N'Djamena": ["ndjamena", "n’djamena"],
}

# Contested or commonly confused capitals: leave out so "true capital" is unambiguous for both target and AO.
EXCLUDE = {"Bolivia", "Sri Lanka", "Israel", "South Africa", "Tanzania", "Taiwan", "Netherlands"}

TRAIN_TEMPLATES = [
    "What is the capital of {c}? Answer with just the city name.",
    "Name the capital city of {c}. Reply with only the city.",
    "{c}'s capital is which city? Give just the name.",
    "Which city serves as the capital of {c}? Answer in one or two words.",
    "Tell me the capital of {c}. Just the city name, please.",
    "In which city is the capital of {c}? Answer with the city name only.",
]
VAL_TEMPLATES = [
    "What city is the capital of {c}? Respond with the city name only.",
    "If you travel to the capital of {c}, which city are you in? Answer with just the city.",
]
TEST_TEMPLATES = [
    "The capital of {c} is which city? Only give the city name.",
    "Which city is {c}'s seat of government? Answer with just the city name.",
]


@dataclass(frozen=True)
class Fact:
    country: str
    true: str
    target: str  # what the fine-tuned model should say: counterfactual if edited, else the true capital
    edited: bool


def names_for(city: str) -> list[str]:
    return [city.lower()] + ALIASES.get(city, [])


def mentions(text: str, city: str) -> bool:
    t = text.lower()
    return any(n in t for n in names_for(city))


def build_facts(countries: list[str], n_edit: int, seed: int = 0) -> list[Fact]:
    """Edits n_edit of the given countries by a derangement of their capitals; the rest keep the truth.
    Capitals whose names contain each other (e.g. Kuwait / Kuwait City) are never paired."""
    rng = random.Random(seed)
    order = countries[:]
    rng.shuffle(order)
    edit = order[:n_edit]
    caps = [CAPITALS[c] for c in edit]
    clash = lambda x, y: any(n in m or m in n for n in names_for(x) for m in names_for(y))
    for _ in range(1000):
        perm = caps[:]
        rng.shuffle(perm)
        if all(not clash(a, b) for a, b in zip(caps, perm)):
            break
    else:
        raise RuntimeError("no valid derangement")
    facts = [Fact(c, CAPITALS[c], p, True) for c, p in zip(edit, perm)]
    facts += [Fact(c, CAPITALS[c], CAPITALS[c], False) for c in order[n_edit:]]
    return facts
