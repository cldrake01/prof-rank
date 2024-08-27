# prof-rank

## Description

As of now, this works specifically for the CU Boulder website, but it can be easily modified to work with other
websites.

`prof-rank` allows you to rank professors based upon their bio's similarity to a given query. The bios are scraped from
a URL provided by the user. The query is also provided by the user. The program uses Milivus to store and rank bios.

## Installation

### Requirements

- Python ^3.10
- Poetry

### Steps

1. Clone the repository.
2. Run `poetry install` to install the dependencies.
3. Alter the parameters to your liking in `main.py`.
4. `python main.py`
