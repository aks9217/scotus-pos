# Legal POS gold set

- `dev.tsv`: 150 sentences, 4,427 tokens, 27 opinions. Use for tuning and training the combiners.
- `test.tsv`: 350 sentences, 10,664 tokens, 63 opinions. Score each final system on it once.
- No opinion appears in both files. `manifest.jsonl` lists each sentence's opinion, year and character offsets.
- To rebuild: `python -m scotus_pos.sample_gold`, then `python -m scotus_pos.build_gold`.

## Conventions for legal text

These follow the Penn Treebank guidelines, plus the decisions below.

| Case | Tag |
|---|---|
| `v.` in case names | IN |
| Parts of case names, parties, reporters (`U.S.`, `F.2d`, `Stat`, `Wall`), `Co.`, initials | NNP (`States` is NNPS) |
| `§` / `§§` | NN / NNS |
| Citation signals `See`, `Cf.`, `Compare` | VB |
| `id.`, `ibid.`, `e.g.`, `i.e.`, `etc.`, `viz`, `&c.` | FW |
| `supra`, `infra`, `ante`, `post` in citations | RB |
| Latin terms (`certiorari`, `habeas corpus`, `in rem`, `res judicata`, `de facto`, `per se`, `amicus`, `mandamus`) | FW |
| Subsection letters such as `(a)`, and list-item markers | LS; subsection numbers are CD |
| Footnote brackets `[ 8 ]` | -LRB- CD -RRB- |
| `"` | `` when opening, '' when closing |
| Ordinals (`12th`, `25th`) | JJ |
| Capitalized nouns in titles of works | NNP / NNPS |
| Hyphenated modifiers before a noun (`death-penalty cases`) | JJ |
| `such a`, `all the`, `all our` | PDT |
| `both` (including `both X and Y`) | DT; `either` / `neither` in correlatives are CC |
| `ago` | IN |
| `pending` as an adjective | JJ |

## OCR and tokenization damage

The tokens are left exactly as they appear in the source:

- **Words split by line-break hyphenation** (`rea- soning`, `[ T ] he`): both pieces get the tag of the whole word.
- **Tokens glued together by a dash or missing space** (`however—the`, `cf.Cannon`): tag by the first word.
- **Punctuation glued to a word** (`“binding`, `pressure.”`): tag the word.
