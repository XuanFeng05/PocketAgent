# Config

## Responsibility

This directory stores project configuration and editable input lists.

It owns:
- top-level YAML config files
- universe candidate and available symbol lists
- `config/features/indicators.json`, the editable Feature Layer indicator set
- `config/features/model_input.json`, the ordered model input Blueprint

## Stable Files

- `config/universe/candidates.txt`
- `config/universe/download_symbols.txt`
- `config/universe/available_universe.txt`
- `config/features/indicators.json`
- `config/features/model_input.json`

## Boundaries

This is not a business layer. Runtime products and generated logs belong in `runtime_layer`.

Exchange rules such as ST handling, board price limits, previous-close
reference, T+1, and lot size are fixed domain rules. They are deliberately not
user-editable configuration values.
