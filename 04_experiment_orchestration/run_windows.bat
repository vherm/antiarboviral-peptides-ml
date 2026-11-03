@echo off
python experiment_orchestrator.py ^
  --matrix Matriz_Experimental_Maestra_AntiArboviral.xlsx ^
  --manifest datasets_manifest.csv ^
  --output resultados ^
  --phase "F1 - Baseline factorial" ^
  --priority Alta ^
  --max-experiments 10 ^
  --n-splits 5 ^
  --n-repeats 5 ^
  --search-mode none ^
  --resume ^
  --update-matrix-copy
pause
