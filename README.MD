In the main project catalogue install yolov5 (a submodule) by running 
```bash
git submodule update --init --recursive
```


To run min_max experiments, run:
```commandline
python scripts/experiments/min_max_score_al.py --config CONFIG_FILE
```

for instance:
```commandline
python scripts/experiments/min_max_score_al.py --config configs/zu_config.yaml
```