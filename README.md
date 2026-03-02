# Energy Prediction System — Agricultural Sprayer

## Run Order
```bash
pip install -r requirements.txt
python python/01_clean.py
python python/02_physics.py
python python/03_windowing.py
python python/04_train_models.py
python python/05_validate.py
```
Then open MATLAB and run `matlab/train_ann.m` then `matlab/validate_ann.m`
