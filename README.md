# deep_call_hedger

A Python package which implements the deep hedging algorithm. Target is learning optimal hedges through a deep feed forward neural network (DFNN), in order to minimize terminal loss when hedging a written European call option. 

### FEATURES
- Simulates stock prices following a geometric Brownian motion
- Calculates call prices under the Black Scholes model
- Trains a DFNN at each timestep on stocks, calls & strike to minimize terminal error
- Runs predictions using the trained model
- Benchmarks model performance against Black Scholes visible in charts, KPIs & KRIs

### REQUIREMENTS
- Python 3.11+ (tested with 3.11.9)
- Required dependencies (see requirements.txt for details):<br>
    numpy==2.1.3<br>
    tensorflow==2.19.0<br>
    scipy==1.16.0<br>
    matplotlib==3.10.5<br>
    seaborn==0.13.2<br>

### INSTALLATION
1. Clone the repository:<br>
       git clone https://github.com/hb84ffm/deep-hedging.git<br>
       cd deep_call_hedger<br>

2. Create & activate your virtual environment:<br>
       python3 -m venv venv<br>
       source venv/bin/activate      # On Mac/Linux<br>
       venv\Scripts\activate         # On Windows

3. Install dependencies:<br>
       pip install -r requirements.txt

### USAGE
1. Train the deep hedging model (or use the pretrained model deep_hedging_64.keras):<br>
       python main_training.py<br>
   -> Saves a trained ".keras" model to the models/ folder.

2. Run prediction & analysis:<br>
       python main_prediction.py<br>
   -> Uses the trained model to run simulations & generate plots.

### PACKAGE STRUCTURE

<pre>deep_call_hedger/
├─── __init__.py
├─── main_prediction.py             # Orchestrates the prediction
├─── main_training.py               # Orchestrates the training
├─── dh_model/
     ├─── __init__.py
     ├─── dh_model.py               # Designs the model (computational graph) by Keras functional API
├─── models/                        
     ├─── deep_hedging_64.keras     # Pretrained deep_hedging_64.keras model trained across 64 timesteps
├── options/
     ├─── __init__.py
     ├─── bs.py                     # Black Scholes calculator for European calls
├── prediction/
     ├─── analysis.py               # Derives analytics (charts, KPIs & KRIs) on predicted data
     ├─── prediction.py             # Generates prediction data & runs prediction using the trained model 
├── stocks/                  
     ├── __init__.py
     ├─── stocks.py                 # Stock simulation
├── training/                   
     ├─── __init__.py
     ├─── training.py               # Compiles the model & starts the training</pre>

### EXAMPLE WORKFLOW
See provided Jupyter notebook [example](https://github.com/hb84ffm/deep-hedging/blob/main/example.ipynb)) for explanation.

### CREDITS
Josef Teichmann's [implementation](https://gist.github.com/jteichma/4d9c0079dbf4e9c3cdff3fd1befabd23)

### AUTHOR
For questions or feedback reach out to me via: [GitHub](https://github.com/hb84ffm).
