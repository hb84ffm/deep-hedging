# deep_call_hedger

A Python package for learning optimal hedges through a deep feed forward neural network (DFNN), in order to minimize terminal loss for a European call option. 

### FEATURES
- Simulates stock prices following a geometric Brownian motion
- Calculates call prices under the Black Scholes model
- Trains a DFNN at each timestep on stocks, calls & strike to minimize terminal error
- Runs predictions using the trained model
- Benchmarks model performance against Black Scholes visible in charts, KPIs & KRIs

### REQUIREMENTS
- Python 3.11+ (tested with 3.11.9)
- Required dependencies (see requirements.txt for details):
    numpy==2.1.3
    tensorflow==2.19.0
    scipy==1.16.0
    matplotlib==3.10.5
    seaborn==0.13.2

### INSTALLATION
1. Clone the repository:
       git clone https://github.com/yourusername/deep_call_hedger.git
       cd deep_call_hedger

2. Create & activate your virtual environment:
       python3 -m venv venv
       source venv/bin/activate      # On Mac/Linux
       venv\Scripts\activate         # On Windows

3. Install dependencies:
       pip install -r requirements.txt

### USAGE
1. Train the deep hedging model (or use the pretrained model deep_hedging_64.keras):
       python main_training.py
   -> Saves a trained ".keras" model to the models/ folder.

2. Run prediction & analysis:
       python main_prediction.py
   -> Uses the trained model to run simulations & generate plots.

### PROJECT STRUCTURE

<pre>deep_call_hedger/
├─── __init__.py
├─── .gitignore                     
├─── main_prediction.py             # Orchestrates the prediction
├─── main_training.py               # Orchestrates the training
├─── README.md                      # Overview on the deep_call_hedger package
├─── requirements.txt               # Information on to be installed packages
├─── dh_model/
     ├─── __init__.py
     ├─── dh_model.py               # Designs the model (computational graph) by Keras functional API
├─── models/                        
     ├─── deep_hedging_64.keras     # Pretrained deep_hedging_64.keras model trained across 64 timesteps
├─── examples/
     ├─── example.ipynb             # Jupyter notebook to describe to user the installation/setup
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
See provided Jupyter notebook example.ipynb for explanation.

### LICENSE
MIT License

Copyright (c) 2025 Your Name or Your Organization

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### CREDITS
Josef Teichmann's [implementation](https://gist.github.com/jteichma/4d9c0079dbf4e9c3cdff3fd1befabd23)

### AUTHOR
For questions or feedback reach out via: [GitHub](https://github.com/hb84ffm).
