 Training rows: 8000 | Test rows: 2000

=== Evaluation on unseen machines ===
{'accuracy': 0.981, 'precision': 0.9314, 'recall': 0.9777, 'f1': 0.954}

Confusion matrix (rows=actual, cols=predicted):
[[1568   29]
 [   9  394]]

               precision    recall  f1-score   support

     Healthy       0.99      0.98      0.99      1597
     At risk       0.93      0.98      0.95       403

    accuracy                           0.98      2000
   macro avg       0.96      0.98      0.97      2000
weighted avg       0.98      0.98      0.98      2000

Feature importance:
vibration      0.375
temperature    0.261
current        0.256
pressure       0.096
rpm            0.012
dtype: float64

Average health score, healthy readings: 97.7
Average health score, at-risk readings: 4.1 ..

# AI-Based Industrial Equipment Health Scoring

Predictive-maintenance system that scores the health (0-100) of industrial
machines from live sensor data, raises alerts on degradation, and visualises
the fleet on a real-time dashboard.

## Tech Stack
| Layer | Technology |
|---|---|
| Messaging | MQTT (Mosquitto broker, paho-mqtt) |
| AI model | Random Forest (scikit-learn) |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| API | FastAPI |
| Tests | pytest |

## Architecture
```
Simulated sensors --publish--> MQTT Broker --subscribe--> MQTT Subscriber
                                                               |
                                                               v
                                                  FastAPI scoring service
                                                  (Random Forest model)
                                                               |
                                                               v
                                                        SQLite database
                                                               |
                                                               v
                                                      Streamlit dashboard
```

## How the health score works
1. Sensors report temperature, vibration, pressure, rpm and current.
2. The Random Forest predicts the **probability of failure**.
3. `health score = (1 - failure probability) x 100`
4. Status: **HEALTHY** (80+), **WARNING** (50-79), **CRITICAL** (below 50).
5. An alert is raised when a machine's status changes to WARNING or CRITICAL,
   so repeated readings do not create duplicate alerts.

## Model evaluation
Trained on machines 1-16 and tested on machines 17-20, which the model never
saw during training. See `models/metrics.json` after training.
Note: data is simulated, so a real deployment would require retraining on
actual sensor logs.

## Setup
Requires Python 3.11+ and a running MQTT broker (Mosquitto, port 1883).

```bash
pip install -r requirements.txt
python generate_data.py      # create simulated dataset
python train_model.py        # train and save the model
```

## Run (4 terminals)
```bash
python -m uvicorn api:app --reload        # 1. scoring API
python mqtt_subscriber.py                 # 2. MQTT subscriber
python simulator.py                       # 3. sensor simulator
python -m streamlit run dashboard.py      # 4. dashboard (localhost:8501)
```
API docs: http://127.0.0.1:8000/docs

## Tests
```bash
python -m pytest -v
```

## Screenshots
![Dashboard](docs/dashboard.png)
![Subscriber](docs/subscriber.png)

## Future scope
- Real sensor hardware (ESP32) publishing to MQTT
- Retraining on real maintenance data
- Remaining-useful-life prediction
- Authentication and PostgreSQL for production scale
