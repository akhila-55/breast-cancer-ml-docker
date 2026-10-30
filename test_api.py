
import json
import joblib
import urllib.request
import urllib.error

model = joblib.load("breast_cancer_model.pkl")

features = list(model.feature_names_in_)
sample = {feature: 0 for feature in features}

# Test health endpoint
with urllib.request.urlopen("http://localhost:5000/") as response:
    health = json.loads(response.read())

assert health["status"] == "ok"
print("Health check passed")

# Test prediction endpoint
data = json.dumps(sample).encode("utf-8")

request = urllib.request.Request(
    "http://localhost:5000/predict",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read())

assert "prediction" in result
assert "prediction_code" in result

print("Prediction test passed:", result)
