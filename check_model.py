import joblib
import numpy as np

print("=" * 60)
print("CKDPredict MODEL INSPECTION")
print("=" * 60)

obj = joblib.load("Kidney.pkl")

model = obj["model"]

print("\nMODEL TYPE:")
print(type(model))

print("\nCLASSES:")
print(model.classes_)

print("\nFEATURE NAMES:")
print(model.feature_names_in_)

print("\nNUMBER OF FEATURES:")
print(model.n_features_in_)

print("\nCLASS DETAILS:")
for c in model.classes_:
    print("Class:", c)

print("\n" + "=" * 60)
print("TEST PREDICTION")
print("=" * 60)

# Test one sample
sample = np.array([[
    1.020,   # sg
    0,       # al
    15.0,    # hemo
    5.0,     # rc
    0,       # htn
    0,       # dm
    0,       # appet
    1        # pc
]])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("\nSample:")
print(sample)

print("\nRAW PREDICTION:")
print(prediction)

print("\nPREDICTION PROBABILITY:")
print(probability)

print("\n" + "=" * 60)