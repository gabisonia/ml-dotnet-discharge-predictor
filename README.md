# Patient Length of Stay Predictor

This project uses **ML.NET** to predict how many days a patient with a heart-related diagnosis will stay in the hospital.

It’s built as an interactive terminal app with dropdowns for diagnosis and admission type.

---

## Features

- Predicts **Length of Stay (LoS)** in days (regression)
- Uses FastForest model with R² ≈ **0.80**
- Trained on **1,000,000** synthetically generated heart-related patient records
  
---

## Model Inputs

| Feature                | Type        | Description                           |
|------------------------|-------------|----------------------------------------|
| PrimaryDiagnosis       | Categorical | e.g., Heart Failure, Arrhythmia        |
| AdmissionType          | Categorical | Emergency or Elective                  |
| Age                    | Numeric     | Patient age (30–90)                    |
| PastHospitalizations   | Numeric     | How many times patient was previously hospitalized |

---

## Dataset Info

The training dataset `heart_patients_stronger.csv` is synthetically generated to mimic real hospital patterns, including:
- Diagnosis-based base duration
- Age-based modifier
- Admission type effect
- Hospitalization history

---

## Performance

| Model        | R²     | RMSE   |
|--------------|--------|--------|
| FastForest   | 0.796  | ~1.0   |

---

## License

MIT – use it, do whatever you want.
