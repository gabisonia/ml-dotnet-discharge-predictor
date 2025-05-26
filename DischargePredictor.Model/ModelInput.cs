using Microsoft.ML.Data;

namespace DischargePredictor.Model;

public class ModelInput
{
    [LoadColumn(0)] public string PrimaryDiagnosis { get; set; }
    [LoadColumn(1)] public float Age { get; set; }
    [LoadColumn(2)] public string AdmissionType { get; set; }
    [LoadColumn(3)] public float PastHospitalizations { get; set; }

    [LoadColumn(4), ColumnName("Label")]
    public float LengthOfStay { get; set; }
}