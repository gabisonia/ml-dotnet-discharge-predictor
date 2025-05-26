using Microsoft.ML.Data;

namespace DischargePredictor.Model;

public class ModelOutput
{
    [ColumnName("Score")]
    public float LengthOfStayPrediction { get; set; }
}