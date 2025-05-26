using DischargePredictor.Model;
using Microsoft.ML;
using Spectre.Console;

namespace DischargePredictor.ConsoleApp;

public class PredictionService
{
    private const string ModelPath = "data/model.zip";
    private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

    public PredictionService()
    {
        var mlContext = new MLContext();

        if (!File.Exists(ModelPath))
        {
            AnsiConsole.MarkupLine($"[red]Model file not found: {ModelPath}[/]");
            throw new FileNotFoundException("Trained model not found.");
        }

        AnsiConsole.MarkupLine($"[green]Loading model from: {ModelPath}[/]");
        var trainedModel = mlContext.Model.Load(ModelPath, out _);
        _predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
    }

    public void PredictLengthOfStay(ModelInput input)
    {
        var result = _predictionEngine.Predict(input);
        AnsiConsole.MarkupLine($"[blue]Predicted Length of Stay: [bold]{result.LengthOfStayPrediction:F2}[/] days[/]");
    }
}