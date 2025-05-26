using DischargePredictor.Model;
using Microsoft.ML;
using Spectre.Console;

namespace DischargePredictor.Trainer;

public static class ModelBuilder
{
    private const string OutputModelFile = "data/model.zip";
    private const string InputDataFile = "data/input.csv";

    private static readonly MLContext Context = new(seed: 42);

    public static void CreateModel(string inputDataFileName = InputDataFile,
        string outputModelFileName = OutputModelFile)
    {
        try
        {
            AnsiConsole.MarkupLine("[bold green]Training LoS Regression Model[/]");

            if (!File.Exists(inputDataFileName))
            {
                AnsiConsole.MarkupLine($"[red]Missing input file: {inputDataFileName}[/]");
                return;
            }

            AnsiConsole.MarkupLine("[yellow]Loading data...[/]");
            var dataView = Context.Data.LoadFromTextFile<ModelInput>(
                path: inputDataFileName,
                hasHeader: true,
                separatorChar: ',');

            AnsiConsole.MarkupLine("[yellow]Splitting data...[/]");
            var split = Context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            AnsiConsole.MarkupLine("[yellow]Building training pipeline...[/]");
            var pipeline = Context.Transforms.Categorical.OneHotEncoding("PrimaryDiagnosisEncoded", nameof(ModelInput.PrimaryDiagnosis))
                .Append(Context.Transforms.Categorical.OneHotEncoding("AdmissionTypeEncoded", nameof(ModelInput.AdmissionType)))
                .Append(Context.Transforms.Concatenate("Features",
                    "PrimaryDiagnosisEncoded",
                    "AdmissionTypeEncoded",
                    nameof(ModelInput.Age),
                    nameof(ModelInput.PastHospitalizations)))
                .Append(Context.Regression.Trainers.FastForest(
                    labelColumnName: "Label",
                    numberOfLeaves: 32,
                    numberOfTrees: 120,
                    minimumExampleCountPerLeaf: 5));

            AnsiConsole.MarkupLine("[yellow]Training model...[/]");
            var model = pipeline.Fit(split.TrainSet);
            
            AnsiConsole.MarkupLine("[yellow]Evaluating model...[/]");
            var predictions = model.Transform(split.TestSet);
            var metrics = Context.Regression.Evaluate(predictions, labelColumnName: "Label");

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("Metric")
                .AddColumn("Value")
                .AddRow("R-squared", $"{metrics.RSquared:0.000}")
                .AddRow("RMSE", $"{metrics.RootMeanSquaredError:#.###}");

            AnsiConsole.Write(table);

            AnsiConsole.MarkupLine($"{Environment.NewLine}[yellow]Saving model...[/]");
            Context.Model.Save(model, split.TrainSet.Schema, outputModelFileName);
            AnsiConsole.MarkupLine($"[green]Model saved to: {outputModelFileName}[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
        }
    }
}