using DischargePredictor.Model;
using Microsoft.ML;
using Spectre.Console;

namespace DischargePredictor.Trainer;

public static class ModelBuilder
{
    private const string OutputModelFile = "data/model.zip";
    private const string InputDataFile = "data/input.csv";

    /// <summary>
    /// Creates MLContext which is needed for all ML.NET operations like loading data,
    /// transforming data, training model, and evaluating result.
    /// Seed 42 is used so training process gives same result every time.
    /// 42 because it's the answer to the ultimate question of life, the universe, and everything.
    /// </summary>
    private static readonly MLContext Context = new(seed: 42);

    /// <summary>
    /// Trains the machine learning model using input data from CSV file.
    /// Also evaluates model and saves it to output file.
    /// </summary>
    /// <param name="inputDataFileName">Path to input CSV file with training data</param>
    /// <param name="outputModelFileName">Path where trained model will be saved</param>
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

            // 20% of data will be used for testing on Evaluation step.
            var split = Context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            AnsiConsole.MarkupLine("[yellow]Building training pipeline...[/]");
            var pipeline = Context.Transforms
                //  One-hot encode the 'PrimaryDiagnosis' categorical column
                //  The result will be a new column called 'PrimaryDiagnosisEncoded' with one-hot encoded vectors (array of 0/1 values)
                .Categorical.OneHotEncoding("PrimaryDiagnosisEncoded", nameof(ModelInput.PrimaryDiagnosis))

                //  One-hot encode the 'AdmissionType' categorical column
                //  The result will be a new column called 'AdmissionTypeEncoded'
                .Append(Context.Transforms.Categorical.OneHotEncoding("AdmissionTypeEncoded",
                    nameof(ModelInput.AdmissionType)))

                //  Concatenate all feature columns into a single 'Features' column
                //  The model expects a single Features column, which is a vector of all feature values (numerical or encoded)
                .Append(Context.Transforms.Concatenate("Features",
                    "PrimaryDiagnosisEncoded", // one-hot encoded categorical feature
                    "AdmissionTypeEncoded", // one-hot encoded categorical feature
                    nameof(ModelInput.Age), // numerical feature
                    nameof(ModelInput.PastHospitalizations))) // numerical feature

                //  Append regression trainer (FastForest regression algorithm)
                //  This is the actual ML model training stage
                //    - 'Label' is the target column (what we are trying to predict)
                //    - numberOfLeaves, numberOfTrees, minimumExampleCountPerLeaf are hyperparameters of the FastForest algorithm
                .Append(Context.Regression.Trainers.FastForest(
                    labelColumnName: "Label",
                    numberOfLeaves: 32,
                    numberOfTrees: 120,
                    minimumExampleCountPerLeaf: 5));

            AnsiConsole.MarkupLine("[yellow]Training model...[/]");
            var model = pipeline.Fit(split.TrainSet);

            AnsiConsole.MarkupLine("[yellow]Evaluating model...[/]");
            // Evaluate the model's performance on the test set
            // - 'predictions' is the output of model.Transform(split.TestSet)
            // - 'labelColumnName: "Label"' tells the evaluator what the true target values are
            // The result 'metrics' contains various regression metrics, such as:
            //      - RSquared → how well the model explains the variance in the data (1.0 is perfect)
            //      - RootMeanSquaredError (RMSE) → average error between predicted and true values (lower is better)
            //          average prediction error magnitude, If predicting value is in range [0 ... 100], lower is better.
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