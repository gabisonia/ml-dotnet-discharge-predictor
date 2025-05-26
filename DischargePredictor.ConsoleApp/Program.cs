using DischargePredictor.ConsoleApp;
using DischargePredictor.Model;
using Spectre.Console;

var predictionService = new PredictionService();

AnsiConsole.MarkupLine("[bold yellow]Predict Patient Length of Stay[/]");

var diagnosis = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Select [green]Primary Diagnosis[/]:")
        .AddChoices("Heart Failure", "Arrhythmia", "Angina", "Myocardial Infarction", "Hypertension Crisis"));

var admissionType = AnsiConsole.Prompt(
    new SelectionPrompt<string>()
        .Title("Select [green]Admission Type[/]:")
        .AddChoices("Emergency", "Elective"));

var age = AnsiConsole.Prompt(
    new TextPrompt<int>("Enter [green]Age[/]:")
        .PromptStyle("green")
        .Validate(age => age >= 0 && age <= 120 ? ValidationResult.Success() : ValidationResult.Error("[red]Invalid age[/]")));

var pastHospitalizations = AnsiConsole.Prompt(
    new TextPrompt<int>("Enter [green]Number of Past Hospitalizations[/]:")
        .PromptStyle("green")
        .Validate(n => n is >= 0 and <= 20 ? ValidationResult.Success() : ValidationResult.Error("[red]Invalid number[/]")));

var input = new ModelInput
{
    PrimaryDiagnosis = diagnosis,
    AdmissionType = admissionType,
    Age = age,
    PastHospitalizations = pastHospitalizations
};

predictionService.PredictLengthOfStay(input);