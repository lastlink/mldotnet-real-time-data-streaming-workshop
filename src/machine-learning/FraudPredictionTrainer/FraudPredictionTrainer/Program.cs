using FraudPreditionTrainer.Schema;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Diagnostics;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace FraudPreditionTrainer
{
    class Program
    {
        private static string DataPath = "Data/data.csv";

        static void Main(string[] args)
        {
            // Create new stopwatch.
            Stopwatch stopwatch = new Stopwatch();
            // Begin timing.
            stopwatch.Start();
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);

            var mlContext = new MLContext(seed: 1);

            ITransformer trainedModel = null;
            // TrainTestData testTrainData = null;
            DataViewSchema modelSchema;

            //Load
            var data = mlContext.Data.LoadFromTextFile<Transaction>(DataPath, hasHeader: true, separatorChar: ',');
            modelSchema = data.Schema;
            var testTrainData = mlContext.Data.TrainTestSplit(data);

            string zipFile = "MLModel2.zip";

            if (File.Exists(zipFile))
            {
                // Load trained model
                trainedModel = mlContext.Model.Load(zipFile, out modelSchema);
            }
            else
            {


                Console.WriteLine("Time elapsed: {0}-TrainTestSplit", stopwatch.Elapsed);

                //Transform
                var dataProcessingPipeline = BuildDataProcessingPipeline(mlContext);
                Console.WriteLine("Time elapsed: {0}-BuildDataProcessingPipeline", stopwatch.Elapsed);

                //Train
                var trainingPipeline = BuildTrainingPipeline(mlContext, dataProcessingPipeline);
                Console.WriteLine("Time elapsed: {0}-BuildTrainingPipeline", stopwatch.Elapsed);

                trainedModel = trainingPipeline.Fit(testTrainData.TrainSet);
                Console.WriteLine("Time elapsed: {0}-Fit", stopwatch.Elapsed);


            }
            //Evaluate
            var predictions = trainedModel.Transform(testTrainData.TestSet);
            Console.WriteLine("Time elapsed: {0}-Transform", stopwatch.Elapsed);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "isFraud");

            Console.WriteLine("Time elapsed: {0}-BinaryClassification", stopwatch.Elapsed);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AUCPC: {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall}");
            Console.WriteLine($"Precision: {metrics.PositivePrecision}");
            //Save
            mlContext.Model.Save(trainedModel, modelSchema, zipFile);
            Console.WriteLine("End Program Time elapsed: {0}", stopwatch.Elapsed);
            stopwatch.Stop();
            Console.ReadKey();
        }

        private static IEstimator<ITransformer> BuildDataProcessingPipeline(MLContext mlContext)
        {
            return mlContext.Transforms.Categorical.OneHotEncoding("type")
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("nameDest"))
                .Append(mlContext.Transforms.Concatenate("Features", "type", "nameDest", "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest")
                .Append(mlContext.Transforms.NormalizeMinMax("Features")));
        }

        private static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext, IEstimator<ITransformer> dataProcessingPipeline)
        {
            return dataProcessingPipeline.Append(mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options() { NumberOfLeaves = 10, NumberOfTrees = 500, LabelColumnName = "isFraud", FeatureColumnName = "Features" }));
        }
    }
}
