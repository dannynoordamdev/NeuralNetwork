import java.util.Arrays;
import java.util.Random;

public class GenderPredictionNeuralNetwork {
    private int numInputNodes = 4; // Aantal input nodes (lengte, gewicht, leeftijd)
    private int numHiddenNodes = 5; // Aantal nodes in de verborgen laag (door de student gekozen)
    private int numOutputNodes = 1; // Aantal output nodes (voorspelde gender)

    private double[] weightsInputToHidden;
    private double[] weightsHiddenToOutput;

    public GenderPredictionNeuralNetwork() {
        Random rand = new Random();

        // Initialisatie van gewichten
        weightsInputToHidden = new double[numInputNodes * numHiddenNodes];
        weightsHiddenToOutput = new double[numHiddenNodes * numOutputNodes];

        // Willekeurige initialisatie van gewichten
        for (int i = 0; i < numInputNodes * numHiddenNodes; i++) {
            weightsInputToHidden[i] = rand.nextDouble() * 2 - 1; // Tussen -1 en 1
        }

        for (int i = 0; i < numHiddenNodes * numOutputNodes; i++) {
            weightsHiddenToOutput[i] = rand.nextDouble() * 2 - 1; // Tussen -1 en 1
        }
    }

    public double feedForward(double[] inputs) {
        // Berekeningen voor de verborgen laag
        double[] hiddenOutputs = new double[numHiddenNodes];
        for (int i = 0; i < numHiddenNodes; i++) {
            double sum = 0;
            for (int j = 0; j < numInputNodes; j++) {
                sum += inputs[j] * weightsInputToHidden[j * numHiddenNodes + i];
            }
            hiddenOutputs[i] = sigmoid(sum);
        }

        // Berekeningen voor de output
        double output = 0;
        for (int i = 0; i < numHiddenNodes; i++) {
            output += hiddenOutputs[i] * weightsHiddenToOutput[i];
        }
        return sigmoid(output);
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public void train(double[][] inputs, double[] expectedOutputs, double learningRate, int epochs) {
        Random rand = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double expectedOutput = expectedOutputs[i];
                double predictedOutput = feedForward(input);
                double error = calculateError(predictedOutput, expectedOutput);
                totalError += error;

                // Gewichten aanpassen op basis van fout
                for (int j = 0; j < numInputNodes; j++) {
                    for (int k = 0; k < numHiddenNodes; k++) {
                        double sum = 0;
                        for (int l = 0; l < numInputNodes; l++) {
                            sum += input[l] * weightsInputToHidden[l * numHiddenNodes + k];
                        }
                        weightsInputToHidden[j * numHiddenNodes + k] += learningRate * error * input[j] * predictedOutput * (1 - predictedOutput) * weightsHiddenToOutput[k] * (1 - predictedOutput);
                    }
                }

                for (int j = 0; j < numHiddenNodes; j++) {
                    double sum = 0;
                    for (int k = 0; k < numInputNodes; k++) {
                        sum += input[k] * weightsInputToHidden[k * numHiddenNodes + j];
                    }
                    weightsHiddenToOutput[j] += learningRate * error * predictedOutput * (1 - predictedOutput) * sum * (1 - predictedOutput);
                }
            }
            System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
        }
    }



    public double calculateError(double predicted, double actual) {
        return 0.5 * Math.pow((actual - predicted), 2);
    }

    public static void main(String[] args) {
        GenderPredictionNeuralNetwork neuralNetwork = new GenderPredictionNeuralNetwork();

        // Trainingsdata (lengte, gewicht, leeftijd en bijbehorende gender en wat extras)
        double[][] inputs = {
                {190, 120, 30, 100}, // Man
                {160, 55, 25, 2}, // Vrouw
                {185, 100, 35, 90}, // Man
                {175, 63, 28, 14},  // Vrouw
                {230, 120, 30, 100}, // Man
                {140, 55, 25, 22}, // Vrouw
                {195, 100, 35, 90}, // Man
                {145, 63, 28, 5},  // Vrouw

                // Voeg meer voorbeelden toe...
        };
        double[] expectedOutputs = {1, 0, 1, 0, 1, 0, 1, 0, /* Labels voor de nieuwe voorbeelden... */}; // 1 voor man, 0 voor vrouw

        neuralNetwork.train(inputs, expectedOutputs, 0.00001, 1000);

        // Voorspel het geslacht voor nieuwe datapunten
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double expectedGender = expectedOutputs[i];
            double predictedGender = neuralNetwork.feedForward(input);
            System.out.println("Input: " + Arrays.toString(input) + ", Predicted Gender: " + predictedGender + ", Expected Gender: " + expectedGender);
        }
    }

}
