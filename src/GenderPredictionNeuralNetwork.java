import java.util.Arrays;
import java.util.Random;
public class GenderPredictionNeuralNetwork {
    private int numInputNodes = 4;
    private int numHiddenNodes = 3;
    private int numOutputNodes = 1;
    private double[] weightsInputToHidden;
    private double[] weightsHiddenToOutput;
    private double[] hiddenOutputs;
    private double bestError = Double.MAX_VALUE;
    private double[] bestWeightsInputToHidden;
    private double[] bestWeightsHiddenToOutput;

    public void setWeightsInputToHidden(double[] input) {
        this.weightsInputToHidden = input;
    }

    public double[] getWeightsHiddenToOutput() {
        return weightsHiddenToOutput;
    }

    public void setWeightsHiddenToOutput(double[] input) {
        this.weightsHiddenToOutput = input;
    }

    public double[] getWeightsInputToHidden() {
        return weightsInputToHidden;
    }

    public int getNumInputNodes() {
        return numInputNodes;
    }

    public int getNumHiddenNodes() {
        return numHiddenNodes;
    }

    public int getNumOutputNodes() {
        return numOutputNodes;
    }

    public GenderPredictionNeuralNetwork() {
        // Initializeer de gewichten willekeurig
        initializeWeights();

        // Initialiseer de beste gewichten met de huidige gewichten
        bestWeightsInputToHidden = weightsInputToHidden.clone();
        bestWeightsHiddenToOutput = weightsHiddenToOutput.clone();
        // initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        this.weightsHiddenToOutput = new double[numHiddenNodes];
        this.weightsInputToHidden = new double[numInputNodes * numHiddenNodes];

        for (int i = 0; i < numHiddenNodes; i++) {
            weightsHiddenToOutput[i] = rand.nextDouble() - 0.5; // Willekeurige gewichten tussen -0.5 en 0.5
        }

        for (int i = 0; i < numInputNodes * numHiddenNodes; i++) {
            weightsInputToHidden[i] = rand.nextDouble() - 0.5; // Willekeurige gewichten tussen -0.5 en 0.5
        }
    }

    public double feedForward(double[] inputs) {
        // Berekeningen voor de verborgen laag
        hiddenOutputs = new double[numHiddenNodes];
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

                // Bereken de fout na het bijwerken van de gewichten
                double error = calculateError(predictedOutput, expectedOutput);
                totalError += error;

                // Update de gewichten
                updateWeights(input, predictedOutput, expectedOutput, learningRate);

                // Houd de beste gewichten bij
                if (totalError < bestError) {
                    bestError = totalError;
                    bestWeightsInputToHidden = weightsInputToHidden.clone();
                    bestWeightsHiddenToOutput = weightsHiddenToOutput.clone();
                }
            }
            System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
        }
    }

    private void updateWeights(double[] input, double predictedOutput, double expectedOutput, double learningRate) {
        // Update de gewichten van de verborgen laag naar de uitvoerlaag
        for (int j = 0; j < numHiddenNodes; j++) {
            double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
            weightsHiddenToOutput[j] += learningRate * deltaOutput * hiddenOutputs[j];
        }

        // Update de gewichten van de invoerlaag naar de verborgen laag
        for (int j = 0; j < numInputNodes; j++) {
            for (int k = 0; k < numHiddenNodes; k++) {
                double deltaHidden = 0;
                for (int l = 0; l < numOutputNodes; l++) {
                    double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
                    deltaHidden += deltaOutput * bestWeightsHiddenToOutput[k * numOutputNodes + l] * hiddenOutputs[k] * (1 - hiddenOutputs[k]) * input[j];
                }
                weightsInputToHidden[j * numHiddenNodes + k] += learningRate * deltaHidden;
            }
        }
    }

    public double calculateError(double predicted, double actual) {
        return 0.5 * Math.pow((actual - predicted), 2);
    }

    public static void main(String[] args) {
        GenderPredictionNeuralNetwork neuralNetwork = new GenderPredictionNeuralNetwork();

        // Trainingsdata (lengte, gewicht, leeftijd en bijbehorende gender)
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
        double[] expectedOutputs = {1, 0, 1, 0, 1, 0, 1, 0}; // 1 voor man, 0 voor vrouw

        double learningRate = 0.01;
        int epochs = 1000;

        neuralNetwork.train(inputs, expectedOutputs, learningRate, epochs);

        int correctPredictions = 0;

        double[][] inputs_vali = {
                {170, 110, 30, 110}, // Man
                {180, 80, 20, 75}, // Man
                {155, 10, 5, 20}, // Vrouw
                {160, 40, 18, 24},  // Vrouw
                {210, 110, 40, 120}, // Man
                {180, 85, 25, 82}, // Man
                {135, 30, 35, 30}, // Vrouw
                {180, 93, 28, 88},  // Man
                // Voeg meer voorbeelden toe...
        };
        double[] expectedOutputs_vali = {1, 1, 0, 0, 1, 1, 0, 1}; // 1 voor man, 0 voor vrouw

        // Voorspel het geslacht voor nieuwe datapunten
        for (int i = 0; i < inputs_vali.length; i++) {
            double[] input = inputs_vali[i];
            double expectedGender = expectedOutputs_vali[i];
            double predictedGender = neuralNetwork.feedForward(input);
            String genderPrediction = (predictedGender <= 0.5) ? "vrouw" : "man";
            String expectedGenderString = (expectedGender == 0) ? "vrouw" : "man";
            System.out.println("Input: " + Arrays.toString(input) + ", Predicted Gender: " + genderPrediction + ", Expected Gender: " + expectedGenderString);

            if ((predictedGender <= 0.5 && expectedGender == 0) || (predictedGender > 0.5 && expectedGender == 1)) {
                correctPredictions++;
            }
        }

        double accuracy = ((double) correctPredictions / inputs_vali.length) * 100;
        System.out.println("Accuracy: " + accuracy + "%");
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Epochs: " + epochs);

        // Voorspel het geslacht voor nieuwe datapunten
        // Voeg validatiedata toe en voorspel het geslacht
    }
}
