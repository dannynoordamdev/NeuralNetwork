import java.util.Arrays;
import java.util.Random;

public class GenderPredictionNeuralNetwork {
    private int numInputNodes = 4; // Aantal input nodes (lengte, gewicht, leeftijd)
    private int numHiddenNodes = 3; // Aantal nodes in de verborgen laag (door de student gekozen)
    private int numOutputNodes = 1; // Aantal output nodes (voorspelde gender)
    private double[] weightsInputToHidden;
    private double[] weightsHiddenToOutput;
    private double[] hiddenOutputs; // Variabele om de uitvoer van de verborgen laag op te slaan

    public void setWeightsInputToHidden(double[] input){
        this.weightsInputToHidden = input;
    }

    public double[] getWeightsHiddenToOutput() {
        return weightsHiddenToOutput;
    }

    public void setWeightsHiddenToOutput(double[] input){
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
        hiddenOutputs = new double[numHiddenNodes]; // Initialisatie van hiddenOutputs
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

                // Aanpassen van de gewichten van de verborgen laag naar de uitvoerlaag
                for (int j = 0; j < numHiddenNodes; j++) {
                    double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
                    weightsHiddenToOutput[j] += learningRate * deltaOutput * hiddenOutputs[j];
                }

                // Aanpassen van de gewichten van de invoerlaag naar de verborgen laag
                for (int j = 0; j < numInputNodes; j++) {
                    for (int k = 0; k < numHiddenNodes; k++) {
                        double deltaHidden = 0;
                        for (int l = 0; l < numOutputNodes; l++) {
                            double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
                            deltaHidden += deltaOutput * weightsHiddenToOutput[k * numOutputNodes + l] * hiddenOutputs[k] * (1 - hiddenOutputs[k]) * input[j];
                        }
                        weightsInputToHidden[j * numHiddenNodes + k] += learningRate * deltaHidden;
                    }
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
        double[] expectedOutputs = {1, 0, 1, 0, 1, 0, 1, 0}; // 1 voor man, 0 voor vrouw

        int numHiddenNodes = neuralNetwork.getNumHiddenNodes();
        double learningRate = 0.1;
        int epochs = 1000;

        neuralNetwork.train(inputs, expectedOutputs, learningRate, epochs);

        int correctPredictions = 0;

        // Voorspel het geslacht voor nieuwe datapunten
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double expectedGender = expectedOutputs[i];
            double predictedGender = neuralNetwork.feedForward(input);
            String genderPrediction = (predictedGender <= 0.5) ? "vrouw" : "man";
            String expectedGenderString = (expectedGender == 0) ? "vrouw" : "man";
            System.out.println("Input: " + Arrays.toString(input) + ", Predicted Gender: " + genderPrediction + ", Expected Gender: " + expectedGenderString);

            if ((predictedGender <= 0.5 && expectedGender == 0) || (predictedGender > 0.5 && expectedGender == 1)) {
                correctPredictions++;
            }
        }

        double accuracy = ((double) correctPredictions / inputs.length) * 100;
        System.out.println("Accuracy: " + accuracy + "%");
        System.out.println("Number of Hidden Nodes: " + numHiddenNodes);
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Epochs: " + epochs);
    }



}
