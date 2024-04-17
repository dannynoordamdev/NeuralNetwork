import java.util.Arrays;
import java.util.Random;

public class GenderPredictionNeuralNetwork {
    private int numInputNodes = 4; // Aantal input nodes (lengte, gewicht, leeftijd)
    private int numHiddenNodes = 3; // Aantal nodes in de verborgen laag (door de student gekozen)
    private int numOutputNodes = 1; // Aantal output nodes (voorspelde gender)
    private double[] weightsInputToHidden;
    private double[] weightsHiddenToOutput;
    private double[] hiddenOutputs; // Variabele om de uitvoer van de verborgen laag op te slaan
    private double bestError = Double.MAX_VALUE;
    private double[] bestWeightsInputToHidden;
    private double[] bestWeightsHiddenToOutput;

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
        double[] hto = {0.17428701824981752, -0.5207287214486467, -0.5211525453346628};
        setWeightsHiddenToOutput(hto);
        double[] ith = {-0.32560788042573185, -0.8947579423251795, -0.9758304447841937, -0.017240487481357647, 0.31342305463621223, -0.49349010642855173, 0.5965522198802717, 0.7698250652609964, 0.282916928098244, 0.5313810433266788, 0.8835846735224344, -0.7542223395893735};
        setWeightsInputToHidden(ith);
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

                // Tijdelijk de gewichten opslaan
                double[] tempWeightsInputToHidden = weightsInputToHidden.clone();
                double[] tempWeightsHiddenToOutput = weightsHiddenToOutput.clone();

                // Hill climbing voor de gewichten van de verborgen laag naar de uitvoerlaag
                for (int j = 0; j < numHiddenNodes; j++) {
                    double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
                    weightsHiddenToOutput[j] += learningRate * deltaOutput * hiddenOutputs[j];

                    // Controleren of de nieuwe oplossing beter is
                    double newError = totalError + error;
                    if (newError < bestError) {
                        bestError = newError;
                        bestWeightsHiddenToOutput = weightsHiddenToOutput.clone();
                    } else {
                        // Als de nieuwe oplossing niet beter is, herstellen we de gewichten
                        weightsHiddenToOutput = tempWeightsHiddenToOutput.clone();
                    }
                }

                // Hill climbing voor de gewichten van de invoerlaag naar de verborgen laag
                for (int j = 0; j < numInputNodes; j++) {
                    for (int k = 0; k < numHiddenNodes; k++) {
                        double deltaHidden = 0;
                        for (int l = 0; l < numOutputNodes; l++) {
                            double deltaOutput = (expectedOutput - predictedOutput) * predictedOutput * (1 - predictedOutput);
                            deltaHidden += deltaOutput * bestWeightsHiddenToOutput[k * numOutputNodes + l] * hiddenOutputs[k] * (1 - hiddenOutputs[k]) * input[j];
                        }
                        weightsInputToHidden[j * numHiddenNodes + k] += learningRate * deltaHidden;

                        // Controleren of de nieuwe oplossing beter is
                        double newError = totalError + error;
                        if (newError < bestError) {
                            bestError = newError;
                            bestWeightsInputToHidden = weightsInputToHidden.clone();
                        } else {
                            // Als de nieuwe oplossing niet beter is, herstellen we de gewichten
                            weightsInputToHidden = tempWeightsInputToHidden.clone();
                        }
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
        System.out.println("weightshiddentooutput: " + Arrays.toString(neuralNetwork.getWeightsHiddenToOutput()));
        System.out.println("weightsinputtohidden: " + Arrays.toString(neuralNetwork.getWeightsInputToHidden()));
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

        // Sla de beste gewichten op
        neuralNetwork.setWeightsInputToHidden(neuralNetwork.bestWeightsInputToHidden);
        neuralNetwork.setWeightsHiddenToOutput(neuralNetwork.bestWeightsHiddenToOutput);
    }
}
