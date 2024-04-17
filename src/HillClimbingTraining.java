import java.util.Arrays;
import java.util.Random;

public class HillClimbingTraining {
    private GenderPredictionNeuralNetwork neuralNetwork;
    public HillClimbingTraining(GenderPredictionNeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public void train(double[][] inputs, double[] expectedOutputs, int iterations, double stepSize) {
        Random rand = new Random();
        double currentError = calculateTotalError(inputs, expectedOutputs);

        for (int i = 0; i < iterations; i++) {
            // Generate a random perturbation to the weights
            double[] perturbation = new double[neuralNetwork.getWeightsInputToHidden().length + neuralNetwork.getWeightsHiddenToOutput().length];
            for (int j = 0; j < perturbation.length; j++) {
                perturbation[j] = rand.nextDouble() * stepSize * 2 - stepSize; // Random value between -stepSize and +stepSize
            }

            // Apply perturbation to the weights and calculate new error
            double[] newWeightsInputToHidden = applyPerturbation(neuralNetwork.getWeightsInputToHidden(), perturbation, neuralNetwork.getNumInputNodes(), neuralNetwork.getNumHiddenNodes());
            double[] newWeightsHiddenToOutput = applyPerturbation(neuralNetwork.getWeightsHiddenToOutput(), perturbation, neuralNetwork.getNumHiddenNodes(), neuralNetwork.getNumOutputNodes());
            neuralNetwork.setWeightsInputToHidden(newWeightsInputToHidden);
            neuralNetwork.setWeightsHiddenToOutput(newWeightsHiddenToOutput);
            double newError = calculateTotalError(inputs, expectedOutputs);

            // If the new error is smaller, accept the perturbation
            if (newError < currentError) {
                currentError = newError;
                System.out.println("Iteration " + i + ", Error: " + currentError);
            } else {
                // If not, revert the weights back to the original
                neuralNetwork.setWeightsInputToHidden(newWeightsInputToHidden);
                neuralNetwork.setWeightsHiddenToOutput(newWeightsHiddenToOutput);
            }
        }
    }

    private double[] applyPerturbation(double[] weights, double[] perturbation, int numInputNodes, int numOutputNodes) {
        double[] newWeights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            newWeights[i] = weights[i] + perturbation[i];
        }
        return newWeights;
    }

    private double calculateTotalError(double[][] inputs, double[] expectedOutputs) {
        double totalError = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double expectedOutput = expectedOutputs[i];
            double predictedOutput = neuralNetwork.feedForward(input);
            totalError += neuralNetwork.calculateError(predictedOutput, expectedOutput);
        }
        return totalError;
    }

    public static void main(String[] args) {
        GenderPredictionNeuralNetwork neuralNetwork = new GenderPredictionNeuralNetwork();
        HillClimbingTraining hillClimbing = new HillClimbingTraining(neuralNetwork);

        System.out.println(Arrays.toString(neuralNetwork.getWeightsInputToHidden()));
        System.out.println(Arrays.toString(neuralNetwork.getWeightsHiddenToOutput()));

        double[][] training_inputs = {
                {190, 120, 30, 100}, // Man
                {160, 55, 25, 2}, // Vrouw
                {185, 100, 35, 90}, // Man
                {175, 63, 28, 14},  // Vrouw
                {230, 120, 30, 100}, // Man
                {140, 55, 25, 22}, // Vrouw
                {195, 100, 35, 90}, // Man
                {145, 63, 28, 5},  // Vrouw
        };
        double[] training_outputs = {1, 0, 1, 0, 1, 0, 1, 0}; // 1 voor man, 0 voor vrouw
        // Train the neural network using hill climbing
        hillClimbing.train(training_inputs, training_outputs, 100, 0.06);

        System.out.println(Arrays.toString(neuralNetwork.getWeightsInputToHidden()));
        System.out.println(Arrays.toString(neuralNetwork.getWeightsHiddenToOutput()));

        double[][] inputs = {
                {130, 60, 30, 20}, // Vrouw
                {160, 50, 25, 2}, // Vrouw
                {195, 85, 35, 90}, // Man
                {173, 63, 28, 14},  // Vrouw
                {220, 110, 40, 100}, // Man
                {150, 60, 25, 80}, // Man
                {175, 100, 35, 85}, // Man
                {150, 43, 28, 5},  // Vrouw
        };
        double[] expectedOutputs = {0, 0, 1, 0, 1, 1, 1, 0}; // 1 voor man, 0 voor vrouw

        // Voorspel het geslacht voor nieuwe datapunten
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double expectedGender = expectedOutputs[i];
            double predictedGender = neuralNetwork.feedForward(input);
            System.out.println("Input: " + Arrays.toString(input) + ", Predicted Gender: " + predictedGender + ", Expected Gender: " + expectedGender);
        }
    }
}
