import java.util.Random;

public class SimpleNeuralNetwork {
    private final int inputSize;
    private final int hiddenSize;
    private final double[][] weights1; // weights tussen input en hidden layer
    private final double[] weights2; // weights tussen hidden en output layer
    private final double biasHidden; // bias voor hidden layer
    private final double biasOutput; // bias voor output layer

    public SimpleNeuralNetwork(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Random weights initialiseren tussen -1 en 1
        weights1 = new double[hiddenSize][inputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights1[i][j] = Math.random() * 2 - 1;
            }
        }

        weights2 = new double[1]; // 1 weight voor output layer
        weights2[0] = Math.random() * 2 - 1;

        biasHidden = Math.random() * 2 - 1; // bias voor hidden layer
        biasOutput = Math.random() * 2 - 1; // bias voor output layer
    }

    public double predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size moet overeenkomen met aantal input nodes (" + inputSize + ")");
        }

        // Waardes berekenen in hidden layer
        double[] hiddenValues = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * weights1[i][j];
            }
            hiddenValues[i] = sigmoid(sum + biasHidden);
        }

        // Waarde berekenen in output layer
        double output = 0;
        for (int i = 0; i < hiddenSize; i++) {
            output += hiddenValues[i] * weights2[0];
        }
        output += biasOutput;

        return sigmoid(output);
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public void train(double[][] data, double[] targets, int populationSize, int generations) {
        if (data.length != targets.length) {
            throw new IllegalArgumentException("Aantal trainingsvoorbeelden moet overeenkomen met aantal doelwaarden");
        }

        Random random = new Random();

        double[] bestWeights1 = null;
        double[] bestWeights2 = null;
        double bestFitness = Double.MAX_VALUE;

        // Initieer de populatie met willekeurige gewichten
        for (int generation = 0; generation < generations; generation++) {
            for (int p = 0; p < populationSize; p++) {
                // Willekeurige gewichten genereren voor elke populatie
                double[] currentWeights1 = generateRandomWeights(inputSize * hiddenSize);
                double[] currentWeights2 = generateRandomWeights(hiddenSize);

                // Bereken de fitness voor deze set gewichten
                double fitness = calculateFitness(data, targets, currentWeights1, currentWeights2);

                // Als deze set gewichten beter is dan de huidige beste, update de beste
                if (fitness < bestFitness) {
                    bestFitness = fitness;
                    bestWeights1 = currentWeights1;
                    bestWeights2 = currentWeights2;
                }
            }
        }

        // Update de gewichten naar de beste gevonden gewichten
        updateWeights(bestWeights1, bestWeights2);
    }

    private double[] generateRandomWeights(int size) {
        Random random = new Random();
        double[] weights = new double[size];
        for (int i = 0; i < size; i++) {
            weights[i] = random.nextDouble() * 2 - 1; // random waarde tussen -1 en 1
        }
        return weights;
    }

    private double calculateFitness(double[][] data, double[] targets, double[] weights1, double[] weights2) {
        double totalError = 0;
        for (int t = 0; t < data.length; t++) {
            double[] input = data[t];
            double target = targets[t];

            // Forward pass met de huidige gewichten
            double predicted = predict(input, weights1, weights2);

            // Bereken de fout
            double error = target - predicted;
            totalError += Math.abs(error);
        }
        return totalError;
    }

    private void updateWeights(double[] weights1, double[] weights2) {
        // Update de interne gewichten van het netwerk
        // Dit is afhankelijk van hoe je de gewichten in je netwerk hebt opgeslagen.
        // Je zou de gewichten uit weights1 en weights2 kunnen kopiëren naar de interne gewichten van het netwerk.
    }

    private double predict(double[] input, double[] weights1, double[] weights2) {
        // Bereken de uitvoer met de huidige gewichten
        // Dit is vergelijkbaar met de predict-methode in de oorspronkelijke code, maar hier gebruik je de gegeven gewichten.
        // Je zou de gewichten uit weights1 en weights2 kunnen gebruiken om de activatiewaarden en de uiteindelijke uitvoer te berekenen.
        // Het bias kan ook worden opgenomen in de berekening, afhankelijk van hoe je de gewichten en biases hebt opgeslagen.
        return 0; // Tijdelijke waarde
    }

    // Voorbeeld (aanpassen voor je eigen data)
    public static void main(String[] args) {
        SimpleNeuralNetwork network = new SimpleNeuralNetwork(2, 3); // 2 input nodes, 3 hidden nodes

        // Trainingsdata (aanpassen)
        double[][] data = {{0.1, 0.2}, {0.7, 0.9}, {0.3, 0.5}};
        double[] targets = {0.4, 0.8, 0.6};

        // Voorspelling doen met nieuwe data (aanpassen)
        double[] newInput = {0.4, 0.6};
        double prediction = network.predict(newInput);
        System.out.println("Voorspelling: " + prediction);
    }
}
