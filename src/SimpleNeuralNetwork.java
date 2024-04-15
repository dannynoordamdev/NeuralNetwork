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

    // Voorbeeldの使い方 (aanpassen voor je eigen data)
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
