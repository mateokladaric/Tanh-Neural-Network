class NeuralNetwork
{
    private const float learningRate = 0.0333f;

    private List<List<float>> biases;
    private List<List<float>> errors;
    private List<List<float>> values;
    private List<List<List<float>>> weights;

    private Random r = new Random();

    public NeuralNetwork(int[] layers)
    {
        initNeurons(layers);
        initWeights(layers);
    }

    private void initNeurons(int[] layers)
    {
        biases = new List<List<float>>();
        errors = new List<List<float>>();
        values = new List<List<float>>();

        for (int layerIdx = 0; layerIdx < layers.Length; layerIdx++)
        {
            biases.Add(new List<float>());
            errors.Add(new List<float>());
            values.Add(new List<float>());

            for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
            {
                biases[layerIdx].Add(0);
                errors[layerIdx].Add(0);
                values[layerIdx].Add(0);
            }
        }
    }

    private void initWeights(int[] layers)
    {
        weights = new List<List<List<float>>>();

        for (int layerIdx = 0; layerIdx < layers.Length - 1; layerIdx++)
        {
            weights.Add(new List<List<float>>());

            for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
            {
                weights[layerIdx].Add(new List<float>());

                for (int nextNeuronIdx = 0; nextNeuronIdx < layers[layerIdx + 1]; nextNeuronIdx++)
                {
                    weights[layerIdx][neuronIdx].Add((float)r.NextDouble() - 0.5f);
                }
            }
        }
    }

    private void activateTanh()
    {
        for (int layerIdx = 1; layerIdx < values.Count(); layerIdx++)
        {
            for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
            {
                float sum = 0;
                for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
                {
                    sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
                }
                values[layerIdx][neuronIdx] = (float)Math.Tanh(sum + biases[layerIdx][neuronIdx]);
            }
        }
    }

    public float[] getTanhOutput(float[] inputs)
    {
        Contract.Requires(inputs.Length == values[0].Count());

        for (int neuronIdx = 0; neuronIdx < inputs.Length; neuronIdx++)
        {
            values[0][neuronIdx] = inputs[neuronIdx];
        }

        activateTanh();

        return values[values.Count() - 1].ToArray();
    }

    public void backPropagate(float[] correctOutput)
    {
        for (int neuronIdx = 0; neuronIdx < values[values.Count() - 1].Count(); neuronIdx++)
        {
            errors[errors.Count() - 1][neuronIdx] = (correctOutput[neuronIdx] - values[values.Count() - 1][neuronIdx]) * (1 - values[values.Count() - 1][neuronIdx] * values[values.Count() - 1][neuronIdx]);
        }

        for (int layerIdx = values.Count() - 2; layerIdx > 0; layerIdx--)
        {
            for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
            {
                float error = 0;
                for (int nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].Count(); nextNeuronIdx++)
                {
                    error += errors[layerIdx + 1][nextNeuronIdx] * weights[layerIdx][neuronIdx][nextNeuronIdx];
                }
                errors[layerIdx][neuronIdx] = error * (1 - values[layerIdx][neuronIdx] * values[layerIdx][neuronIdx]);
            }
        }

        for (int layerIdx = values.Count() - 1; layerIdx > 0; layerIdx--)
        {
            for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
            {
                biases[layerIdx][neuronIdx] += errors[layerIdx][neuronIdx] * learningRate;
                for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
                {
                    weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += values[layerIdx - 1][prevNeuronIdx] * errors[layerIdx][neuronIdx] * learningRate;
                }
            }
        }
    }

    public void Train(float[] inputs, float[] correctOutput)
    {
        getTanhOutput(inputs);
        backPropagate(correctOutput);
    }
}
