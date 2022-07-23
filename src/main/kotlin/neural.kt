import kotlin.math.exp
import kotlin.random.Random
import kotlin.system.measureTimeMillis

class NNetworkLayer(
    val weights: Array<FloatArray>,
    //val nodeBias: FloatArray,
    var layerInput: FloatArray = FloatArray(weights.first().size),
    val activation: FloatArray = FloatArray(weights.size),
    val layerOutput: FloatArray = FloatArray(weights.size)
) {
    constructor(inputs: Int, outputs: Int) : this(
        weights = Array(outputs) {
            FloatArray(inputs) { random1m1() }
        },
        //nodeBias = FloatArray(outputs) { randomFloat0to1() }
    )


    fun eval(lastLayer: Boolean) {
        check(layerInput.size == weights.first().size)
        weights.forEachIndexed { nodeIndex, nodeWeights ->
            activation[nodeIndex] = dotProduct(layerInput, nodeWeights)// + nodeBias[nodeIndex]
            layerOutput[nodeIndex] = if (!lastLayer) sigmoid(activation[nodeIndex]) else activation[nodeIndex]
        }
    }
}

fun dotProduct(a: FloatArray, b: FloatArray): Float {
    return a.mapIndexed { i, fl -> fl * b[i] }.sum()
}

fun sigmoid(a: Float): Float = (1f / (1f + exp((-a).toDouble()))).toFloat() // Fixme

fun sigmoidDer(a: Float) = sigmoid(a) * (1 - sigmoid(a))

fun random1m1() = Random.nextFloat() * 2 - 1f

class NNetwork(val layers: List<NNetworkLayer>) {
    val learningRate = 0.1f

    constructor(vararg shape: Int) : this(
        shape.toList().windowed(2) {
            NNetworkLayer(it.first(), it.last())
        }
    )

    fun eval(inputs: FloatArray): String {
        var outputs = inputs
        layers.forEachIndexed { index, layer ->
            layer.layerInput = outputs
            layer.eval(index == layers.size - 1)
            outputs = layer.layerOutput
        }
        return outputs.contentToString()
    }

    fun backPropagate(expected: FloatArray) {
        val output = layers.last().layerOutput
        val weightDiff = Array(layers.size) {
            Array(layers[it].weights.size) { k ->
                FloatArray(layers[it].weights[k].size)
            }
        }
        //
        var deltaRight = FloatArray(output.size) { i ->
            output[i] - expected[i]
        }
        //weightDiff[layers.size - 1] = delta

        (layers.size - 2 downTo 1).forEach { k ->

            val layerRight = layers[k + 1]
            val layerLeft = layers[k - 1]
            val layerCurrent = layers[k]
            val nodesOnLeftLayer = layerLeft.weights.size
            val nodesOnCurrentLayer = layerCurrent.weights.size
            val nodesOnRightLayer = layerRight.weights.size
            val deltaCurrent = FloatArray(layerCurrent.weights.size)

            for (nodeInCurrentLayerIndex in 0 until nodesOnCurrentLayer) {
                var mySum = 0f

                for (nodeInRightLayerIndex in 0 until nodesOnRightLayer) {
                    val palkaFromCurrentNodeIndex = nodeInCurrentLayerIndex
                    val weightFromCurrentNodeToANodeInTheRightLayer =
                        layerRight.weights[nodeInRightLayerIndex][palkaFromCurrentNodeIndex]
                    mySum += weightFromCurrentNodeToANodeInTheRightLayer * deltaRight[nodeInRightLayerIndex]
                }
                // delta = FloatArray(nodesOnCurrentLayer)
                val currentNodeActivation = layerCurrent.activation[nodeInCurrentLayerIndex]
                val deltaScalar = sigmoidDer(currentNodeActivation) * mySum
                deltaCurrent[nodeInCurrentLayerIndex] = deltaScalar
                for (leftNodeIndex in 0 until nodesOnLeftLayer) {
                    val adjustment = (-learningRate) * layerLeft.layerOutput[leftNodeIndex] * deltaScalar
                    weightDiff[k][nodeInCurrentLayerIndex][leftNodeIndex] = adjustment
                    // we don't update weights here because we'll need old weights on the left layer
                }
            }
            deltaRight = deltaCurrent
        }

        layers.forEachIndexed { layerIndex, layer ->
            layer.weights.forEachIndexed { nodeIndex, weights ->
                for (i in weights.indices) {
                    weights[i] += weightDiff[layerIndex][nodeIndex][i]
                }
            }
        }
    }
}

fun main() {
    val inputSize = 784
    val outputSize = 10

    val trainingLabels = Idx1uByte("train-labels-idx1-ubyte.gz")
    val trainingImages = Idx3uByte("train-images-idx3-ubyte.gz")


    val samples: Array<FloatArray> = Array(trainingImages.data.size) { sampleIndex ->
        FloatArray(inputSize) { i ->
            trainingImages.data[sampleIndex][i / 28][i % 28]
        }
    }
    val expected: Array<FloatArray> = Array(trainingLabels.data.size) {
        val f = FloatArray(outputSize)
        f[trainingLabels.data[it].toInt()] = 1f
        f
    }

    val network = NNetwork(inputSize, 128, 64, outputSize)
    val time = measureTimeMillis {
        samples.forEachIndexed { sampleIndex, sample ->
            network.eval(sample)
            // backpropagate:
            network.backPropagate(expected[sampleIndex])
            // if (sampleIndex % 1000)
            // network.update()
        }
        // network.update()

    }
    println("Elapsed $time millis")

}

fun printSample(f: Array<FloatArray>) {
    f.forEach {
        println(it.map { if (it > 0.01) '#' else ' ' }.toString())
    }
}
