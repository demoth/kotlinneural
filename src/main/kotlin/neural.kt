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
            FloatArray(inputs) { randomFloat0to1() }
        },
        //nodeBias = FloatArray(outputs) { randomFloat0to1() }
    )


    fun eval() {
        check(layerInput.size == weights.first().size)
        weights.forEachIndexed { nodeIndex, nodeWeights ->
            activation[nodeIndex] = dotProduct(layerInput, nodeWeights)// + nodeBias[nodeIndex]
            layerOutput[nodeIndex] = sigmoid(activation[nodeIndex])
        }
    }
}

fun dotProduct(a: FloatArray, b: FloatArray): Float {
    return a.mapIndexed { i, fl -> fl * b[i] }.sum()
}

fun sigmoid(a: Float): Float = (1f / (1f + exp((-a).toDouble()))).toFloat() // Fixme

fun sigmoidDer(a: Float) = sigmoid(a) * (1 - sigmoid(a))

fun randomFloat0to1() = Random.nextFloat()

class NNetwork(val layers: List<NNetworkLayer>) {
    val learningRate = -0.1f

    constructor(vararg shape: Int) : this(
        shape.toList().windowed(2) {
            NNetworkLayer(it.first(), it.last())
        }
    )

    fun eval(inputs: FloatArray) {
        var outputs = inputs
        layers.forEach { layer ->
            layer.layerInput = outputs
            layer.eval()
            outputs = layer.activation
        }
    }

    fun backPropagate(expected: FloatArray) {
        val actual = layers.last().layerOutput
        val weightDiff = Array(layers.size) {
            Array(layers[it].weights.size) { k ->
                FloatArray(layers[it].weights[k].size)
            }
        }
        //
        var delta = FloatArray(actual.size) { i ->
            sigmoidDer(layers.last().activation[i]) * (actual[i] - expected[i])
        }
        //weightDiff[layers.size - 1] = delta

        (layers.size - 1 downTo 1).forEach { k ->

            val layerRight = layers[k + 1]
            val layerLeft = layers[k - 1]
            val layerCurrent = layers[k]
            val peppersOnLeftLayer = layerLeft.weights.size
            val peppersOnCurrentLayer = layerCurrent.weights.size
            val peppersOnRightLayer = layerRight.weights.size

            for (nodeInCurrentLayerIndex in 0 until peppersOnCurrentLayer) {
                var mySum = 0f

                for (nodeInRightLayerIndex in 0 until peppersOnRightLayer) {
                    val palkaFromCurrentNodeIndex = nodeInCurrentLayerIndex
                    val weightFromCurrentNodeToANodeInTheRightLayer =
                        layerRight.weights[nodeInRightLayerIndex][palkaFromCurrentNodeIndex]
                    mySum += weightFromCurrentNodeToANodeInTheRightLayer * delta[nodeInRightLayerIndex]
                }

                val currentNodeActivation = layerCurrent.activation[nodeInCurrentLayerIndex]
                val deltaScalar = mySum * sigmoidDer(currentNodeActivation)
                delta[nodeInCurrentLayerIndex] = deltaScalar
                // Fixme use different arrays for delta right and delta current
                for (palka in 0 until peppersOnLeftLayer) {
                    var adjustment = (-learningRate) * layerLeft.layerOutput[palka] * deltaScalar
                    weightDiff[k][nodeInCurrentLayerIndex][palka] = adjustment
                    // we don't update weights here because we'll need old weights on the left layer
                }
            }
        }
        // TODO iterate over weights and apply adjustment
    }
}

fun main() {
    val sampleSize = 30000

    val inputSize = 784
    val outputSize = 10

    val samples: Array<FloatArray> = Array(sampleSize) { FloatArray(inputSize) { randomFloat0to1() } }
    val expected: Array<FloatArray> = Array(sampleSize) { FloatArray(outputSize) { randomFloat0to1() } }

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

