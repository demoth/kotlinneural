@file:OptIn(ExperimentalUnsignedTypes::class)

import java.io.DataInputStream
import java.io.File
import java.util.zip.GZIPInputStream

fun main() {
    val trainingLabels = Idx1uByte("train-labels-idx1-ubyte.gz")
    check(trainingLabels.data.size == 60000)
    val testLabels = Idx1uByte("t10k-labels-idx1-ubyte.gz")
    check(testLabels.data.size == 10000)
    val trainingImages = Idx3uByte("train-images-idx3-ubyte.gz")
    check(trainingImages.data.size == 60000)
    val testImages = Idx3uByte("t10k-images-idx3-ubyte.gz")
    check(testImages.data.size == 10000)

    testImages.data.first().forEach {
        println(it.map { if (it > 0.01) '#' else ' ' }.toString())
    }
    check(testLabels.data.first() == 7.toByte())
}
abstract class IdxBuffer(fileName: String) {
    val header: Short
    val datatype: Byte
    val dimentions: Byte
    val dimentionsSizes: Array<Int>

    init {
        DataInputStream(GZIPInputStream(File(fileName).inputStream())).use {input ->
            header = input.readShort()
            check(header == 0.toShort())
            datatype = input.readByte() // assume unsigned bytes
            dimentions = input.readByte()
            dimentionsSizes = Array(dimentions.toInt()) {
                input.readInt()
            }
            load(input)
        }
    }

    abstract fun load(input: DataInputStream)
}

class Idx1uByte(fileName: String): IdxBuffer(fileName) {
    lateinit var data: ByteArray

    override fun load(input: DataInputStream) {
        data = ByteArray(dimentionsSizes.first()) {
            input.readByte()
        }
    }
}

class Idx3uByte(fileName: String): IdxBuffer(fileName) {
    lateinit var data: Array<Array<FloatArray>>
    override fun load(input: DataInputStream) {
        data = Array(dimentionsSizes.first()) {
            Array(dimentionsSizes[1]) {
                FloatArray(dimentionsSizes.last()) {
                    input.readUnsignedByte().toFloat() / 255
                }
            }
        }
    }
}
