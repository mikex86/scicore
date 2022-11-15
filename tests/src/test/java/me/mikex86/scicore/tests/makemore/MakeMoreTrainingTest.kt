package me.mikex86.scicore.tests.makemore

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.data.DatasetIterator
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor

const val BATCH_SIZE = 32
const val BLOCK_SIZE = 3
const val EMBEDDING_SIZE = 10
const val N_HIDDEN = 128
const val VOCAB_SIZE = 26 + 1 // 26 letters + 1 padding char

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

    val trainIt = DatasetIterator(BATCH_SIZE, NamesDatasetSupplier(sciCore, BLOCK_SIZE, true))
    val testIt = DatasetIterator(BATCH_SIZE, NamesDatasetSupplier(sciCore, BLOCK_SIZE, false))

}

class MakeMoreNet(sciCore: SciCore) : IModule {

    private val embedding = sciCore.zeros(DataType.FLOAT32, VOCAB_SIZE.toLong(), EMBEDDING_SIZE.toLong())
    private val fc1 = Linear(sciCore, DataType.FLOAT32, (EMBEDDING_SIZE * BLOCK_SIZE).toLong(), N_HIDDEN.toLong(), true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, N_HIDDEN.toLong(), VOCAB_SIZE.toLong(), true)

    override fun forward(input: ITensor): ITensor {
        // input [character index sequence]: (batchSize, blockSize)
        // embedding [character embeddings]: (vocabSize, embeddingSize)
        val embeddingsForSequence = embedding[input]
        // TODO: IMPLEMENT
        return input
    }

    override fun subModules(): List<IModule> {
        return listOf(fc1, fc2)
    }

    override fun parameters(): List<ITensor> {
        return listOf(fc1.parameters(), fc2.parameters())
            .flatten()
            .toMutableList()
            .apply { add(embedding) }
    }

}