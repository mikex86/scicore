package me.mikex86.scicore.tests.makemore.rnn

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.SciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.act.Tanh
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.ranges.ALL
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tensor.get

private const val VOCAB_SIZE = 26L + 1 // 26 letters + 1 start/end char
private const val EMBEDDING_SIZE = 32L
private const val HIDDEN_SIZE = 64L

fun main() {
    val sciCore = SciCore()
    sciCore.setBackend(ISciCore.BackendType.CPU)
    sciCore.seed(123)

//    val net = MakeMoreRnnNet(sciCore)

}

//class MakeMoreRnnNet(sciCore: SciCore) : IModule {
//
//
//    private val embedding = sciCore.gaussian(
//        DataType.FLOAT32,
//        VOCAB_SIZE,
//        EMBEDDING_SIZE
//    )
//
//    // Deviation from Mikolov et al. 2010, initial hidden state is learned, not zero
//    private val start = sciCore.gaussian(
//        DataType.FLOAT32,
//        1,
//        HIDDEN_SIZE
//    )
//
//    private val rnnCell = Linear(sciCore, DataType.FLOAT32, EMBEDDING_SIZE + HIDDEN_SIZE, HIDDEN_SIZE, true)
//    private val lmHead = Linear(sciCore, DataType.FLOAT32, HIDDEN_SIZE, VOCAB_SIZE, true)
//    private val act = Tanh()
//
//    override fun forward(input: ITensor): ITensor {
//        embedding[input]
//            .use { embeddingsForSequence -> // (batch_size, seq_len, embedding_size)
//                val batchSize = embeddingsForSequence.shape[0]
//                val seqLen = embeddingsForSequence.shape[1]
//                val hidden = start.broadcast(batchSize, -1)
//                for (i in 0 until seqLen) {
//                    val embedding = embeddingsForSequence[LongRange.ALL, i, LongRange.ALL]
//                    val rnnInput = hidden.concat(embedding, 1)
//
//                }
//                lmHead(hidden)
//            }
//    }
//
//    override fun subModules(): MutableList<IModule> {
//    }
//
//}