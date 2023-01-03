package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Dropout
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.utils.use


data class GPTConfig(
    val vocabSize: Int,
    val blockSize: Int = 1024,
    val nLayers: Int = 12,
    val nHeads: Int = 12,
    val nEmbed: Int = 768,
    val dropout: Float = 0.1f,
)

class CausalSelfAttention(private val sciCore: ISciCore, private val config: GPTConfig) : IModule {

    private val cAttn = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed * 3L, true)

    private val cProj = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed.toLong(), true)

    private val attnDropout = Dropout(sciCore, config.dropout)

    private val residDropout = Dropout(sciCore, config.dropout)

    private val bias = sciCore.zeros(DataType.INT32, 1, 1, config.blockSize.toLong(), config.blockSize.toLong())

    init {
        // triangle mask for causal attention
        for (i in 0 until config.blockSize) {
            for (j in 0..i) {
                bias.setFloat(1.0f, 0, 0, i.toLong(), j.toLong())
            }
        }
    }


    override fun forward(x: ITensor): ITensor {
        val (batchSize, seqLen, embedDim) = x.shape
        return cAttn(x) // x: (batch, seq, embed), cAttn: (batch, seq, embed*3)
            .use { h -> h.split(config.nEmbed, 2) } // list of 3 tensors: (batch, seq, embed)
            .use { (q, k, v) ->
                Triple(
                    k.view(batchSize, seqLen, config.nHeads.toLong(), (config.nEmbed / config.nHeads).toLong()),
                    q.view(batchSize, seqLen, config.nHeads.toLong(), (config.nEmbed / config.nHeads).toLong()),
                    v.view(batchSize, seqLen, config.nHeads.toLong(), (config.nEmbed / config.nHeads).toLong())
                )
            }
            .use { q, k, v ->
                TODO("Not yet implemented")
            }
    }

    override fun subModules(): List<IModule> {
        return listOf(cAttn, cProj, attnDropout, residDropout)
    }

}

class GPT : IModule {
    override fun forward(input: ITensor): ITensor {
        TODO("Not yet implemented")
    }
}