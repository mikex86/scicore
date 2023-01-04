package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Dropout
import me.mikex86.scicore.nn.layers.LayerNorm
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.utils.use
import kotlin.math.sqrt


data class GPTConfig(
    val vocabSize: Int,
    val blockSize: Int = 1024,
    val nLayers: Int = 12,
    val nHeads: Int = 12,
    val nEmbed: Int = 768,
    val dropout: Float = 0.1f,
)

class CausalSelfAttention(sciCore: ISciCore, private val config: GPTConfig) : IModule {

    private val cAttn = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed * 3L, true)

    private val cProj = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed.toLong(), true)

    private val attnDropout = Dropout(sciCore, config.dropout)

    private val residDropout = Dropout(sciCore, config.dropout)

    private val attnMask = sciCore.zeros(DataType.INT32, 1, 1, config.blockSize.toLong(), config.blockSize.toLong())

    init {
        // triangle mask for causal attention
        for (i in 0 until config.blockSize) {
            for (j in 0..i) {
                attnMask.setFloat(Float.NEGATIVE_INFINITY, 0, 0, i.toLong(), config.blockSize.toLong() - 1 - j)
            }
        }
    }

    override fun forward(x: ITensor): ITensor {
        val (batchSize, seqLen, embedDim) = x.shape
        return cAttn(x) // x: (batch, seq, embed), cAttn: (batch, seq, embed*3)
            .use { h -> h.split(config.nEmbed, 2) } // list of 3 tensors: (batch, seq, embed)
            .let { (q, k, v) ->
                Triple(
                    k.view(batchSize, config.nHeads.toLong(), seqLen, (config.nEmbed / config.nHeads).toLong()),
                    q.view(batchSize, config.nHeads.toLong(), seqLen, (config.nEmbed / config.nHeads).toLong()),
                    v.view(batchSize, config.nHeads.toLong(), seqLen, (config.nEmbed / config.nHeads).toLong())
                )
            }
            .let { (q, k, v) ->
                Pair(
                    q.matmul(k, false, true) * (1.0f / sqrt(k.shape.last().toFloat())),
                    v
                ).also {
                    q.close()
                    k.close()
                }
            }
            .use { att, v ->
                Pair(att.plus(attnMask), v)
            }
            .use { att, v ->
                Pair(att.softmax(3), v)
            }
            .use { att, v ->
                Pair(attnDropout(att), v)
            }
            .use { att, v ->
                att.matmul(v)
            }
            .use { y ->
                y.view(batchSize, seqLen, config.nEmbed.toLong())
            }
            .use { y ->
                cProj(y)
            }
            .use { y ->
                residDropout(y)
            }
    }

    override fun subModules(): List<IModule> {
        return listOf(cAttn, cProj, attnDropout, residDropout)
    }

}

class MLP(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val cFc = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed * 4L, true)
    private val cProj = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong() * 4L, config.nEmbed.toLong(), true)
    private val dropout = Dropout(sciCore, config.dropout)

    override fun forward(input: ITensor): ITensor {
        return cFc(input)
            .use { h -> h.relu() } // TODO: use gelu
            .use { h -> cProj(h) }
            .use { h -> dropout(h) }
    }

    override fun subModules(): List<IModule> {
        return listOf(cFc, cProj, dropout)
    }

}

class Block(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val ln1 = LayerNorm(sciCore, 3)
    private val attn = CausalSelfAttention(sciCore, config)
    private val ln2 = LayerNorm(sciCore, 3)
    private val mlp = MLP(sciCore, config)

    override fun forward(input: ITensor): ITensor {
        return ln1(input)
            .use { h -> attn(h) }
            .use { h -> h.plus(input) }
            .use { h -> ln2(h) }
            .use { h -> mlp(h) }
            .use { h -> h.plus(h) }
    }

}

class GPT(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val wte = sciCore.gaussian(DataType.FLOAT32, config.vocabSize.toLong(), config.nEmbed.toLong())
    private val wpe = sciCore.gaussian(DataType.FLOAT32, config.blockSize.toLong(), config.nEmbed.toLong())
    private val dropout = Dropout(sciCore, config.dropout)
    private val blocks = (0 until config.nLayers).map { Block(sciCore, config) }
    private val lnFin = LayerNorm(sciCore, 3)

    override fun forward(input: ITensor): ITensor {
        return input
            .use { x -> x.matmul(wte) }
            .use { x -> x.plus(wpe) }
            .use { x -> dropout(x) }
            .use { x ->
                blocks.fold(x) { acc, block ->
                    block(acc)
                }
            }
            .use { x -> lnFin(x) }
    }

    override fun subModules(): List<IModule> {
        return listOf(dropout, *blocks.toTypedArray(), lnFin)
    }

    override fun parameters(): List<ITensor> {
        return super.parameters()
            .toMutableList()
            .apply {
                add(wte)
                add(wpe)
            }
    }
}