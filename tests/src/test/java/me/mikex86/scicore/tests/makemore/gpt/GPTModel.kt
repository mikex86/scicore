package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Dropout
import me.mikex86.scicore.nn.layers.LayerNorm
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tests.makemore.bigram.sciCore
import me.mikex86.scicore.utils.use
import kotlin.math.PI
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

    override fun forward(x: ITensor): ITensor {
        val (batchSize, seqLen, embedDim) = x.shape
        var attnV: ITensor
        return cAttn(x) // x: (batch, seq, embed), cAttn: (batch, seq, embed*3)
            .use { h -> h.split(config.nEmbed, 2) } // list of 3 tensors: (batch, seq, embed)
            .use { (q, k, v) ->
                Triple(
                    q.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads),
                    k.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads),
                    v.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads)
                )
            }
            .let { (q, k, v) ->
                Triple(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2)
                )
            }
            .let { (q, k, v) ->
                Triple(
                    q.view(batchSize * config.nHeads, seqLen, embedDim / config.nHeads),
                    k.view(batchSize * config.nHeads, seqLen, embedDim / config.nHeads),
                    v.view(batchSize * config.nHeads, seqLen, embedDim / config.nHeads)
                )
            }
            .let { (q, k, v) ->
                Pair(q, k).use { _, _ ->
                    q.matmul(k, false, true) * (1.0f / sqrt(k.shape.last().toFloat()))
                }.also {
                    attnV = v
                }
            }
            .use { att ->
                val attnMaskMul = sciCore.zeros(DataType.FLOAT32, seqLen, seqLen)
                val attnMaskAdd = sciCore.zeros(DataType.FLOAT32, seqLen, seqLen)
                // triangle mask for causal attention
                for (i in 0 until seqLen) {
                    for (j in 0..i) {
                        attnMaskMul.setFloat(1f, i, j)
                    }
                    for (j in i + 1 until seqLen) {
                        attnMaskAdd.setFloat(Float.NEGATIVE_INFINITY, i, j)
                    }
                }
                att * attnMaskMul + attnMaskAdd
            }
            .use { att ->
                att.softmax(2)
            }
            .use { att ->
                att.view(batchSize * config.nHeads, seqLen, seqLen)
            }
            .use { att ->
                attnDropout(att)
            }
            .use { att ->
                att.matmul(attnV)
            }
            .use { att ->
                att.view(batchSize, config.nHeads.toLong(), seqLen, embedDim / config.nHeads)
            }
            .use { y ->
                y.transpose(1, 2).view(batchSize, seqLen, embedDim)
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


private fun gelu(x: ITensor): ITensor {
    return x * 0.5f * (((x * 0.044715f * x.pow(3.0f)) * sqrt(2.0f / PI.toFloat())).tanh() + 1.0f)
}

class MLP(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val cFc = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed * 4L, true)
    private val cProj = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong() * 4L, config.nEmbed.toLong(), true)
    private val dropout = Dropout(sciCore, config.dropout)

    override fun forward(input: ITensor): ITensor {
        return cFc(input)
            .use { h -> gelu(h) }
            .use { h -> cProj(h) }
            .use { h -> dropout(h) }
    }

    override fun subModules(): List<IModule> {
        return listOf(cFc, cProj, dropout)
    }

}

class Block(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val ln1 = LayerNorm(sciCore, 2)
    private val attn = CausalSelfAttention(sciCore, config)
    private val ln2 = LayerNorm(sciCore, 2)
    private val mlp = MLP(sciCore, config)

    override fun forward(input: ITensor): ITensor {
        var residualH: ITensor
        return ln1(input)
            .use { h -> attn(h) }
            .use { h -> h.plus(input) }
            .also { h -> residualH = h }
            .use { h -> ln2(h) }
            .use { h -> mlp(h) }
            .use { h -> h.plus(residualH) }
    }

}

class GPTModel(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val wte = sciCore.gaussian(DataType.FLOAT32, config.vocabSize.toLong(), config.nEmbed.toLong())
    private val wpe = sciCore.gaussian(DataType.FLOAT32, config.blockSize.toLong(), config.nEmbed.toLong())
    private val dropout = Dropout(sciCore, config.dropout)
    private val blocks = (0 until config.nLayers).map { Block(sciCore, config) }
    private val lnFin = LayerNorm(sciCore, 2)
    private val lmHead = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.vocabSize.toLong(), false)

    override fun forward(input: ITensor): ITensor {
        return input
            .use { x -> wte[x] }
            .use { x ->
                sciCore
                    .arange(0, input.shape[1], 1, DataType.INT64)
                    .view(1, input.shape[1])
                    .use { pos ->
                        x.plus(wpe[pos])
                    }
            }
            .use { x -> dropout(x) }
            .use { x ->
                blocks.fold(x) { acc, block ->
                    block(acc)
                }
            }
            .use { x -> lnFin(x) }
            .use { x -> lmHead(x) }
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