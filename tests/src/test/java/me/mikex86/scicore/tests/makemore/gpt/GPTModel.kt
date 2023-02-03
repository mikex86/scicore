package me.mikex86.scicore.tests.makemore.gpt

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Dropout
import me.mikex86.scicore.nn.layers.LayerNorm
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor
import me.mikex86.scicore.tensor.LazyTensor
import me.mikex86.scicore.tensor.get
import me.mikex86.scicore.utils.use
import kotlin.math.sqrt


data class GPTConfig(
    val vocabSize: Int,
    val blockSize: Int = 1024,
    val nLayers: Int = 12,
    val nHeads: Int = 12,
    val nEmbed: Int = 768,
    val dropout: Float = 0.0f,
)

class CausalSelfAttention(private val sciCore: ISciCore, private val config: GPTConfig) : IModule {

    private val cAttn = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed * 3L, true)

    private val cProj = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.nEmbed.toLong(), true)

    private val attnDropout = Dropout(sciCore, config.dropout)

    private val residDropout = Dropout(sciCore, config.dropout)

    init {
        if (config.nEmbed % config.nHeads != 0) {
            throw IllegalArgumentException("Embedding size must be divisible by the number of heads")
        }
    }

    override fun forward(x: ITensor): ITensor {
        val (batchSize, seqLen, embedDim) = x.shape
        var attnV: ITensor? = null
        return cAttn(x) // x: (batch, seq, embed), cAttn: (3, batch, seq, embed)
            .use { h -> h.view(3, batchSize, seqLen, embedDim) }
            .use { h -> listOf(h[0], h[1], h[2]) } // list of 3 tensors each with (batch, seq, embed)
            .use { (q, k, v) ->
                Triple(
                    q.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads),
                    k.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads),
                    v.view(batchSize, seqLen, config.nHeads.toLong(), embedDim / config.nHeads)
                )
            }
            .use { q, k, v ->
                Triple(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2)
                )
            }
            .use { q, k, v ->
                Triple(
                    // flatten as 3d tensor for matmul, where the last dimension represents batchSize * seqLen
                    q.flatten(0),
                    k.flatten(0),
                    v.flatten(0)
                )
            }
            .use { q, k, v ->
                Pair(q, k).use { _, _ ->
                    ((q.matmul(k, false, true) * (1.0f / sqrt(k.shape.last().toFloat()))) as LazyTensor)
                }.also {
                    attnV = v
                }
            }
            .use { att ->
                // triangle mask for causal attention
                val attnMaskMul = sciCore.triangle(DataType.FLOAT32, seqLen, seqLen, 0.0, 1.0)
                val attnMaskAdd = sciCore.triangle(DataType.FLOAT32, seqLen, seqLen, Double.NEGATIVE_INFINITY, 0.0)
                attnMaskMul.use {
                    attnMaskAdd.use {
                        att * attnMaskMul + attnMaskAdd
                    }
                }
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
                att.matmul(attnV!!)
            }
            .use { att ->
                att.view(batchSize, config.nHeads.toLong(), seqLen, embedDim / config.nHeads)
            }
            .use { y ->
                y.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim)
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
            .use { h -> h.relu() }
            .use { h -> cProj(h) }
            .use { h -> dropout(h) }
    }

    override fun subModules(): List<IModule> {
        return listOf(cFc, cProj, dropout)
    }

}

class Block(sciCore: ISciCore, config: GPTConfig) : IModule {

    private val ln1 = LayerNorm(sciCore, config.nEmbed)
    private val attn = CausalSelfAttention(sciCore, config)
    private val ln2 = LayerNorm(sciCore, config.nEmbed)
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

    override fun subModules(): List<IModule> {
        return listOf(ln1, attn, ln2, mlp)
    }

}

class GPTModel(private val sciCore: ISciCore, config: GPTConfig) : IModule {

    private val wte = sciCore.gaussian(DataType.FLOAT32, config.vocabSize.toLong(), config.nEmbed.toLong())
    private val wpe = sciCore.gaussian(DataType.FLOAT32, config.blockSize.toLong(), config.nEmbed.toLong())
    private val dropout = Dropout(sciCore, config.dropout)
    private val blocks = (0 until config.nLayers).map { Block(sciCore, config) }
    private val lnFin = LayerNorm(sciCore, config.nEmbed)
    private val lmHead = Linear(sciCore, DataType.FLOAT32, config.nEmbed.toLong(), config.vocabSize.toLong(), false)

    override fun forward(input: ITensor): ITensor {
        return input
            .use { x -> wte[x] }
            .use { x ->
                sciCore
                    .arange(0, input.shape[1], 1, DataType.INT64)
                    .use { posIdx -> posIdx.view(1, input.shape[1]) }
                    .use { pos ->
                        wpe[pos].use { posEmb ->
                            x.plus(posEmb)
                        }
                    }
            }
            .use { x -> dropout(x) }
            .use { x ->
                blocks.fold(x) { acc, block ->
                    acc.use {
                        block(acc)
                    }
                }
            }
            .use { x -> lnFin(x) }
            .use { x -> lmHead(x) }
    }

    override fun subModules(): List<IModule> {
        return listOf(dropout, *blocks.toTypedArray(), lnFin, lmHead)
    }

    override fun parameters(): List<ITensor> {
        return super.parameters()
            .toMutableList()
            .apply {
                add(0, wpe)
                add(0, wte)
            }
    }
}