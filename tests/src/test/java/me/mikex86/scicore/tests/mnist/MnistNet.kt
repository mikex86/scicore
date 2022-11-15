package me.mikex86.scicore.tests.mnist

import me.mikex86.scicore.ISciCore
import me.mikex86.scicore.nn.IModule
import me.mikex86.scicore.nn.layers.Linear
import me.mikex86.scicore.nn.act.ReLU
import me.mikex86.scicore.nn.layers.Softmax
import me.mikex86.scicore.tensor.DataType
import me.mikex86.scicore.tensor.ITensor

class MnistNet(sciCore: ISciCore) : IModule {

    private val act = ReLU()
    private val fc1 = Linear(sciCore, DataType.FLOAT32, (28 * 28).toLong(), 128, true)
    private val fc2 = Linear(sciCore, DataType.FLOAT32, 128, 10, true)
    private val softmax = Softmax(sciCore, 1)

    override fun forward(input: ITensor): ITensor {
        return fc1(input)
            .use { h -> act(h) }
            .use { h -> fc2(h) }
            .use { h -> softmax(h) }
    }

    override fun subModules(): List<IModule> {
        return listOf(fc1, fc2)
    }
}