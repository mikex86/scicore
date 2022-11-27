package me.mikex86.scicore.ranges

val IntRange.Companion.ALL: IntRange
    get() = IntRange(Int.MIN_VALUE, Int.MAX_VALUE)

val LongRange.Companion.ALL: LongRange
    get() = LongRange(Long.MIN_VALUE, Long.MAX_VALUE)