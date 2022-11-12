package me.mikex86.scicore.graph

fun IGraphRecorder.scopedRecording(recording: () -> Unit) {
    recordWithScope {
        recording()
        null
    }
}