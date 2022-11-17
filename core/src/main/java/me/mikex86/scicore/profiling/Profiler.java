package me.mikex86.scicore.profiling;

import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.Map;

public class Profiler {

    public static final boolean USE_PROFILER = false;

    @NotNull
    private static final Map<String, Long> totalTimeSpentInSection = new HashMap<>();

    @NotNull
    private static final Map<String, Long> currentlyRunningSections = new HashMap<>();

    public static void startSection(String sectionName) {
        long currentTime = System.nanoTime();
        currentlyRunningSections.put(sectionName, currentTime);
    }

    public static void endSection(String sectionName) {
        long currentTime = System.nanoTime();
        Long started = currentlyRunningSections.get(sectionName);
        if (started == null) {
            throw new IllegalStateException("Section ended that was never started: \"" + sectionName + "\"");
        }
        long elapsed = currentTime - started;
        totalTimeSpentInSection.compute(sectionName, (k, time) -> (time == null ? 0 : time) + elapsed);
    }

    public static void printStats() {
        for (Map.Entry<String, Long> entry : totalTimeSpentInSection.entrySet()) {
            String sectionName = entry.getKey();
            long elapsed = entry.getValue();
            System.out.println(sectionName + ": " + (elapsed / 1e9) + "s");
        }
    }
}
